"""Go1 puppeteer — reads MANN skeleton each tick, retargets to Go1 joints,
steps MuJoCo physics, returns root pose + joint angles.

Design:
  * One instance per session (same lifecycle as the quadruped WebProgram).
  * MuJoCo stepping runs inside the existing inference thread, budgeted so
    the 30 Hz MANN frame loop stays on schedule.
  * Kinematic retarget mirrors spec-llm/src/experiments/mann_live_go1.py —
    body-frame IK with sweep-gain / zero-abduction to compensate for PD lag.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import mujoco  # type: ignore
    _HAS_MUJOCO = True
except Exception:
    _HAS_MUJOCO = False


ASSETS_DIR = Path(__file__).parent
SCENE_XML = str(ASSETS_DIR / "scene.xml")

# Mirrors spec-llm/src/experiments/mann_to_go1.py JOINT_RANGES
JOINT_RANGES = np.array([
    [-0.863,  0.863], [-0.686,  4.501], [-2.818, -0.888],  # FR
    [-0.863,  0.863], [-0.686,  4.501], [-2.818, -0.888],  # FL
    [-0.863,  0.863], [-0.686,  4.501], [-2.818, -0.888],  # RR
    [-0.863,  0.863], [-0.686,  4.501], [-2.818, -0.888],  # RL
], dtype=np.float32)

HOME_QPOS = np.array(
    [0, 0, 0.27, 1, 0, 0, 0,
     0, 0.9, -1.8,  0, 0.9, -1.8,  0, 0.9, -1.8,  0, 0.9, -1.8],
    dtype=np.float64,
)
HOME_CTRL = np.array(
    [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8],
    dtype=np.float32,
)

L_THIGH = 0.213
L_CALF = 0.213

# 30 Hz control (matches MANN server frame rate). We previously stepped 2x
# per MANN frame at SUBSTEPS=30, but inside the same inference thread we
# need to keep well under 33 ms/frame. Default to one control step with
# SUBSTEPS_DEFAULT substeps; env override available.
CONTROL_DT = 1.0 / 30.0
SUBSTEPS_DEFAULT = int(os.environ.get("GO1_SUBSTEPS", "10"))

# Per-joint integral correction: drives actual qpos toward the commanded
# reference by accumulating tracking error each frame. Different gains per
# joint type because hips (abduction) have different dynamics from the
# sagittal thigh/calf chain. Anti-windup via clipping the integral term.
KI_HIP_DEFAULT = float(os.environ.get("GO1_KI_HIP", "0.0"))
KI_THIGH_DEFAULT = float(os.environ.get("GO1_KI_THIGH", "1.5"))
KI_CALF_DEFAULT = float(os.environ.get("GO1_KI_CALF", "1.5"))
INTEGRAL_LIMIT_DEFAULT = float(os.environ.get("GO1_INTEGRAL_LIMIT", "0.3"))

# Yaw-tracking feedback: when the caller passes a MANN target yaw to step(),
# we drive Go1's base angular velocity about +Z (qvel[5]) toward it. The
# body-frame leg retarget doesn't convey MANN's world yaw, so without this
# Go1 keeps facing its initial heading while MANN turns. KP is rad/s per
# rad of yaw error, capped at MAX (rad/s) to avoid snappy over-rotation.
YAW_FEEDBACK_KP_DEFAULT = float(os.environ.get("GO1_YAW_FEEDBACK_KP", "4.0"))
YAW_FEEDBACK_MAX_DEFAULT = float(os.environ.get("GO1_YAW_FEEDBACK_MAX", "3.0"))
KI_PER_JOINT_DEFAULT = np.array([
    KI_HIP_DEFAULT, KI_THIGH_DEFAULT, KI_CALF_DEFAULT,  # FR
    KI_HIP_DEFAULT, KI_THIGH_DEFAULT, KI_CALF_DEFAULT,  # FL
    KI_HIP_DEFAULT, KI_THIGH_DEFAULT, KI_CALF_DEFAULT,  # RR
    KI_HIP_DEFAULT, KI_THIGH_DEFAULT, KI_CALF_DEFAULT,  # RL
], dtype=np.float32)

# Legs in Go1 ctrl order: (label, MANN hip bone, MANN foot bone, side_y)
LEG_MAP = [
    ("FR", "RightShoulder", "RightHand", -1),
    ("FL", "LeftShoulder",  "LeftHand",  +1),
    ("RR", "RightUpLeg",    "RightFoot", -1),
    ("RL", "LeftUpLeg",     "LeftFoot",  +1),
]
# Middle-of-leg bone used to resolve knee direction
KNEE_BONE_BY_LEG = {
    "FR": "RightForeArm",
    "FL": "LeftForeArm",
    "RR": "RightLeg",
    "RL": "LeftLeg",
}


def _compute_body_frame(hips: np.ndarray, spine1: np.ndarray) -> np.ndarray:
    """World→body rotation (rows = body axes in world coords).

    Body frame: x=forward, y=up, z=right. Based on dog's y-up convention.
    """
    fwd = spine1 - hips
    fwd[1] = 0.0
    n = float(np.linalg.norm(fwd))
    fwd = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else fwd / n
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    rn = float(np.linalg.norm(right))
    right = np.array([0.0, 0.0, 1.0]) if rn < 1e-6 else right / rn
    fwd_a = np.cross(up, right)
    return np.stack([fwd_a, up, right], axis=0)  # 3x3


def _leg_ik_knee_aware(foot_rel: np.ndarray, knee_rel: np.ndarray,
                       side_y: int) -> tuple[float, float, float]:
    """2-link IK returning (hip_abd, thigh, calf) in Go1 actuator convention.

    Matches spec-llm/src/experiments/mann_puppeteer_go1.py._leg_ik_knee_aware.
    """
    x, y, z = map(float, foot_rel)
    lateral = side_y * z
    down = -y
    fwd = x
    hip_abd = float(np.arctan2(lateral, max(down, 1e-3)))
    sag_dist_sq = down * down + fwd * fwd
    sag_dist = float(np.sqrt(sag_dist_sq))
    max_reach = (L_THIGH + L_CALF) * 0.99
    min_reach = abs(L_THIGH - L_CALF) * 1.05
    sag_dist = float(np.clip(sag_dist, min_reach, max_reach))
    sag_dist_sq = sag_dist * sag_dist
    cos_knee = (L_THIGH * L_THIGH + L_CALF * L_CALF - sag_dist_sq) \
        / (2 * L_THIGH * L_CALF)
    cos_knee = float(np.clip(cos_knee, -1.0, 1.0))
    knee_inner = float(np.arccos(cos_knee))
    calf = -(np.pi - knee_inner)
    alpha = float(np.arctan2(fwd, down))
    cos_beta = (L_THIGH * L_THIGH + sag_dist_sq - L_CALF * L_CALF) \
        / (2 * L_THIGH * max(sag_dist, 1e-6))
    cos_beta = float(np.clip(cos_beta, -1.0, 1.0))
    beta = float(np.arccos(cos_beta))
    thigh = beta - alpha
    return hip_abd, thigh, calf


def retarget_positions_to_ctrl(
    name_to_pos: dict[str, np.ndarray],
    scale: float,
    sweep_gain: float = 1.5,
    zero_abduction: bool = True,
    foot_drop_m: float = 0.0,
) -> np.ndarray:
    """Map MANN bone positions → Go1 12-DOF ctrl (in joint-angle space, rad).

    Defaults match the best walk config: --zero-abduction --sweep-gain 1.5.
    """
    hips = name_to_pos["Hips"]
    spine1 = name_to_pos["Spine1"]
    R_bw = _compute_body_frame(hips, spine1)
    ctrl = np.zeros(12, dtype=np.float32)
    for li, (label, hip_name, foot_name, side_y) in enumerate(LEG_MAP):
        hp = name_to_pos[hip_name]
        ft = name_to_pos[foot_name]
        md = name_to_pos[KNEE_BONE_BY_LEG[label]]
        foot_world = (ft - hp) * scale
        knee_world = (md - hp) * scale
        foot_body = R_bw @ foot_world
        knee_body = R_bw @ knee_world
        if sweep_gain != 1.0:
            foot_body = foot_body.copy()
            knee_body = knee_body.copy()
            foot_body[0] *= sweep_gain
            knee_body[0] *= sweep_gain
        if foot_drop_m != 0.0:
            foot_body = foot_body.copy()
            foot_body[1] -= foot_drop_m
        h, th, cf = _leg_ik_knee_aware(foot_body, knee_body, side_y)
        if zero_abduction:
            h = 0.0
        ctrl[li * 3 + 0] = h
        ctrl[li * 3 + 1] = th
        ctrl[li * 3 + 2] = cf
    return np.clip(ctrl, JOINT_RANGES[:, 0], JOINT_RANGES[:, 1])


def compute_scale(name_to_pos: dict[str, np.ndarray]) -> float:
    fl_sh = name_to_pos["LeftShoulder"]
    fr_sh = name_to_pos["RightShoulder"]
    rl_hp = name_to_pos["LeftUpLeg"]
    rr_hp = name_to_pos["RightUpLeg"]
    body_len = float(np.linalg.norm(
        ((fl_sh + fr_sh) / 2) - ((rl_hp + rr_hp) / 2)
    ))
    return 0.376 / max(body_len, 1e-6)


class Go1Puppeteer:
    """One Go1 MuJoCo instance driven per-frame by MANN skeleton positions.

    Usage:
        p = Go1Puppeteer()
        for frame in mann_frames:
            p.step(name_to_pos=frame_bones)
        root, qpos = p.root_pose(), p.joint_qpos()

    If MuJoCo is not available, Go1Puppeteer remains inert (enabled = False).
    Caller can check `puppeteer.enabled` before querying state.
    """

    def __init__(self, sweep_gain: float = 1.5, zero_abduction: bool = True,
                 foot_drop_m: float = 0.0, warmup_frames: int = 20,
                 blend_frames: int = 30, smooth_alpha: float = 0.5,
                 substeps: int | None = None,
                 ki_per_joint: np.ndarray | None = None,
                 integral_limit: float | None = None,
                 yaw_feedback_kp: float | None = None,
                 yaw_feedback_max: float | None = None):
        self.enabled = _HAS_MUJOCO
        self.sweep_gain = sweep_gain
        self.zero_abduction = zero_abduction
        self.foot_drop_m = foot_drop_m
        self.warmup_frames = warmup_frames
        self.blend_frames = blend_frames
        self.smooth_alpha = smooth_alpha
        self._scale: float | None = None
        self._frame_idx = 0
        self._smoothed = HOME_CTRL.copy()
        self._fell = False

        self.ki_per_joint = (
            KI_PER_JOINT_DEFAULT.copy() if ki_per_joint is None
            else np.asarray(ki_per_joint, dtype=np.float32).copy()
        )
        self.integral_limit = (
            INTEGRAL_LIMIT_DEFAULT if integral_limit is None
            else float(integral_limit)
        )
        self._integral = np.zeros(12, dtype=np.float32)
        self._last_error = np.zeros(12, dtype=np.float32)

        self.yaw_feedback_kp = (
            YAW_FEEDBACK_KP_DEFAULT if yaw_feedback_kp is None
            else float(yaw_feedback_kp)
        )
        self.yaw_feedback_max = (
            YAW_FEEDBACK_MAX_DEFAULT if yaw_feedback_max is None
            else float(yaw_feedback_max)
        )
        self._last_yaw_err = 0.0

        self.model = None
        self.data = None
        self.substeps = substeps or SUBSTEPS_DEFAULT
        if not self.enabled:
            return
        try:
            self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
            self.model.opt.timestep = CONTROL_DT / self.substeps
            self.data = mujoco.MjData(self.model)
            self._reset()
        except Exception as e:  # pragma: no cover — surface init errors
            print(f"[Go1Puppeteer] init failed: {e}")
            self.enabled = False

    def _reset(self) -> None:
        if not self.enabled:
            return
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._scale = None
        self._frame_idx = 0
        self._smoothed = HOME_CTRL.copy()
        self._fell = False
        self._integral[:] = 0.0
        self._last_error[:] = 0.0

    def reset(self) -> None:
        self._reset()

    @staticmethod
    def _quat_yaw_rad(quat_wxyz) -> float:
        """Yaw (rad) about MuJoCo +Z from a world-from-body quaternion."""
        w, x, y, z = (float(v) for v in quat_wxyz)
        fx = 1.0 - 2.0 * (y * y + z * z)
        fy = 2.0 * (x * y + w * z)
        return float(np.arctan2(fy, fx))

    def step(self, name_to_pos: dict[str, np.ndarray],
             mann_yaw_rad: float | None = None) -> None:
        """Retarget one MANN frame to Go1 PD ctrl and step physics.

        When mann_yaw_rad is supplied, the base angular velocity about +Z is
        driven toward (mann_yaw_rad - go1_yaw) * yaw_feedback_kp, clamped to
        ±yaw_feedback_max. This is what makes Go1 track MANN's turning —
        body-frame leg retargeting alone is yaw-invariant.
        """
        if not self.enabled:
            return
        if self._fell:
            return

        try:
            if self._scale is None and self._frame_idx >= self.warmup_frames:
                self._scale = compute_scale(name_to_pos)

            w = 0.0
            if self._scale is None:
                target = HOME_CTRL.copy()
            else:
                mann_ctrl = retarget_positions_to_ctrl(
                    name_to_pos, self._scale,
                    sweep_gain=self.sweep_gain,
                    zero_abduction=self.zero_abduction,
                    foot_drop_m=self.foot_drop_m,
                )
                blend_t = max(0, self._frame_idx - self.warmup_frames)
                w = min(1.0, blend_t / max(1, self.blend_frames))
                target = HOME_CTRL * (1 - w) + mann_ctrl * w

            self._smoothed = (
                (1 - self.smooth_alpha) * self._smoothed
                + self.smooth_alpha * target
            )

            # Integral correction is added to the reference sent to the PD
            # controller. It only accumulates once the blend to MANN targets
            # has completed — during warmup+blend the reference is still
            # slewing from HOME, and any error there is expected (not bias).
            commanded = self._smoothed + self._integral
            self.data.ctrl[:] = np.clip(
                commanded.astype(np.float64),
                JOINT_RANGES[:, 0], JOINT_RANGES[:, 1],
            )

            # Yaw tracking: override base angular velocity about world +Z.
            # Applied only after the home→MANN blend has completed so the
            # robot doesn't spin during stand-up. qvel[5] is the base's
            # z-axis angular velocity for a free joint in MuJoCo.
            apply_yaw = (
                mann_yaw_rad is not None
                and self.yaw_feedback_kp > 0.0
                and w >= 1.0
            )
            if apply_yaw:
                go1_yaw = self._quat_yaw_rad(self.data.qpos[3:7])
                err = float(mann_yaw_rad) - go1_yaw
                # Wrap to (-pi, pi] so we always turn the short way.
                err = float((err + np.pi) % (2 * np.pi) - np.pi)
                self._last_yaw_err = err
                wz = float(np.clip(
                    err * self.yaw_feedback_kp,
                    -self.yaw_feedback_max, self.yaw_feedback_max,
                ))
                self.data.qvel[5] = wz

            # Apply the yaw kick once before the substep loop — letting
            # physics integrate contact forces across substeps keeps feet
            # planted instead of dragging them along a forced angular rate.
            # Hard-overriding qvel[5] every substep felt aggressive: it
            # kept overshooting and the body wobbled tall↔short.
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)

            # Measure tracking error after physics advanced by CONTROL_DT,
            # then integrate with anti-windup clipping. Only when blend
            # completed so we're tracking MANN, not the HOME→MANN slew.
            actual = self.data.qpos[7:19].astype(np.float32)
            self._last_error = self._smoothed - actual
            if w >= 1.0:
                self._integral = np.clip(
                    self._integral + self._last_error
                    * self.ki_per_joint * CONTROL_DT,
                    -self.integral_limit,
                    self.integral_limit,
                )

            # Detect fall — terminates further stepping so the robot rests
            # on the ground (visual) instead of flailing.
            z = float(self.data.qpos[2])
            if z < 0.13:
                self._fell = True
                self._reset()
        except Exception as e:  # pragma: no cover
            print(f"[Go1Puppeteer] step error: {e}")
            self._fell = True
        finally:
            self._frame_idx += 1

    def root_pose(self) -> np.ndarray:
        """Return (7,) = [x, y, z, qw, qx, qy, qz]. Zeros if disabled."""
        if not self.enabled:
            return np.zeros(7, dtype=np.float32)
        return self.data.qpos[:7].astype(np.float32)

    def joint_qpos(self) -> np.ndarray:
        """Return (12,) joint angles in actuator order. Zeros if disabled."""
        if not self.enabled:
            return np.zeros(12, dtype=np.float32)
        return self.data.qpos[7:19].astype(np.float32)

    def has_fallen(self) -> bool:
        return self._fell

    def tracking_error(self) -> np.ndarray:
        """Return (12,) per-joint (reference - actual) from the last step."""
        return self._last_error.copy()

    def integral(self) -> np.ndarray:
        """Return (12,) accumulated integral correction in actuator order."""
        return self._integral.copy()
