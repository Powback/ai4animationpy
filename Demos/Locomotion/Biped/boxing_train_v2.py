"""
Boxing Controller V2 Training Script
======================================
Trains CodebookMatching on 84 boxer NPZs from boxer_clips_v2/ (real SMPL-X rotations).
V2 additions over pilot: 300 epochs, mirror augmentation, ONNX export, 200-step stability test.

Usage:
  /tmp/boxing_train_venv/bin/python boxing_train_v2.py

Outputs (fightergame/output/controller_design/training_v2/):
  boxer_controller_v2.pt          — final checkpoint
  boxer_controller_v2.onnx        — ONNX export (opset 17)
  loss_curves.png                 — training loss curves
  loss_history.npy                — raw loss data
  inference_sample.npz            — 3s inference sample
  inference_preview.png           — matplotlib skeleton preview
  stability_test.png              — 200-step autoregressive position plot
"""
import os
import sys
import time
import glob
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths -----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
AI4ANIM_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(AI4ANIM_DIR))

NPZ_DIR    = Path("/Users/macback/Projects/fightergame/output/controller_design/boxer_clips_v2")
OUTPUT_DIR = Path("/Users/macback/Projects/fightergame/output/controller_design/training_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from ai4animation.AI.Networks.CodebookMatching import Model as CodebookMatching

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
FRAMERATE        = 30
SEQUENCE_LENGTH  = 16
SEQUENCE_WINDOW  = 0.5
BATCH_SIZE       = 32
EPOCHS           = 300

INPUT_DIM        = 441
OUTPUT_DIM       = 352

ENCODER_DIM      = 256
ESTIMATOR_DIM    = 256
DECODER_DIM      = 256
CODEBOOK_CHANNELS = 8
CODEBOOK_DIMS    = 8
DROPOUT          = 0.1
HARD             = False

LR               = 1e-4
WEIGHT_DECAY     = 1e-4

# Bone indices (from Definitions.FULL_BODY_NAMES, 0-indexed)
IDX_HIPS         = 0
IDX_LEFT_ARM     = 16   # LeftArm (upper arm, used for root forward)
IDX_RIGHT_ARM    = 20   # RightArm
IDX_LEFT_FOOT    = 3
IDX_LEFT_TOE     = 4
IDX_RIGHT_FOOT   = 7
IDX_RIGHT_TOE    = 8

CONTACT_INDICES = [IDX_LEFT_FOOT, IDX_LEFT_TOE, IDX_RIGHT_FOOT, IDX_RIGHT_TOE]
CONTACT_HEIGHT  = 0.08
CONTACT_VEL     = 0.5

# Left/right bone pairs for mirror augmentation (FULL_BODY_NAMES indices)
MIRROR_PAIRS = [
    (1, 5),   # LeftUpLeg  ↔ RightUpLeg
    (2, 6),   # LeftLeg    ↔ RightLeg
    (3, 7),   # LeftFoot   ↔ RightFoot
    (4, 8),   # LeftToeBase ↔ RightToeBase
    (15, 19), # LeftShoulder ↔ RightShoulder
    (16, 20), # LeftArm    ↔ RightArm
    (17, 21), # LeftForeArm ↔ RightForeArm
    (18, 22), # LeftHand   ↔ RightHand
]

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """WXYZ quaternion(s) to 3x3 rotation matrices. q: [..., 4] → [..., 3, 3]"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1-2*(y**2+z**2), 2*(x*y-w*z),   2*(x*z+w*y),
        2*(x*y+w*z),   1-2*(x**2+z**2), 2*(y*z-w*x),
        2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x**2+y**2),
    ], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def compute_root_transform(positions: np.ndarray, t: int):
    """
    Root transform at frame t.
    Returns (root_pos [3], root_mat [3,3]), columns = [right, up, forward].
    """
    hips = positions[t, IDX_HIPS]
    root_pos = np.array([hips[0], 0.0, hips[2]])

    left_arm  = positions[t, IDX_LEFT_ARM]
    right_arm = positions[t, IDX_RIGHT_ARM]
    diff = left_arm - right_arm
    diff[1] = 0.0
    up = np.array([0.0, 1.0, 0.0])
    forward = np.cross(diff, up)
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / norm

    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    root_mat = np.stack([right, up, forward], axis=-1)
    return root_pos, root_mat


def to_root_local(world_pos: np.ndarray, root_pos: np.ndarray, root_mat: np.ndarray) -> np.ndarray:
    return (world_pos - root_pos) @ root_mat.T


def rot_to_root_local(world_rot: np.ndarray, root_mat: np.ndarray) -> np.ndarray:
    return np.einsum("ij,bjk->bik", root_mat.T, world_rot)


# ---------------------------------------------------------------------------
# Mirror augmentation
# ---------------------------------------------------------------------------

def mirror_clip(pos: np.ndarray, quat: np.ndarray):
    """
    Reflect clip across the YZ plane (X → -X), swap left/right bones.
    Quaternion mirror: [w, x, y, z] → [w, x, -y, -z] then swap L/R pairs.
    Verified: M@R@M where M=diag(-1,1,1) corresponds to [w, x, -y, -z] WXYZ.
    """
    pos_m  = pos.copy()
    quat_m = quat.copy()

    # Flip X axis for positions
    pos_m[:, :, 0] *= -1

    # Flip quat: negate y, z components (keep w, x)
    quat_m[:, :, 2] *= -1  # y
    quat_m[:, :, 3] *= -1  # z

    # Swap left/right bone pairs
    for l_idx, r_idx in MIRROR_PAIRS:
        pos_m[:, [l_idx, r_idx]]  = pos_m[:, [r_idx, l_idx]].copy()
        quat_m[:, [l_idx, r_idx]] = quat_m[:, [r_idx, l_idx]].copy()

    return pos_m, quat_m


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_sample(positions: np.ndarray, quaternions: np.ndarray, t: int,
                   clip_mean_pos: np.ndarray) -> tuple | None:
    """
    Extract (input_vec [441], output_seq [16, 352]) for frame t.
    Returns None if out of range.
    """
    F = positions.shape[0]
    if t < 1 or t + SEQUENCE_LENGTH > F:
        return None

    root_pos, root_mat = compute_root_transform(positions, t)
    root_inv = root_mat.T

    rot_mats  = quat_to_matrix(quaternions[t])
    local_pos = to_root_local(positions[t], root_pos, root_mat)
    local_rot = rot_to_root_local(rot_mats, root_mat)
    axis_z = local_rot[:, :, 2]
    axis_y = local_rot[:, :, 1]

    vel_world = (positions[t] - positions[t - 1]) * FRAMERATE
    local_vel = vel_world @ root_inv

    root_xz_pos = np.zeros((SEQUENCE_LENGTH, 2))
    root_xz_dir = np.zeros((SEQUENCE_LENGTH, 2))
    root_xz_vel = np.zeros((SEQUENCE_LENGTH, 2))

    R2d = np.array([[root_mat[0, 0], root_mat[2, 0]],
                    [root_mat[0, 2], root_mat[2, 2]]])

    for i in range(SEQUENCE_LENGTH):
        ti = t + i
        future_hips = positions[ti, IDX_HIPS]
        rel_xz = np.array([future_hips[0] - root_pos[0], future_hips[2] - root_pos[2]])
        root_xz_pos[i] = R2d @ rel_xz

        fl = positions[ti, IDX_LEFT_ARM]
        fr = positions[ti, IDX_RIGHT_ARM]
        diff = fl - fr
        diff[1] = 0.0
        up = np.array([0.0, 1.0, 0.0])
        fwd = np.cross(diff, up)
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm > 1e-6:
            fwd = fwd / fwd_norm
        else:
            fwd = np.array([0.0, 0.0, 1.0])
        root_xz_dir[i] = R2d @ np.array([fwd[0], fwd[2]])

        if ti > 0:
            fv = (positions[ti, IDX_HIPS] - positions[ti - 1, IDX_HIPS]) * FRAMERATE
        else:
            fv = np.zeros(3)
        root_xz_vel[i] = R2d @ np.array([fv[0], fv[2]])

    guidance_local = to_root_local(clip_mean_pos, root_pos, root_mat)

    input_vec = np.concatenate([
        local_pos.flatten(),
        axis_z.flatten(),
        axis_y.flatten(),
        local_vel.flatten(),
        root_xz_pos.flatten(),
        root_xz_dir.flatten(),
        root_xz_vel.flatten(),
        guidance_local.flatten(),
    ]).astype(np.float32)

    assert input_vec.shape[0] == INPUT_DIM

    output_frames = []
    for i in range(SEQUENCE_LENGTH):
        ti = t + i
        root_delta = np.array([root_xz_pos[i, 0], 0.0, root_xz_pos[i, 1]])

        future_local_pos = to_root_local(positions[ti], root_pos, root_mat)
        future_rot_mats  = quat_to_matrix(quaternions[ti])
        future_local_rot = rot_to_root_local(future_rot_mats, root_mat)
        future_az = future_local_rot[:, :, 2]
        future_ay = future_local_rot[:, :, 1]
        future_rot_6dof = np.concatenate([future_az, future_ay], axis=-1)

        if ti > 0:
            fvel = (positions[ti] - positions[ti - 1]) * FRAMERATE
        else:
            fvel = np.zeros_like(positions[ti])
        future_local_vel = fvel @ root_inv

        contacts = np.zeros(4, dtype=np.float32)
        for j, ci in enumerate(CONTACT_INDICES):
            y_world = positions[ti, ci, 1]
            if ti > 0:
                vel_mag = np.linalg.norm(positions[ti, ci] - positions[ti - 1, ci]) * FRAMERATE
            else:
                vel_mag = 0.0
            contacts[j] = 1.0 if (y_world < CONTACT_HEIGHT and vel_mag < CONTACT_VEL) else 0.0

        frame_out = np.concatenate([
            root_delta,
            future_local_pos.flatten(),
            future_rot_6dof.flatten(),
            future_local_vel.flatten(),
            contacts,
            guidance_local.flatten(),
        ]).astype(np.float32)

        assert frame_out.shape[0] == OUTPUT_DIM
        output_frames.append(frame_out)

    return input_vec, np.array(output_frames, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading + augmentation
# ---------------------------------------------------------------------------

def load_all_clips(npz_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    files = sorted(glob.glob(str(npz_dir / "boxer_*.npz")))
    print(f"Found {len(files)} boxer NPZ files in {npz_dir}")
    clips = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        pos  = d["positions"].astype(np.float32)
        # NPZs store XYZW quaternions (scipy default); quat_to_matrix expects WXYZ.
        # Convert: [x, y, z, w] → [w, x, y, z]
        q_xyzw = d["quaternions"].astype(np.float32)
        quat = np.concatenate([q_xyzw[..., 3:4], q_xyzw[..., :3]], axis=-1)
        clips.append((pos, quat))
    return clips


def build_dataset(clips: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Build dataset with mirror augmentation.
    Returns pre-stacked (X [N, 441], Y [N, 16, 352]) tensors.
    """
    # Double clips with mirrored copies
    augmented = []
    for pos, quat in clips:
        augmented.append((pos, quat))
        augmented.append(mirror_clip(pos, quat))
    print(f"Clips after mirror augmentation: {len(augmented)} ({len(clips)} original + {len(clips)} mirrored)")

    xs, ys = [], []
    total_frames = 0
    for pos, quat in augmented:
        F = pos.shape[0]
        total_frames += F
        clip_mean_pos = pos.mean(axis=0)
        for t in range(1, F - SEQUENCE_LENGTH):
            result = extract_sample(pos, quat, t, clip_mean_pos)
            if result is not None:
                xs.append(result[0])
                ys.append(result[1])

    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    print(f"Total frames: {total_frames}, training samples: {len(X)}")
    return X, Y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X: np.ndarray, Y: np.ndarray, device: torch.device, args) -> tuple:
    model = CodebookMatching(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        sequence_length=SEQUENCE_LENGTH,
        sequence_window=SEQUENCE_WINDOW,
        encoder_dim=ENCODER_DIM,
        estimator_dim=ESTIMATOR_DIM,
        codebook_channels=CODEBOOK_CHANNELS,
        codebook_dims=CODEBOOK_DIMS,
        decoder_dim=DECODER_DIM,
        dropout=DROPOUT,
        hard=HARD,
        plotting=0,
    ).to(device)

    optim_prior    = torch.optim.AdamW(
        list(model.Encoder.parameters()) + list(model.Decoder.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    optim_matcher  = torch.optim.AdamW(model.Estimator.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optim_denoiser = torch.optim.AdamW(model.Denoiser.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler_prior    = torch.optim.lr_scheduler.CosineAnnealingLR(optim_prior,    T_max=args.epochs)
    scheduler_matcher  = torch.optim.lr_scheduler.CosineAnnealingLR(optim_matcher,  T_max=args.epochs)
    scheduler_denoiser = torch.optim.lr_scheduler.CosineAnnealingLR(optim_denoiser, T_max=args.epochs)

    N = len(X)
    batches_per_epoch = max(1, N // BATCH_SIZE)

    # Pre-convert to float32 tensors on CPU for faster batch assembly
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()
    indices = torch.arange(N)

    history = {"recon": [], "match": [], "denoise": [], "epoch": []}
    t0 = time.time()

    print(f"\nTraining CodebookMatching V2 on {N} samples")
    print(f"  Epochs={args.epochs}, BatchSize={BATCH_SIZE}, Batches/epoch={batches_per_epoch}")
    print(f"  InputDim={INPUT_DIM}, OutputDim={OUTPUT_DIM}, SeqLen={SEQUENCE_LENGTH}")
    print(f"  Encoder/Estimator/Decoder dim={ENCODER_DIM}, Codebook={CODEBOOK_CHANNELS}x{CODEBOOK_DIMS}")
    print(f"  Device: {device}\n")

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(N)
        X_t = X_t[perm]
        Y_t = Y_t[perm]
        epoch_recon = epoch_match = epoch_denoise = 0.0
        update_stats = epoch == 1

        for b in range(batches_per_epoch):
            start = b * BATCH_SIZE
            end   = min(start + BATCH_SIZE, N)
            if end - start < 2:
                continue
            x = X_t[start:end].to(device)
            y = Y_t[start:end].to(device)

            _, losses = model.learn(x, y, update_stats and b == 0)

            optim_prior.zero_grad()
            optim_matcher.zero_grad()
            optim_denoiser.zero_grad()

            losses["Reconstruction Loss"].backward(retain_graph=True)
            losses["Matching Loss"].backward(retain_graph=True)
            losses["Denoising Loss"].backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optim_prior.step()
            optim_matcher.step()
            optim_denoiser.step()

            epoch_recon   += losses["Reconstruction Loss"].item()
            epoch_match   += losses["Matching Loss"].item()
            epoch_denoise += losses["Denoising Loss"].item()

        scheduler_prior.step()
        scheduler_matcher.step()
        scheduler_denoiser.step()

        avg_recon   = epoch_recon   / batches_per_epoch
        avg_match   = epoch_match   / batches_per_epoch
        avg_denoise = epoch_denoise / batches_per_epoch
        elapsed = time.time() - t0
        eta = elapsed / epoch * (args.epochs - epoch)

        history["epoch"].append(epoch)
        history["recon"].append(avg_recon)
        history["match"].append(avg_match)
        history["denoise"].append(avg_denoise)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Recon={avg_recon:.4f}  Match={avg_match:.4f}  Denoise={avg_denoise:.4f} | "
                  f"Elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m", flush=True)

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    return model, history, total_time


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_loss_curves(history: dict, output_dir: Path) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes,
                               ["recon", "match", "denoise"],
                               ["Reconstruction Loss", "Matching Loss", "Denoising Loss"]):
        ax.semilogy(epochs, history[key])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log)")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"BoxingController V2 — CodebookMatching ({EPOCHS} epochs, mirror aug, 84 NPZs)")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=120)
    plt.close()
    np.save(output_dir / "loss_history.npy", history)
    print(f"Saved loss curves to {output_dir / 'loss_curves.png'}")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

class BoxingControllerONNX(nn.Module):
    """Thin ONNX-traceable wrapper: normalizes input, runs Estimator→Denoiser×3→Decoder, denormalizes."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.iterations = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Model.forward() handles norm/denorm and runs the full pipeline
        y, _, _, _ = self.model(
            x,
            iterations=self.iterations,
            seed=torch.zeros(x.shape[0], self.model.LatentDim, device=x.device),
        )
        return y  # [B, SeqLen, OutputDim]


def export_onnx(model, output_dir: Path, device: torch.device) -> None:
    model.eval()
    wrapper = BoxingControllerONNX(model).to(device)
    dummy_x = torch.zeros(1, INPUT_DIM, device=device)

    onnx_path = output_dir / "boxer_controller_v2.onnx"
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_x,),
                str(onnx_path),
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        print(f"ONNX export saved to {onnx_path}")

        # Quick numerical diff: compare PT vs ONNX on random input
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            test_x = torch.randn(4, INPUT_DIM)
            with torch.no_grad():
                pt_out = wrapper(test_x).cpu().numpy()
            ort_out = sess.run(["output"], {"input": test_x.numpy()})[0]
            max_diff = float(np.abs(pt_out - ort_out).max())
            print(f"ONNX vs PT max abs diff: {max_diff:.6f}")
        except Exception as e:
            print(f"ONNX numerical check skipped: {e}")

    except Exception as e:
        print(f"ONNX export failed (model may use unsupported ops): {e}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, clips: list, output_dir: Path, device: torch.device) -> None:
    model.eval()
    pos, quat = clips[0]
    clip_mean_pos = pos.mean(axis=0)
    F = pos.shape[0]

    output_positions = []
    output_contacts  = []

    START_FRAME = min(30, F // 4)
    n_steps = min(6, (F - START_FRAME) // SEQUENCE_LENGTH)
    print(f"\nInference: {n_steps} prediction steps × {SEQUENCE_LENGTH} frames")

    for step in range(n_steps):
        t = START_FRAME + step * SEQUENCE_LENGTH
        if t >= F - SEQUENCE_LENGTH:
            break
        result = extract_sample(pos, quat, t, clip_mean_pos)
        if result is None:
            break
        x_np, _ = result
        x = torch.tensor(x_np[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            y, _, _, _ = model(x, iterations=3,
                               seed=torch.zeros(1, model.LatentDim, device=device))
        y_np = y.squeeze(0).cpu().numpy()

        root_pos, root_mat = compute_root_transform(pos, t)
        for i in range(SEQUENCE_LENGTH):
            bone_pos_local = y_np[i, 3:72].reshape(23, 3)
            contacts       = y_np[i, 279:283]
            bone_pos_world = bone_pos_local @ root_mat.T + root_pos
            output_positions.append(bone_pos_world)
            output_contacts.append(contacts)

    if not output_positions:
        print("Warning: no inference frames generated")
        return

    out_positions = np.array(output_positions)
    out_contacts  = np.array(output_contacts)

    bone_names = np.array(["Hips","LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase",
                            "RightUpLeg","RightLeg","RightFoot","RightToeBase",
                            "Spine","Spine1","Spine2","Spine3","Neck","Head",
                            "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
                            "RightShoulder","RightArm","RightForeArm","RightHand"])
    npz_path = output_dir / "inference_sample.npz"
    np.savez(str(npz_path), positions=out_positions, contacts=out_contacts,
             bone_names=bone_names, framerate=np.float32(FRAMERATE),
             start_frame=np.int32(START_FRAME))
    print(f"Saved inference sample ({len(output_positions)} frames) to {npz_path}")


# ---------------------------------------------------------------------------
# 200-step autoregressive stability test
# ---------------------------------------------------------------------------

def reconstruct_input(
    bone_pos_local: np.ndarray,  # [23, 3] predicted, root-local
    az: np.ndarray,               # [23, 3] axis-Z
    ay: np.ndarray,               # [23, 3] axis-Y
    vel_local: np.ndarray,        # [23, 3] velocities
    root_pos_world: np.ndarray,   # [3] accumulated world root pos
    root_mat: np.ndarray,         # [3, 3] current root orientation
    guidance_local: np.ndarray,   # [23, 3] clip mean in root-local
    root_vel_xz: np.ndarray,      # [2] current root velocity in root-local XZ
    root_dir_xz: np.ndarray,      # [2] current root forward dir in root-local XZ
) -> np.ndarray:
    """
    Reconstruct an input vector [441] from predicted state.
    Future root trajectory is approximated as constant-velocity extrapolation.
    """
    root_xz_pos = np.zeros((SEQUENCE_LENGTH, 2))
    root_xz_dir = np.zeros((SEQUENCE_LENGTH, 2))
    root_xz_vel = np.zeros((SEQUENCE_LENGTH, 2))

    for i in range(SEQUENCE_LENGTH):
        # Constant-velocity extrapolation
        root_xz_pos[i] = root_vel_xz * (i + 1) / FRAMERATE
        root_xz_vel[i] = root_vel_xz
        root_xz_dir[i] = root_dir_xz

    input_vec = np.concatenate([
        bone_pos_local.flatten(),
        az.flatten(),
        ay.flatten(),
        vel_local.flatten(),
        root_xz_pos.flatten(),
        root_xz_dir.flatten(),
        root_xz_vel.flatten(),
        guidance_local.flatten(),
    ]).astype(np.float32)
    return input_vec


def autoregressive_stability_test(model, clips: list, output_dir: Path, device: torch.device) -> None:
    """
    Run 200-step autoregressive inference.
    Each step feeds the first predicted frame back as the next input.
    Reports max positional deviation, NaN/Inf, and saves a hip-trajectory plot.
    """
    model.eval()
    N_STEPS = 200

    pos, quat = clips[0]
    clip_mean_pos = pos.mean(axis=0)
    F = pos.shape[0]
    START_FRAME = min(30, F // 4)

    result = extract_sample(pos, quat, START_FRAME, clip_mean_pos)
    if result is None:
        print("Stability test: cannot extract initial frame, skipping")
        return

    x_np, _ = result
    root_pos_world = np.zeros(3)
    root_mat_world = compute_root_transform(pos, START_FRAME)[1]
    guidance_local = to_root_local(clip_mean_pos, np.zeros(3), np.eye(3))

    hip_traj = []  # accumulated world-space hip positions
    y_heights = []

    nan_at = None
    inf_at = None

    print(f"\n200-step autoregressive stability test...")
    for step in range(N_STEPS):
        x = torch.tensor(x_np[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            y, _, _, _ = model(x, iterations=3,
                               seed=torch.zeros(1, model.LatentDim, device=device))
        y0 = y.squeeze(0)[0].cpu().numpy()  # [352] — first predicted frame

        # Check stability
        if np.any(np.isnan(y0)):
            nan_at = step
            print(f"  NaN detected at step {step}!")
            break
        if np.any(np.isinf(y0)):
            inf_at = step
            print(f"  Inf detected at step {step}!")
            break

        # Extract predicted state
        root_delta   = y0[0:3]            # [3], y=0
        bone_pos_loc = y0[3:72].reshape(23, 3)
        rot_6dof     = y0[72:210].reshape(23, 6)  # [az, ay] per bone
        vel_loc      = y0[210:279].reshape(23, 3)

        az = rot_6dof[:, 0:3]  # [23, 3]
        ay = rot_6dof[:, 3:6]  # [23, 3]

        # Accumulate root position in world space
        root_delta_world = root_mat_world @ root_delta  # [3]
        root_pos_world += root_delta_world

        hip_traj.append(root_pos_world.copy())
        y_heights.append(bone_pos_loc[0, 1])  # hips Y in root-local (≈ height above ground)

        # Root velocity XZ in root-local frame (from root_delta XZ)
        dt = 1.0 / FRAMERATE
        root_vel_xz = np.array([root_delta[0], root_delta[2]]) / dt if dt > 0 else np.zeros(2)
        # Root forward direction from current root_mat projected to XZ
        fwd_world = root_mat_world[:, 2]  # third column = forward
        root_dir_xz = np.array([fwd_world[0], fwd_world[2]])
        norm = np.linalg.norm(root_dir_xz)
        if norm > 1e-6:
            root_dir_xz /= norm
        else:
            root_dir_xz = np.array([0.0, 1.0])

        # Reconstruct next input
        x_np = reconstruct_input(
            bone_pos_loc, az, ay, vel_loc,
            root_pos_world, root_mat_world,
            guidance_local, root_vel_xz, root_dir_xz,
        )

        # Update root orientation from predicted forward dir
        new_fwd = np.array([fwd_world[0] + root_delta[0] * 0.1,
                             0.0,
                             fwd_world[2] + root_delta[2] * 0.1])
        norm = np.linalg.norm(new_fwd)
        if norm > 1e-6:
            new_fwd /= norm
        new_right = np.cross(np.array([0., 1., 0.]), new_fwd)
        new_right /= (np.linalg.norm(new_right) + 1e-8)
        root_mat_world = np.stack([new_right, np.array([0., 1., 0.]), new_fwd], axis=-1)

    hip_traj   = np.array(hip_traj)    # [steps, 3]
    y_heights  = np.array(y_heights)

    completed_steps = len(hip_traj)
    stable = nan_at is None and inf_at is None

    if completed_steps > 0:
        xz_drift = np.linalg.norm(hip_traj[:, [0, 2]] - hip_traj[0:1, [0, 2]], axis=-1)
        max_xz   = float(xz_drift.max())
        max_y    = float(np.abs(y_heights).max())
        print(f"  Steps completed: {completed_steps}/{N_STEPS}")
        print(f"  Max XZ drift: {max_xz:.3f} m")
        print(f"  Max hip Y (root-local): {max_y:.3f} m")
        print(f"  Stable: {stable}")

        # Plot hip trajectory
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        steps = np.arange(completed_steps)

        axes[0].plot(hip_traj[:, 0], hip_traj[:, 2])
        axes[0].set_title("Hip XZ trajectory (top view)")
        axes[0].set_xlabel("X (m)")
        axes[0].set_ylabel("Z (m)")
        axes[0].axis("equal")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, xz_drift)
        axes[1].set_title("XZ drift from start")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Distance (m)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, y_heights)
        axes[2].set_title("Hip Y (height above root, root-local)")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Y (m)")
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f"200-step Autoregressive Stability — {'STABLE' if stable else 'UNSTABLE'}")
        plt.tight_layout()
        plt.savefig(output_dir / "stability_test.png", dpi=120)
        plt.close()
        print(f"Saved stability plot to {output_dir / 'stability_test.png'}")
    else:
        print(f"  No steps completed (NaN/Inf at step {nan_at or inf_at})")


# ---------------------------------------------------------------------------
# Preview render
# ---------------------------------------------------------------------------

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (12, 13), (13, 14),
    (12, 15), (15, 16), (16, 17), (17, 18),
    (12, 19), (19, 20), (20, 21), (21, 22),
]


def render_preview(npz_path: Path, output_dir: Path) -> None:
    d = np.load(str(npz_path), allow_pickle=True)
    positions = d["positions"]
    F = positions.shape[0]
    n_frames_to_show = min(6, F)
    indices = [int(i * (F - 1) / max(n_frames_to_show - 1, 1)) for i in range(n_frames_to_show)]

    fig, axes = plt.subplots(2, n_frames_to_show, figsize=(3 * n_frames_to_show, 8),
                              subplot_kw={"projection": "3d"})
    if n_frames_to_show == 1:
        axes = axes.reshape(2, 1)

    for col, fi in enumerate(indices):
        pos = positions[fi]
        for row, (x_ax, y_ax, z_ax, title) in enumerate([
            (0, 1, 2, "Front (XY)"),
            (0, 2, 1, "Side (XZ)"),
        ]):
            ax = axes[row, col]
            ax.scatter(pos[:, x_ax], pos[:, z_ax], pos[:, y_ax], s=15, c="steelblue", zorder=5)
            for e0, e1 in SKELETON_EDGES:
                ax.plot([pos[e0, x_ax], pos[e1, x_ax]],
                        [pos[e0, z_ax], pos[e1, z_ax]],
                        [pos[e0, y_ax], pos[e1, y_ax]], "k-", lw=1, alpha=0.6)
            ax.set_title(f"f{fi} {title}", fontsize=7)
            ax.set_xlabel("X", fontsize=6)
            ax.set_ylabel("Z", fontsize=6)
            ax.set_zlabel("Y", fontsize=6)
            ax.tick_params(labelsize=5)

    plt.suptitle("BoxingController V2 — Inference Preview (300 epochs, mirror aug)", fontsize=11)
    plt.tight_layout()
    out_path = output_dir / "inference_preview.png"
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    print(f"Saved inference preview to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    print(f"Device: {device}")

    clips = load_all_clips(NPZ_DIR)
    if len(clips) == 0:
        raise RuntimeError(f"No boxer_*.npz files found in {NPZ_DIR}")

    checkpoint_path = OUTPUT_DIR / "boxer_controller_v2.pt"

    if not args.skip_train:
        X, Y = build_dataset(clips)
        model, history, total_time = train(X, Y, device, args)

        torch.save(model, str(checkpoint_path))
        print(f"\nSaved checkpoint to {checkpoint_path}")
        print(f"Total training time: {total_time/60:.1f} minutes")

        save_loss_curves(history, OUTPUT_DIR)

        final_recon = history["recon"][-1]
        print(f"\nFinal epoch losses:")
        print(f"  Reconstruction: {final_recon:.4f}  (target < 0.18)")
        print(f"  Matching:       {history['match'][-1]:.4f}")
        print(f"  Denoising:      {history['denoise'][-1]:.4f}")
        if final_recon < 0.18:
            print("  QUALITY TARGET MET: Recon < 0.18 ✓")
        else:
            print(f"  WARNING: Recon {final_recon:.4f} >= 0.18 target")

        # ONNX export
        export_onnx(model, OUTPUT_DIR, device)
    else:
        print(f"Loading checkpoint from {checkpoint_path}")
        model = torch.load(str(checkpoint_path), weights_only=False)

    # Inference sample
    run_inference(model, clips, OUTPUT_DIR, device)

    # Preview
    inference_npz = OUTPUT_DIR / "inference_sample.npz"
    if inference_npz.exists():
        render_preview(inference_npz, OUTPUT_DIR)

    # 200-step stability test
    autoregressive_stability_test(model, clips, OUTPUT_DIR, device)

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
