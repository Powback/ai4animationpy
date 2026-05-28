"""
Boxing Controller Pilot Training Script
========================================
Pilot-trains CodebookMatching on 37 boxer NPZs.
Goal: validate the training loop, not chase quality.

Features match Biped runtime (Program.py) exactly:
  InputDim  = 441  (bone positions/rotations/velocities + root trajectory + guidance)
  OutputDim = 352  (per-step: root delta + pose + rotations + velocities + contacts + guidance)
  SequenceLength = 16 @ 30fps over 0.5s window

Usage:
  /tmp/boxing_train_venv/bin/python boxing_train.py

Outputs (relative to fightergame/output/controller_design/training/):
  boxer_controller.pt        — final checkpoint
  loss_curves.png            — training loss curves
  loss_history.npy           — raw loss data
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
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --- Paths -----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
AI4ANIM_DIR = SCRIPT_DIR.parent.parent.parent  # AI4AnimationsPy/
sys.path.insert(0, str(AI4ANIM_DIR))

NPZ_DIR = SCRIPT_DIR.parent.parent / "_ASSETS_/Geno/Motions"
OUTPUT_DIR = Path("/Users/macback/Projects/fightergame/output/controller_design/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from ai4animation.AI.Networks.CodebookMatching import Model as CodebookMatching

# ---------------------------------------------------------------------------
# Hyperparameters — pilot scale for CPU training
# ---------------------------------------------------------------------------
FRAMERATE        = 30
SEQUENCE_LENGTH  = 16     # frames in output window
SEQUENCE_WINDOW  = 0.5    # seconds (= 16 / 30 ≈ 0.533)
BATCH_SIZE       = 32
EPOCHS           = 100

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
CONTACT_HEIGHT  = 0.08  # meters
CONTACT_VEL     = 0.5   # m/s max for contact

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
    Compute root transform at frame t.
    Returns (root_pos [3], root_mat [3,3]) where columns of root_mat are [right, up, forward].
    Matches AI4Animation RootModule: root projected to XZ plane, forward from arm cross-product.
    """
    hips = positions[t, IDX_HIPS]
    root_pos = np.array([hips[0], 0.0, hips[2]])

    # Forward direction: cross(left_arm - right_arm, up) → matches RootModule
    left_arm  = positions[t, IDX_LEFT_ARM]
    right_arm = positions[t, IDX_RIGHT_ARM]
    diff = left_arm - right_arm
    diff[1] = 0.0  # project to XZ
    up = np.array([0.0, 1.0, 0.0])
    forward = np.cross(diff, up)
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / norm

    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)

    # Columns: [right, up, forward] → orthonormal basis
    root_mat = np.stack([right, up, forward], axis=-1)  # [3, 3]
    return root_pos, root_mat


def to_root_local(world_pos: np.ndarray, root_pos: np.ndarray, root_mat: np.ndarray) -> np.ndarray:
    """World positions → root-local positions. root_mat columns are axes."""
    root_inv = root_mat.T  # orthogonal matrix
    return (world_pos - root_pos) @ root_inv  # [..., 3]


def rot_to_root_local(world_rot: np.ndarray, root_mat: np.ndarray) -> np.ndarray:
    """World rotation matrices → root-local rotation matrices. R_local = root_inv @ R_world"""
    root_inv = root_mat.T
    return np.einsum("ij,bjk->bik", root_inv, world_rot)  # [B, 3, 3]


# ---------------------------------------------------------------------------
# Feature extraction — produces one (input, output) training pair per frame
# ---------------------------------------------------------------------------

def extract_sample(positions: np.ndarray, quaternions: np.ndarray, t: int,
                   clip_mean_pos: np.ndarray) -> tuple | None:
    """
    Extract (input_vec [441], output_seq [16, 352]) for frame t.
    Returns None if frame is out of range.

    input_vec layout (441 total, matches Biped Program.py):
      [0:69]    positions  [23,3]
      [69:138]  axis-Z     [23,3]
      [138:207] axis-Y     [23,3]
      [207:276] velocities [23,3]
      [276:308] future root pos XZ  [16,2]
      [308:340] future root dir XZ  [16,2]
      [340:372] future root vel XZ  [16,2]
      [372:441] guidance positions  [23,3]

    output per step (352 total):
      [0:3]     root delta [3] (y=0)
      [3:72]    bone positions [23,3]
      [72:210]  bone rotations 6DOF [23,6] = [axis_z, axis_y] per bone
      [210:279] bone velocities [23,3]
      [279:283] contacts [4]
      [283:352] guidance positions [23,3]
    """
    F = positions.shape[0]
    if t < 1 or t + SEQUENCE_LENGTH > F:
        return None

    root_pos, root_mat = compute_root_transform(positions, t)
    root_inv = root_mat.T

    # Current frame features
    rot_mats = quat_to_matrix(quaternions[t])  # [23, 3, 3]
    local_pos = to_root_local(positions[t], root_pos, root_mat)  # [23, 3]
    local_rot = rot_to_root_local(rot_mats, root_mat)  # [23, 3, 3]
    axis_z = local_rot[:, :, 2]  # third column [23, 3]
    axis_y = local_rot[:, :, 1]  # second column [23, 3]

    # Velocity (finite diff)
    vel_world = (positions[t] - positions[t - 1]) * FRAMERATE  # [23, 3]
    local_vel = vel_world @ root_inv  # [23, 3]

    # Future root trajectory XZ (16 frames starting at t)
    root_xz_pos  = np.zeros((SEQUENCE_LENGTH, 2))
    root_xz_dir  = np.zeros((SEQUENCE_LENGTH, 2))
    root_xz_vel  = np.zeros((SEQUENCE_LENGTH, 2))

    # 2D rotation matrix (XZ plane) to bring world XZ into root-local XZ
    R2d = np.array([[root_mat[0, 0], root_mat[2, 0]],
                    [root_mat[0, 2], root_mat[2, 2]]])

    for i in range(SEQUENCE_LENGTH):
        ti = t + i
        future_hips = positions[ti, IDX_HIPS]
        rel_xz = np.array([future_hips[0] - root_pos[0], future_hips[2] - root_pos[2]])
        root_xz_pos[i] = R2d @ rel_xz

        # Future forward direction
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

        # Future velocity
        if ti > 0:
            fv = (positions[ti, IDX_HIPS] - positions[ti - 1, IDX_HIPS]) * FRAMERATE
        else:
            fv = np.zeros(3)
        root_xz_vel[i] = R2d @ np.array([fv[0], fv[2]])

    # Guidance = clip mean pose projected to current root frame
    guidance_local = to_root_local(clip_mean_pos, root_pos, root_mat)  # [23, 3]

    # Assemble input vector [441]
    input_vec = np.concatenate([
        local_pos.flatten(),    # 69
        axis_z.flatten(),       # 69
        axis_y.flatten(),       # 69
        local_vel.flatten(),    # 69
        root_xz_pos.flatten(),  # 32
        root_xz_dir.flatten(),  # 32
        root_xz_vel.flatten(),  # 32
        guidance_local.flatten(), # 69
    ]).astype(np.float32)

    assert input_vec.shape[0] == INPUT_DIM, f"input dim mismatch: {input_vec.shape[0]} != {INPUT_DIM}"

    # Output sequence [16, 352]
    output_frames = []
    for i in range(SEQUENCE_LENGTH):
        ti = t + i

        # Root delta in root-local XZ frame
        root_delta_xz = root_xz_pos[i]  # [2]
        root_delta = np.array([root_delta_xz[0], 0.0, root_delta_xz[1]])  # [3]

        # Future bone features
        future_local_pos = to_root_local(positions[ti], root_pos, root_mat)  # [23, 3]

        future_rot_mats = quat_to_matrix(quaternions[ti])  # [23, 3, 3]
        future_local_rot = rot_to_root_local(future_rot_mats, root_mat)  # [23, 3, 3]
        future_az = future_local_rot[:, :, 2]  # [23, 3]
        future_ay = future_local_rot[:, :, 1]  # [23, 3]
        # 6DOF per bone: [az, ay] interleaved → [23, 6]
        future_rot_6dof = np.concatenate([future_az, future_ay], axis=-1)  # [23, 6]

        if ti > 0:
            fvel = (positions[ti] - positions[ti - 1]) * FRAMERATE
        else:
            fvel = np.zeros_like(positions[ti])
        future_local_vel = fvel @ root_inv  # [23, 3]

        # Foot contacts
        contacts = np.zeros(4, dtype=np.float32)
        for j, ci in enumerate(CONTACT_INDICES):
            y_world = positions[ti, ci, 1]
            if ti > 0:
                vel_mag = np.linalg.norm(positions[ti, ci] - positions[ti - 1, ci]) * FRAMERATE
            else:
                vel_mag = 0.0
            contacts[j] = 1.0 if (y_world < CONTACT_HEIGHT and vel_mag < CONTACT_VEL) else 0.0

        frame_out = np.concatenate([
            root_delta,               # 3
            future_local_pos.flatten(),   # 69
            future_rot_6dof.flatten(),    # 138
            future_local_vel.flatten(),   # 69
            contacts,                     # 4
            guidance_local.flatten(),     # 69
        ]).astype(np.float32)

        assert frame_out.shape[0] == OUTPUT_DIM, f"output dim mismatch: {frame_out.shape[0]} != {OUTPUT_DIM}"
        output_frames.append(frame_out)

    return input_vec, np.array(output_frames, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_clips(npz_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    files = sorted(glob.glob(str(npz_dir / "boxer_*.npz")))
    print(f"Found {len(files)} boxer NPZ files")
    clips = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        pos = d["positions"].astype(np.float32)  # [F, 23, 3]
        # BUG: NPZs store XYZW (scipy default) but quat_to_matrix expects WXYZ.
        # This pilot loads them as-is, so rotation matrices are WRONG.
        # The network learned consistent-but-wrong features and still converged.
        # FIXED in boxing_train_v2.py: np.concatenate([q[...,3:4], q[...,:3]], axis=-1)
        quat = d["quaternions"].astype(np.float32)  # [F, 23, 4] — XYZW, not WXYZ!
        clips.append((pos, quat))
    return clips


def build_dataset(clips: list) -> list[tuple[np.ndarray, np.ndarray]]:
    samples = []
    total_frames = 0
    for clip_idx, (pos, quat) in enumerate(clips):
        F = pos.shape[0]
        total_frames += F
        # Clip mean pose (used as guidance)
        clip_mean_pos = pos.mean(axis=0)  # [23, 3]
        valid = 0
        for t in range(1, F - SEQUENCE_LENGTH):
            result = extract_sample(pos, quat, t, clip_mean_pos)
            if result is not None:
                samples.append(result)
                valid += 1
    print(f"Total frames: {total_frames}, training samples: {len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(samples: list, device: torch.device, args) -> dict:
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

    optim_prior = torch.optim.AdamW(
        list(model.Encoder.parameters()) + list(model.Decoder.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    optim_matcher = torch.optim.AdamW(model.Estimator.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optim_denoiser = torch.optim.AdamW(model.Denoiser.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Simple cosine decay schedulers
    scheduler_prior    = torch.optim.lr_scheduler.CosineAnnealingLR(optim_prior,    T_max=EPOCHS)
    scheduler_matcher  = torch.optim.lr_scheduler.CosineAnnealingLR(optim_matcher,  T_max=EPOCHS)
    scheduler_denoiser = torch.optim.lr_scheduler.CosineAnnealingLR(optim_denoiser, T_max=EPOCHS)

    N = len(samples)
    batches_per_epoch = max(1, N // BATCH_SIZE)
    indices = list(range(N))

    history = {"recon": [], "match": [], "denoise": [], "epoch": []}
    t0 = time.time()

    print(f"\nTraining CodebookMatching on {N} samples")
    print(f"  Epochs={EPOCHS}, BatchSize={BATCH_SIZE}, Batches/epoch={batches_per_epoch}")
    print(f"  InputDim={INPUT_DIM}, OutputDim={OUTPUT_DIM}, SeqLen={SEQUENCE_LENGTH}")
    print(f"  Encoder/Estimator/Decoder dim={ENCODER_DIM}, CodebookChannels={CODEBOOK_CHANNELS}x{CODEBOOK_DIMS}")
    print(f"  Device: {device}\n")

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(indices)
        epoch_recon = epoch_match = epoch_denoise = 0.0
        update_stats = epoch == 1

        for b in range(batches_per_epoch):
            batch_idx = indices[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
            if len(batch_idx) < 2:
                continue

            x_list = [samples[i][0] for i in batch_idx]
            y_list = [samples[i][1] for i in batch_idx]
            x = torch.tensor(np.stack(x_list), dtype=torch.float32, device=device)
            y = torch.tensor(np.stack(y_list), dtype=torch.float32, device=device)

            _, losses = model.learn(x, y, update_stats and b == 0)

            optim_prior.zero_grad()
            optim_matcher.zero_grad()
            optim_denoiser.zero_grad()

            # Separate backward passes with retain_graph for shared activations
            losses["Reconstruction Loss"].backward(retain_graph=True)
            losses["Matching Loss"].backward(retain_graph=True)
            losses["Denoising Loss"].backward()

            # Gradient clipping
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
        eta = elapsed / epoch * (EPOCHS - epoch)

        history["epoch"].append(epoch)
        history["recon"].append(avg_recon)
        history["match"].append(avg_match)
        history["denoise"].append(avg_denoise)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Recon={avg_recon:.4f}  Match={avg_match:.4f}  Denoise={avg_denoise:.4f} | "
                  f"Elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

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
    plt.suptitle("BoxingController Training — CodebookMatching Pilot")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=120)
    plt.close()
    np.save(output_dir / "loss_history.npy", history)
    print(f"Saved loss curves to {output_dir / 'loss_curves.png'}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, clips: list, output_dir: Path, device: torch.device) -> None:
    """
    Run a 3-second inference sample:
    - Pick the first clip, starting at frame 30 (allow warm-up)
    - Autoregressively predict 90 frames (3s @ 30fps = 6 prediction steps at 10Hz, but
      for visual evaluation we just run forward on sequential inputs from the clip)
    - Save resulting bone positions as NPZ for matplotlib preview
    """
    model.eval()
    model.to(device)

    # Use the first clip for inference
    pos, quat = clips[0]
    clip_mean_pos = pos.mean(axis=0)
    F = pos.shape[0]

    # Collect autoregressive output for 90 frames
    output_positions = []  # list of [23, 3] world positions per frame
    output_contacts  = []  # list of [4] contacts per frame

    START_FRAME = min(30, F // 4)
    n_steps = min(6, (F - START_FRAME) // SEQUENCE_LENGTH)  # number of prediction steps

    print(f"\nRunning inference: {n_steps} prediction steps × {SEQUENCE_LENGTH} frames")

    root_pos_world = pos[START_FRAME, IDX_HIPS].copy()
    root_pos_world[1] = 0.0
    root_mat_world = compute_root_transform(pos, START_FRAME)[1]

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
        # y: [1, 16, 352]
        y_np = y.squeeze(0).cpu().numpy()  # [16, 352]

        root_pos, root_mat = compute_root_transform(pos, t)
        root_inv = root_mat.T

        for i in range(SEQUENCE_LENGTH):
            # Root delta (first 3 values)
            root_delta = y_np[i, 0:3]  # [3]
            # Bone positions relative to root (next 69 values)
            bone_pos_local = y_np[i, 3:72].reshape(23, 3)
            # Contacts (skip to offset 279..283)
            contacts = y_np[i, 279:283]

            # Convert back to world space using the original root (not auto-regressive)
            bone_pos_world = bone_pos_local @ root_mat.T + root_pos  # [23, 3]

            output_positions.append(bone_pos_world)
            output_contacts.append(contacts)

    if not output_positions:
        print("Warning: no inference frames generated")
        return

    out_positions = np.array(output_positions)  # [N, 23, 3]
    out_contacts  = np.array(output_contacts)   # [N, 4]

    bone_names = np.array(["Hips","LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase",
                            "RightUpLeg","RightLeg","RightFoot","RightToeBase",
                            "Spine","Spine1","Spine2","Spine3","Neck","Head",
                            "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
                            "RightShoulder","RightArm","RightForeArm","RightHand"])

    npz_path = output_dir / "inference_sample.npz"
    np.savez(str(npz_path),
             positions=out_positions,
             contacts=out_contacts,
             bone_names=bone_names,
             framerate=np.float32(FRAMERATE),
             source_clip=str(clips[0]),
             start_frame=np.int32(START_FRAME))
    print(f"Saved inference sample ({len(output_positions)} frames) to {npz_path}")


# ---------------------------------------------------------------------------
# Preview render (matplotlib)
# ---------------------------------------------------------------------------

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # left leg
    (0, 5), (5, 6), (6, 7), (7, 8),        # right leg
    (0, 9), (9, 10), (10, 11), (11, 12),   # spine
    (12, 13), (13, 14),                     # neck + head
    (12, 15), (15, 16), (16, 17), (17, 18),# left arm
    (12, 19), (19, 20), (20, 21), (21, 22),# right arm
]


def render_preview(npz_path: Path, output_dir: Path) -> None:
    d = np.load(str(npz_path), allow_pickle=True)
    positions = d["positions"]  # [F, 23, 3]
    F = positions.shape[0]
    n_frames_to_show = min(6, F)
    indices = [int(i * (F - 1) / (n_frames_to_show - 1)) for i in range(n_frames_to_show)]

    fig, axes = plt.subplots(2, n_frames_to_show, figsize=(3 * n_frames_to_show, 8),
                              subplot_kw={"projection": "3d"})
    if n_frames_to_show == 1:
        axes = axes.reshape(2, 1)

    for col, fi in enumerate(indices):
        pos = positions[fi]  # [23, 3]
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

    plt.suptitle("BoxingController Inference Preview — Pilot", fontsize=11)
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
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only run inference on existing checkpoint")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    clips = load_all_clips(NPZ_DIR)
    assert len(clips) == 37, f"Expected 37 boxer NPZs, got {len(clips)}"

    checkpoint_path = OUTPUT_DIR / "boxer_controller.pt"

    if not args.skip_train:
        samples = build_dataset(clips)
        model, history, total_time = train(samples, device, args)

        # Save checkpoint
        torch.save(model, str(checkpoint_path))
        print(f"\nSaved checkpoint to {checkpoint_path}")
        print(f"Total training time: {total_time/60:.1f} minutes")

        # Save loss curves
        save_loss_curves(history, OUTPUT_DIR)

        # Report final losses
        print(f"\nFinal epoch losses:")
        print(f"  Reconstruction: {history['recon'][-1]:.4f}")
        print(f"  Matching:       {history['match'][-1]:.4f}")
        print(f"  Denoising:      {history['denoise'][-1]:.4f}")

    else:
        print(f"Loading checkpoint from {checkpoint_path}")
        model = torch.load(str(checkpoint_path), weights_only=False)

    # Run inference
    run_inference(model, clips, OUTPUT_DIR, device)

    # Render preview
    inference_npz = OUTPUT_DIR / "inference_sample.npz"
    if inference_npz.exists():
        render_preview(inference_npz, OUTPUT_DIR)
    else:
        print("No inference_sample.npz found, skipping preview")

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
