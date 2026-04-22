"""Drive Go1Puppeteer with real MANN mocap and compare tracking error
with vs without integral correction.

Runs two passes over the same quadruped mocap clip:
  (a) baseline (ki=0): pure feed-forward
  (b) integral: ki_thigh/ki_calf = 1.5

Reports per-joint MAE (reference - actual), split by startup window vs
steady-state. Also prints the final integral values.
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, '/app/Demos/Go1')
MOCAP_PATH = '/app/Demos/_ASSETS_/Quadruped/Motions/D1_001_KAN01_001.npz'


def load_frames():
    d = np.load(MOCAP_PATH, allow_pickle=True)
    names = [str(x) for x in d['bone_names']]
    pos = d['positions']  # (T, 27, 3)
    return names, pos


def name_to_pos_frame(names, pos_frame):
    return {n: np.asarray(pos_frame[i], dtype=np.float32)
            for i, n in enumerate(names)}


def run(ki_thigh, ki_calf, n_frames=600):
    os.environ['GO1_KI_HIP'] = '0.0'
    os.environ['GO1_KI_THIGH'] = str(ki_thigh)
    os.environ['GO1_KI_CALF'] = str(ki_calf)
    os.environ['GO1_INTEGRAL_LIMIT'] = '0.3'
    import go1_puppeteer as gp
    importlib.reload(gp)
    p = gp.Go1Puppeteer()
    print(f"  ki_per_joint = {p.ki_per_joint}, lim={p.integral_limit}")

    names, pos = load_frames()
    errs_pre_blend = []
    errs_post_blend = []
    integrals_last = None
    for i in range(n_frames):
        frame_bones = name_to_pos_frame(names, pos[i % pos.shape[0]])
        p.step(frame_bones)
        if p.has_fallen():
            print(f"  FELL at frame {i}")
            break
        err = p.tracking_error()
        if i < 50:
            errs_pre_blend.append(np.abs(err))
        else:
            errs_post_blend.append(np.abs(err))
        integrals_last = p.integral()

    pre = np.stack(errs_pre_blend) if errs_pre_blend else np.zeros((1, 12))
    post = np.stack(errs_post_blend) if errs_post_blend else np.zeros((1, 12))
    thigh_idx = [1, 4, 7, 10]
    calf_idx = [2, 5, 8, 11]
    mae_thigh_pre = float(np.mean(pre[:, thigh_idx]))
    mae_calf_pre = float(np.mean(pre[:, calf_idx]))
    mae_thigh_post = float(np.mean(post[:, thigh_idx]))
    mae_calf_post = float(np.mean(post[:, calf_idx]))
    tail_n = 60
    mae_thigh_tail = (float(np.mean(post[-tail_n:, thigh_idx]))
                      if len(post) > tail_n else None)
    mae_calf_tail = (float(np.mean(post[-tail_n:, calf_idx]))
                     if len(post) > tail_n else None)
    print(f"  thigh |err|: pre-blend={mae_thigh_pre:.4f} "
          f"post={mae_thigh_post:.4f} last-2s={mae_thigh_tail}")
    print(f"  calf  |err|: pre-blend={mae_calf_pre:.4f} "
          f"post={mae_calf_post:.4f} last-2s={mae_calf_tail}")
    if integrals_last is not None:
        print(f"  final integral: thigh="
              f"{[round(float(integrals_last[i]), 3) for i in thigh_idx]}")
        print(f"                  calf ="
              f"{[round(float(integrals_last[i]), 3) for i in calf_idx]}")
    return post


if __name__ == '__main__':
    print("=== BASELINE (ki=0) ===")
    run(0.0, 0.0)
    print()
    print("=== WITH INTEGRAL (ki=1.5 thigh/calf) ===")
    run(1.5, 1.5)
