"""Jitter probe — run on YOUR real video to locate the flicker source.

    python tools_jitter_probe.py /path/to/clip.mp4 --frames 200 --hi-landmarker

It measures, per stage, how much each quantity moves frame-to-frame AFTER
removing smooth real motion (it fits and subtracts a short moving average, so a
panning head doesn't count as jitter). High residual on a stage = that stage is
the flicker source. This replaces guessing.

Stages measured for the primary (largest) face:
  bbox_raw      detector bounding box
  kps_raw       detector 5 keypoints
  lm68_raw      68-pt landmarks (buffalo_l or 2dfan4 if --hi-landmarker)
  M_raw         alignment matrix from RAW landmarks (no smoothing)
  M_smoothed    alignment matrix AFTER the LandmarkStabilizer
The last two together tell you how much smoothing actually removes.
"""
import sys, os, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roop.globals as G
from settings import Settings
G.CFG = Settings('config.yaml')

from roop.face_util import (get_all_faces_multi, landmark_68_to_5,
                            align_crop, align_crop_robust)
from roop.landmark68 import refine_faces_landmark68
from roop.face_stabilizer import LandmarkStabilizer
from roop.align_conditioner import Align5Conditioner


def largest(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))


def residual_jitter(series, win=5):
    """Std of (signal - moving_average): removes real motion, keeps flicker.
    Works for any per-frame shape: (4,) bboxes, (5,2) kps, (68,2) landmarks."""
    a = np.asarray(series, dtype=np.float64)
    if len(a) < win + 2:
        return float('nan')
    flat = a.reshape(len(a), -1)
    k = np.ones(win) / win
    pad = win // 2
    fp = np.pad(flat, ((pad, pad), (0, 0)), mode='edge')
    ma = np.stack([np.convolve(fp[:, j], k, 'valid') for j in range(flat.shape[1])], axis=1)
    resid = flat - ma[:len(flat)]
    return float(resid.std(0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--hi-landmarker', action='store_true')
    ap.add_argument('--strength', type=float, default=0.95)
    ap.add_argument('--deadzone', type=float, default=0.012)
    ap.add_argument('--motion-frac', type=float, default=0.05)
    ap.add_argument('--size', type=int, default=256)
    args = ap.parse_args()

    G.use_hi_landmarker = args.hi_landmarker
    G.use_landmark_alignment = True
    G.landmark_sanity_gate = True

    # Use the GPU when available (a CPU run is fine, just slow).
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            G.execution_providers = ['CUDAExecutionProvider']
    except Exception:
        pass

    # --hi-landmarker must actually test 2dfan4: fetch it from the HF repo if
    # this session doesn't have it yet (the probe doesn't run pre_check).
    if args.hi_landmarker:
        import shutil as _sh
        import roop.utilities as _util
        mdir = _util.resolve_relative_path('../models')
        p2 = os.path.join(mdir, '2dfan4.onnx')
        if not os.path.exists(p2):
            print('[probe] 2dfan4.onnx missing -> fetching from antorio/runleashed-models ...')
            from huggingface_hub import hf_hub_download
            os.makedirs(mdir, exist_ok=True)
            _sh.copyfile(hf_hub_download(repo_id='antorio/runleashed-models', filename='2dfan4.onnx'), p2)

    cap = cv2.VideoCapture(args.video)
    sz = args.size
    fc = np.array([[0, 0], [sz, 0], [0, sz], [sz, sz], [sz//2, sz//2]], np.float64)

    bbox_raw, kps_raw, lm68_raw = [], [], []
    Mraw_pts, Msm_pts, Mpipe_pts = [], [], []
    gate_trips = 0

    st = LandmarkStabilizer(strength=args.strength, deadzone_frac=args.deadzone,
                            motion_frac=args.motion_frac)
    # The SAME conditioner class process_face uses -> the numbers below measure
    # the real render path (align5 smoothing + gate v2), not a probe replica.
    cond = Align5Conditioner(
        alpha_min=getattr(G, 'align5_alpha_min', 0.12),
        motion_frac=getattr(G, 'align5_motion_frac', 0.04),
        gate_thr=getattr(G, 'landmark_sanity_threshold', 0.08),
        smoothing=True, gate=True)

    n = 0
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        faces = get_all_faces_multi(frame)
        f = largest(faces)
        if f is None:
            continue

        # RAW (pre-smoothing) snapshot
        refine_faces_landmark68(frame, [f])   # fills lm68 (hi or buffalo)
        bb = np.asarray(f.bbox, float).copy()
        kp = np.asarray(f.kps, float).copy()
        l68 = np.asarray(f.landmark_3d_68, float).copy()
        bbox_raw.append(bb)
        kps_raw.append(kp)
        lm68_raw.append(l68[:, :2])

        # M from RAW landmarks
        lmk5 = landmark_68_to_5(l68)
        try:
            _, Mr = align_crop_robust(frame, lmk5, sz)
        except Exception:
            _, Mr = align_crop(frame, kp, sz)
        # map crop corners into frame space -> position jitter of the paste
        A = Mr[:, :2]; b = Mr[:, 2]
        inv = np.linalg.inv(A)
        Mraw_pts.append((fc[:, :2] - b) @ inv.T)

        # Now run the FULL smoothing pipeline (bbox->refine->landmark) on a fresh
        # detection object so the stabilizer sees a proper sequence.
        st.stabilize([f], refine_fn=lambda: refine_faces_landmark68(frame, [f]))
        l68s = np.asarray(f.landmark_3d_68, float)
        lmk5s = landmark_68_to_5(l68s)
        try:
            _, Ms = align_crop_robust(frame, lmk5s, sz)
        except Exception:
            _, Ms = align_crop(frame, np.asarray(f.kps, float), sz)
        A = Ms[:, :2]; b = Ms[:, 2]
        inv = np.linalg.inv(A)
        Msm_pts.append((fc[:, :2] - b) @ inv.T)

        # FULL render path: stabilizer output -> align5 conditioner -> fit.
        bb_s = np.asarray(f.bbox, float)
        fsize_s = max(bb_s[2]-bb_s[0], bb_s[3]-bb_s[1]) + 1e-6
        pts5, use_lmk = cond.condition(lmk5s, np.asarray(f.kps, float), fsize_s)
        try:
            if use_lmk:
                _, Mp = align_crop_robust(frame, pts5, sz)
            else:
                _, Mp = align_crop(frame, np.asarray(f.kps, float), sz)
        except Exception:
            _, Mp = align_crop(frame, np.asarray(f.kps, float), sz)
        A = Mp[:, :2]; b = Mp[:, 2]
        inv = np.linalg.inv(A)
        Mpipe_pts.append((fc[:, :2] - b) @ inv.T)

        # gate check
        kps5 = kp[:5]
        fsize = max(bb[2]-bb[0], bb[3]-bb[1])+1e-6
        d = np.linalg.norm(lmk5 - kps5, axis=1)/fsize
        if d.mean() > 0.08 or d.max() > 0.16:
            gate_trips += 1
        n += 1

    cap.release()
    if n < 10:
        print(f"only {n} faces found; need a clearer clip"); return

    print(f"\nprofiled {n} frames  (hi_landmarker={args.hi_landmarker}, "
          f"strength={args.strength}, deadzone={args.deadzone}, motion_frac={args.motion_frac})")
    print(f"{'stage':<14}{'residual jitter (px)':>22}")
    print(f"{'bbox_raw':<14}{residual_jitter(bbox_raw):>22.3f}")
    print(f"{'kps_raw':<14}{residual_jitter(kps_raw):>22.3f}")
    print(f"{'lm68_raw':<14}{residual_jitter(lm68_raw):>22.3f}")
    print(f"{'M_raw(paste)':<14}{residual_jitter(Mraw_pts):>22.3f}")
    print(f"{'M_smoothed':<14}{residual_jitter(Msm_pts):>22.3f}")
    print(f"{'M_pipeline':<14}{residual_jitter(Mpipe_pts):>22.3f}   <- the ACTUAL render path (align5 + gate v2)")
    print(f"\nraw disagreement above the OLD fixed threshold: {gate_trips}/{n} frames "
          f"({100*gate_trips/n:.1f}%)  [info only]")
    print(f"gate v2: fallback frames {cond.n_fallback}/{cond.n_frames} "
          f"({100*cond.n_fallback/max(cond.n_frames,1):.1f}%), state flips {cond.n_flips}, "
          f"learned baseline {0 if cond.baseline is None else round(cond.baseline,3)}")
    r0 = residual_jitter(Mraw_pts); r1 = residual_jitter(Msm_pts)
    if r0 == r0 and r1 == r1 and r0 > 1e-6:
        print(f"smoothing removes {100*(1-r1/r0):.0f}% of the paste jitter "
              f"({r0:.3f} -> {r1:.3f} px)")
    r2 = residual_jitter(Mpipe_pts)
    if r0 == r0 and r2 == r2 and r0 > 1e-6:
        print(f"full pipeline removes {100*(1-r2/r0):.0f}% of the paste jitter "
              f"({r0:.3f} -> {r2:.3f} px)")
    print("\nread: M_pipeline is what the render actually uses. If M_pipeline is "
          "low but you still see flicker, the source is AFTER alignment "
          "(enhancer / mask / ER / swap texture).")


if __name__ == '__main__':
    main()
