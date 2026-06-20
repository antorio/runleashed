"""
Math helpers for the LivePortrait expression restorer, ported from the
reference implementations:
  - KwaiVGI/LivePortrait  src/utils/camera.py  (headpose_pred_to_degree,
    get_rotation_matrix) and src/live_portrait_wrapper.py (transform_keypoint)
  - facefusion/processors/live_portrait.py (limit_expression) and
    expression_restorer.py (apply_restore blend)

Pure numpy, no torch/scipy, so it can run under onnxruntime only.
"""
import numpy as np
import cv2

PI = np.pi

# Expression clamp range. The previous flat +/-0.028 squashed strong driving
# expressions (e.g. a wide-open mouth) down to almost nothing, which is why the
# restorer looked like a near no-op. The clamp magnitude is now controlled by
# roop.globals.expression_clamp; 0 (default) disables clamping so the target
# expression passes through in full. A small positive value (e.g. 0.1) re-enables
# a gentle clamp if strong expressions ever cause artifacts.
def _clamp_mag():
    import roop.globals
    try:
        return float(getattr(roop.globals, 'expression_clamp', 0.0) or 0.0)
    except Exception:
        return 0.0

# Semantic groups within LivePortrait's 21 implicit keypoints (0-indexed).
# These index sets are the de-facto mapping used across the LivePortrait
# community (AdvancedLivePortrait / FaceFusion face editor). They are
# empirically derived, not officially labelled, so they can be tuned. With all
# three areas enabled the union covers the expressive keypoints, i.e. full
# restoration, which is robust regardless of the exact partition.
EXP_IDX_EYES = [11, 13, 15, 16, 18]
EXP_IDX_MOUTH = [6, 12, 14, 17, 19, 20]
EXP_IDX_BROWS = [1, 2]


def headpose_pred_to_degree(pred):
    """Decode a head-pose prediction to degrees.

    Handles both ONNX export variants: raw 66-bin logits (softmax expectation
    * 3 - 97.5) or an already-decoded scalar/degree.
    """
    pred = np.asarray(pred, dtype=np.float32)
    flat = pred.reshape(pred.shape[0], -1) if pred.ndim > 1 else pred.reshape(1, -1)
    if flat.shape[1] == 66:
        idx = np.arange(66, dtype=np.float32)
        e = np.exp(flat - flat.max(axis=1, keepdims=True))
        soft = e / e.sum(axis=1, keepdims=True)
        return (soft * idx).sum(axis=1) * 3.0 - 97.5
    return flat.reshape(-1)


def get_rotation_matrix(pitch_, yaw_, roll_):
    """Return per-batch 3x3 rotation matrices for use as `kp @ R`.

    Mirrors LivePortrait camera.get_rotation_matrix exactly, including the final
    transpose (permute(0,2,1)).
    """
    pitch = np.atleast_1d(np.asarray(pitch_, np.float32)).reshape(-1) / 180.0 * PI
    yaw = np.atleast_1d(np.asarray(yaw_, np.float32)).reshape(-1) / 180.0 * PI
    roll = np.atleast_1d(np.asarray(roll_, np.float32)).reshape(-1) / 180.0 * PI
    bs = pitch.shape[0]
    ones = np.ones(bs, np.float32)
    zeros = np.zeros(bs, np.float32)

    rot_x = np.stack([ones, zeros, zeros,
                      zeros, np.cos(pitch), -np.sin(pitch),
                      zeros, np.sin(pitch), np.cos(pitch)], axis=1).reshape(bs, 3, 3)
    rot_y = np.stack([np.cos(yaw), zeros, np.sin(yaw),
                      zeros, ones, zeros,
                      -np.sin(yaw), zeros, np.cos(yaw)], axis=1).reshape(bs, 3, 3)
    rot_z = np.stack([np.cos(roll), -np.sin(roll), zeros,
                      np.sin(roll), np.cos(roll), zeros,
                      zeros, zeros, ones], axis=1).reshape(bs, 3, 3)
    rot = rot_z @ rot_y @ rot_x
    return np.transpose(rot, (0, 2, 1))


def limit_expression(expression):
    mag = _clamp_mag()
    if mag <= 0.0:
        return expression
    return np.clip(expression, -mag, mag)


def transform_motion_points(motion_points, rotation, expression, scale, translation):
    """scale * (kp @ R + exp) + t   (LivePortrait Eqn.2 / FaceFusion apply_restore)

    motion_points: (1, 21, 3); rotation: (1, 3, 3); expression: (1, 21, 3);
    scale: (1,1) or scalar; translation: (1,3) or (1,1,3).
    """
    kp = motion_points.reshape(1, -1, 3).astype(np.float32)
    exp = expression.reshape(1, -1, 3).astype(np.float32)
    out = kp @ rotation + exp
    out = out * np.asarray(scale, np.float32).reshape(1, 1, 1)
    t = np.asarray(translation, np.float32).reshape(1, -1)
    out[:, :, 0] += t[0, 0]
    out[:, :, 1] += t[0, 1]
    return out.astype(np.float32)


def build_applied_expression(temp_exp, target_exp, factor, use_eyes, use_mouth, use_brows):
    """Blend the driving (target) expression onto the source, then restrict to
    the selected areas -- matching FaceFusion's pattern exactly:

        blended = target*factor + temp*(1-factor)        # all keypoints
        for each area NOT selected: blended[area_idx] = temp[area_idx]

    So with all three areas on, the full expression is restored. With an area
    off, those keypoints keep the source (swapped) expression unchanged.
    """
    temp = temp_exp.reshape(1, -1, 3).astype(np.float32)
    targ = target_exp.reshape(1, -1, 3).astype(np.float32)
    # blend as: source + (target - source) * factor * power.
    # `power` (roop.globals.expression_power) lets you over-drive subtle
    # expressions so they become clearly visible; 1.0 = natural target amount.
    import roop.globals
    power = float(getattr(roop.globals, 'expression_power', 1.0) or 1.0)
    amt = float(factor) * power
    blended = temp + (targ - temp) * amt

    n = blended.shape[1]

    def reset(idxs):
        ids = [i for i in idxs if 0 <= i < n]
        if ids:
            blended[:, ids, :] = temp[:, ids, :]

    if not use_eyes:
        reset(EXP_IDX_EYES)
    if not use_mouth:
        reset(EXP_IDX_MOUTH)
    if not use_brows:
        reset(EXP_IDX_BROWS)

    return limit_expression(blended)


def prepare_crop(crop_bgr, size=256):
    """BGR uint8 crop -> (1,3,size,size) float32 RGB in [0,1] (LivePortrait input)."""
    if crop_bgr.shape[0] != size or crop_bgr.shape[1] != size:
        crop_bgr = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA)
    x = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0     # BGR -> RGB
    x = np.clip(x, 0.0, 1.0)
    x = np.transpose(x, (2, 0, 1))[None, ...]                # NCHW
    return np.ascontiguousarray(x, dtype=np.float32)


def parse_output(out):
    """generator output (1,3,H,W) float [0,1] -> HxWx3 uint8 BGR."""
    img = np.asarray(out)
    if img.ndim == 4:
        img = img[0]
    img = np.transpose(img, (1, 2, 0))      # CHW -> HWC (RGB)
    img = np.clip(img, 0.0, 1.0) * 255.0
    img = img.astype(np.uint8)[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(img)
