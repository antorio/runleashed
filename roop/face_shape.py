"""Face shape from the source: move the target's jaw / chin to the source's.

inswapper replaces the identity of the inner face (eyes, nose, mouth) but keeps
the target's face shape: its jaw line and chin come through unchanged, so a
strong target chin (long, square, wide) stays visible whatever the source is.
Here the FRAME around the target face is warped before the face is aligned and
swapped, so that the target's jaw and chin already sit where the source's would
be; inswapper keeps the shape it is given, and the swapped face is pasted onto
the warped frame, so the new jaw line is continuous with the neck / background.

How:
- shapes are compared in 3D (buffalo_l's 1k3d68 landmarks: x, y and depth in
  pixels). Each face is normalised with a 3D similarity fit (Umeyama) on rigid
  inner points -- nose and eye corners -- into one reference frame, so the head
  pose drops out and only the shape is left. (A 2D fit mixes pose into shape:
  the same face turned 10 degrees already looks like a different jaw.)
- source shape: the median of the faceset's near-frontal, mouth-closed photos
  of its person, normalised; once per faceset.
- target shape: normalised per frame; in a video render it is averaged along the
  face's track (TargetTracks), so the jaw offset does not jitter.
- jaw offset = source jaw - target jaw in the reference frame; the chin part is
  taken relative to the lower lip, so the target's jaw opening (speech, a laugh)
  is kept and only the chin / jaw shape comes across. The offset is rotated back
  into the frame with the target's current pose.
- a thin-plate-spline warp moves the target's jaw points by the offset; brows,
  eyes, nose and lips are pinned (the alignment keypoints do not move), a ring
  outside the jaw is pinned, and the warp fades to exactly zero at the edge of
  the warped area, so nothing else in the frame changes.
- the effect fades out for strongly turned or tilted heads (yaw 30-45, pitch
  20-35 degrees) and on the far side of a turned face.
Strength 0 leaves the frame untouched (the module is not even imported).
"""
import os
import threading

import cv2
import numpy as np

JAW = np.arange(0, 17)
INNER = np.array([27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 42, 45])   # nose + eye corners (not the lids)
PINNED = np.arange(17, 68)                                             # brows, eyes, nose, lips
LOWER_LIP = 57
# per jaw point: the ends near the ears / hair stay, the jaw and chin move
JAW_W = np.array([0.0, 0.25, 0.55, 0.85, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.85, 0.55, 0.25, 0.0])
# how much of the lower lip's offset each jaw point discounts: 1 at the chin,
# 0 from the jaw corners up (keeps the target's mouth opening)
FOLLOW_LIP = np.clip(1.0 - np.abs(np.arange(17) - 8) / 5.0, 0.0, 1.0)
MAX_SHIFT = 0.12       # largest jaw move, share of the jaw width
RING = 1.6             # pinned ring: this far from the nose, relative to the jaw
MIN_JACOBIAN = 0.3     # the warp must not fold (or even come close)
# source photos used for the source shape: near-frontal with the mouth closed
# (the jaw of a turned face is partly the model's guess; an open mouth or a
# laugh moves the chin); the other photos only fill up to SHAPE_MIN_PHOTOS
SHAPE_YAW, SHAPE_PITCH = 25.0, 20.0
MOUTH_OPEN = 0.08      # inner-lip gap / outer eye-corner distance above this = open mouth
SHAPE_MIN_PHOTOS = 3
SAME_PERSON = 0.40     # cosine to the faceset's median identity (FaceSet 'robust' rule)

# insightface's meanshape_68 (the 1k3d68 pose reference) at INNER: x right,
# y down, z toward the camera
REF_INNER = np.array([
    [-0.0076, -0.3231, 0.4619], [-0.0079, -0.2557, 0.5105], [-0.0077, -0.1992, 0.5525],
    [-0.0073, -0.1426, 0.5987], [-0.1450, 0.0331, 0.4197], [-0.0843, 0.0313, 0.4732],
    [-0.0055, 0.0398, 0.5147], [0.0635, 0.0461, 0.4792], [0.1340, 0.0220, 0.4191],
    [-0.3868, -0.3134, 0.2596], [-0.1552, -0.3152, 0.3143], [0.1390, -0.3098, 0.3183],
    [0.3767, -0.3135, 0.2632]])

_lock = threading.Lock()
_warned = set()
_model = None


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(f'[face-shape] {msg}')


def umeyama(src, dst):
    """Least-squares similarity dst ~ s * R @ src + t (Umeyama 1991), R a proper
    rotation. Returns (s, R, t)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    var = (xs ** 2).sum() / len(src)
    s = (S * np.diag(D)).sum() / var if var > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def _fit(L3):
    """Similarity from the reference frame onto the face's inner points. The
    reference is tried with both depth directions and the better fit is kept,
    so the landmark model's depth sign does not matter. Returns (s, R, t, sign)."""
    best = None
    for sign in (1.0, -1.0):
        ref = REF_INNER * (1.0, 1.0, sign)
        s, R, t = umeyama(ref, L3[INNER])
        err = float((((ref @ R.T) * s + t - L3[INNER]) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, s, R, t, sign)
    return best[1:]


def canonical(L3):
    """The face's 68x3 landmarks in the reference frame (pose and size removed)."""
    s, R, t, sign = _fit(L3)
    return ((L3 - t) @ R / s) * (1.0, 1.0, sign)


def landmarks3d(face):
    """buffalo_l's 68x3 landmarks with depth, or None. When 2dfan4 replaced
    landmark_3d_68 (2D, depth 0), its buffalo copy is kept as
    landmark_3d_68_buffalo (roop/landmark68.py)."""
    for key in ('landmark_3d_68_buffalo', 'landmark_3d_68'):
        try:
            lm = face.get(key)
        except Exception:
            lm = getattr(face, key, None)
        if lm is None:
            continue
        lm = np.asarray(lm, np.float64)
        if lm.shape == (68, 3) and np.abs(lm[:, 2]).max() > 1e-3:
            return lm
    return None


def _landmark_model():
    """buffalo_l's 1k3d68 on its own, for source photos that were analysed
    without it."""
    global _model
    if _model is not None:
        return _model or None
    try:
        from roop.face_util import get_face_analyser
        analyser = get_face_analyser()
        m = analyser.models.get('landmark_3d_68') if hasattr(analyser, 'models') else None
        if m is None:
            import insightface
            import roop.globals
            from roop.utilities import resolve_relative_path
            path = os.path.join(resolve_relative_path('..'), 'models', 'buffalo_l', '1k3d68.onnx')
            providers = ['CPUExecutionProvider'] if roop.globals.CFG.force_cpu else roop.globals.execution_providers
            m = insightface.model_zoo.get_model(path, providers=providers)
            m.prepare(ctx_id=0)
        _model = m
    except Exception as e:
        _warn_once('model', f'68-point 3D landmark model unavailable ({e})')
        _model = False
    return _model or None


def source_landmarks(face, image):
    """68x3 landmarks with depth for a source face: its own, or 1k3d68 run on
    the photo it was found in."""
    lm = landmarks3d(face)
    if lm is not None or image is None:
        return lm
    model = _landmark_model()
    if model is None:
        return None
    try:
        from insightface.app.common import Face
        probe = Face(bbox=np.asarray(face['bbox'], np.float32))
        model.get(image, probe)
        return landmarks3d(probe)
    except Exception:
        return None


def photo_geometry(L3):
    """(yaw, pitch, mouth) of a face from its 68x3 landmarks: head angles in
    degrees from frontal, mouth = inner-lip gap / outer eye-corner distance."""
    s, R, t, sign = _fit(L3)
    yaw, pitch = _angles(R)
    c = (L3 - t) @ R / s
    mouth = np.linalg.norm(c[66] - c[62]) / max(np.linalg.norm(c[45] - c[36]), 1e-6)
    return float(yaw), float(pitch), float(mouth)


def shape_usable(yaw, pitch, mouth):
    return yaw <= SHAPE_YAW and pitch <= SHAPE_PITCH and mouth <= MOUTH_OPEN


def _raw_embeddings(faceset):
    """The faceset's per-photo embeddings (faces[0]'s own one is kept in
    embeddings_backup once FaceSet.AverageEmbeddings has replaced it)."""
    out = []
    backup = getattr(faceset, 'embeddings_backup', None)
    for i, f in enumerate(getattr(faceset, 'faces', [])):
        e = backup if (i == 0 and backup is not None) else f.get('embedding')
        out.append(None if e is None else np.asarray(e, np.float64).reshape(-1))
    return out


def pick_shape_photos(units, geometry):
    """Which source photos make the source shape: indices, usable ones first,
    and how many are usable. units: unit identity embeddings (or None);
    geometry: (yaw, pitch, mouth) per photo (or None: no 3D landmarks).
    A photo counts only if it shows the faceset's person (cosine >= SAME_PERSON
    to the median identity, with 3+ embeddings). Usable = near-frontal with the
    mouth closed; with fewer than SHAPE_MIN_PHOTOS usable, the closest others
    fill up."""
    valid = [u for u in units if u is not None]
    same = [True] * len(units)
    if len(valid) >= 3:
        centre = np.median(np.stack(valid), axis=0)
        centre = centre / (np.linalg.norm(centre) + 1e-12)
        same = [u is None or float(u @ centre) >= SAME_PERSON for u in units]
    scored = []
    for i, g in enumerate(geometry):
        if g is not None and same[i]:
            yaw, pitch, mouth = g
            scored.append((max(yaw / SHAPE_YAW, pitch / SHAPE_PITCH, mouth / MOUTH_OPEN), i))
    scored.sort()
    usable = sum(1 for score, _ in scored if score <= 1.0)
    return [i for _, i in scored[:max(usable, min(SHAPE_MIN_PHOTOS, len(scored)))]], usable


def source_shape(faceset):
    """The faceset's face shape in the reference frame (68x3), or None: the
    median of the photos pick_shape_photos chooses. Cached on the faceset once
    computed. A failure is not cached, so a faceset analysed later with 3D
    landmarks still gets its shape; retrying costs nothing when there is no
    landmark and no photo to run the model on."""
    shape = getattr(faceset, '_shape3d', None)
    if shape is not None:
        return shape
    with _lock:
        shape = getattr(faceset, '_shape3d', None)
        if shape is not None:
            return shape
        refs = getattr(faceset, 'ref_images', None) or []
        faces = getattr(faceset, 'faces', [])
        units = [None if e is None else e / (np.linalg.norm(e) + 1e-12) for e in _raw_embeddings(faceset)]
        lms = [source_landmarks(f, refs[i] if i < len(refs) else None) for i, f in enumerate(faces)]
        used, usable = pick_shape_photos(units, [None if lm is None else photo_geometry(lm) for lm in lms])
        if not used:
            _warn_once('nosrc', 'no 3D landmarks for the source face; face shape is off for it')
            return None
        shape = np.median(np.stack([canonical(lms[i]) for i in used]), axis=0)
        faceset._shape3d = shape
        print(f'[face-shape] source shape from {len(used)} of {len(faces)} photos '
              f'({usable} near-frontal with the mouth closed)')
        return shape


def _tps(ctrl, vals, pts):
    """Thin-plate spline through ctrl -> vals, evaluated at pts (all 2D).
    Control points closer than 1 px to an earlier one are dropped (a closed
    mouth puts inner-lip points on top of each other: a singular system)."""
    keep = [0]
    for i in range(1, len(ctrl)):
        if np.min(((ctrl[keep] - ctrl[i]) ** 2).sum(-1)) >= 1.0:
            keep.append(i)
    ctrl, vals = ctrl[keep], vals[keep]
    n = len(ctrl)
    d2 = ((ctrl[:, None, :] - ctrl[None, :, :]) ** 2).sum(-1)
    K = d2 * np.log(d2 + 1e-12)                  # r^2 log r^2, 0 at r = 0
    P = np.hstack([np.ones((n, 1)), ctrl])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
    b = np.zeros((n + 3, vals.shape[1]))
    b[:n] = vals
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    d2 = ((pts[:, None, :] - ctrl[None, :, :]) ** 2).sum(-1)
    return (d2 * np.log(d2 + 1e-12)) @ coef[:n] + coef[n] + pts @ coef[n + 1:]


def _smoothstep(v):
    v = np.clip(v, 0.0, 1.0)
    return v * v * (3.0 - 2.0 * v)


def _angles(R):
    """(yaw, pitch) in degrees from frontal: the reference frame's depth axis
    (the face's forward direction) in the frame, R its rotation into it."""
    n = np.abs(R[:, 2])
    return np.degrees(np.arctan2(n[0], n[2])), np.degrees(np.arctan2(n[1], n[2]))


def pose_gate(R):
    """1 for a face within 30 degrees of frontal (yaw) / 20 (pitch), fading to
    0 at 45 / 35. R: the reference frame's rotation into the frame."""
    yaw, pitch = _angles(R)
    return float(_smoothstep((45.0 - yaw) / 15.0) * _smoothstep((35.0 - pitch) / 15.0)), yaw


def gate_of(L3):
    """pose_gate for 68x3 landmarks (TargetTracks averages it along a track)."""
    return pose_gate(_fit(L3)[1])[0]


def jaw_offsets(face, faceset, strength):
    """Per jaw point 2D offsets in frame pixels (17x2), or None when the effect
    does not apply."""
    S = source_shape(faceset)
    if S is None:
        return None
    L3 = landmarks3d(face)
    if L3 is None:
        _warn_once('notgt', 'target faces have no 3D landmarks; face shape is off')
        return None
    s, R, t, sign = _fit(L3)
    T = face.get('shape_canon')
    if T is None:
        T = ((L3 - t) @ R / s) * (1.0, 1.0, sign)
    D = (S[JAW] - T[JAW]) - FOLLOW_LIP[:, None] * (S[LOWER_LIP] - T[LOWER_LIP])[None, :]
    d = (s * ((D * (1.0, 1.0, sign)) @ R.T))[:, :2]

    gate, yaw = pose_gate(R)
    smoothed = face.get('shape_gate')             # averaged along the track (video)
    w = float(strength) * (gate if smoothed is None else float(smoothed))
    if w <= 1e-3:
        return None
    P = np.asarray(face['landmark_3d_68'], np.float64)[:, :2]
    weights = JAW_W.copy()
    # far side of a turned face: fade out (its jaw line is behind the cheek)
    far_w = _smoothstep((35.0 - yaw) / 25.0)
    nose = P[27:36].mean(0)
    far_left = np.linalg.norm(P[0] - nose) < np.linalg.norm(P[16] - nose)
    side = np.clip(np.abs(np.arange(17) - 8) / 4.0, 0.0, 1.0)
    far = (np.arange(17) < 8) if far_left else (np.arange(17) > 8)
    weights = np.where(far, weights * (1.0 - (1.0 - far_w) * side), weights)
    d = d * (weights * w)[:, None]
    cap = MAX_SHIFT * np.linalg.norm(P[16] - P[0])
    mag = np.linalg.norm(d, axis=1, keepdims=True)
    return d * np.minimum(1.0, cap / np.maximum(mag, 1e-6))


def warp_frame(frame, face, faceset, strength):
    """The frame with the target face's jaw / chin moved toward the source's
    shape (a new array), or the frame itself when the effect does not apply."""
    if strength <= 0.0 or frame is None:
        return frame
    d = jaw_offsets(face, faceset, strength)
    if d is None or np.abs(d).max() < 0.25:
        return frame
    H, W = frame.shape[:2]
    P = np.asarray(face['landmark_3d_68'], np.float64)[:, :2]
    nose = P[27:36].mean(0)
    ring = nose + (P[JAW] - nose) * RING
    lo = np.minimum(ring.min(0), P.min(0))
    hi = np.maximum(ring.max(0), P.max(0))
    pad = np.maximum((hi - lo) * 0.15, 8.0)
    x0, y0 = np.maximum(np.floor(lo - pad).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(hi + pad).astype(int), (W, H))
    rw, rh = x1 - x0, y1 - y0
    if rw < 8 or rh < 8:
        return frame
    off = np.array([x0, y0], np.float64)
    # the displacement fades to 0 over the outer band of the area (pad wide),
    # measured from the area's own edges, clipped or not
    step = max(4.0, max(rw, rh) / 64.0)
    gx = np.arange(0.0, rw - 1 + step, step)
    gy = np.arange(0.0, rh - 1 + step, step)
    G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    fade = (_smoothstep(np.minimum(gx, rw - 1 - gx) / pad[0])[None, :] *
            _smoothstep(np.minimum(gy, rh - 1 - gy) / pad[1])[:, None])[..., None]
    for _ in range(3):
        # inverse map (output pixel -> where to sample): control points at the
        # destination, values = source positions
        ctrl = np.vstack([P[JAW] + d, P[PINNED], ring]) - off
        vals = np.vstack([P[JAW], P[PINNED], ring]) - off
        disp = (_tps(ctrl, vals, G) - G).reshape(len(gy), len(gx), 2) * fade
        mx = gx[None, :] + disp[..., 0]
        my = gy[:, None] + disp[..., 1]
        jac = (np.gradient(mx, step, axis=1) * np.gradient(my, step, axis=0) -
               np.gradient(mx, step, axis=0) * np.gradient(my, step, axis=1))
        if jac.min() >= MIN_JACOBIAN:
            break
        d = d * 0.5
    else:
        return frame
    # grid -> every pixel of the area (grid point i sits at pixel i * step)
    u, v = np.meshgrid(np.arange(rw, dtype=np.float32) / np.float32(step),
                       np.arange(rh, dtype=np.float32) / np.float32(step))
    dx = cv2.remap(disp[..., 0].astype(np.float32), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dy = cv2.remap(disp[..., 1].astype(np.float32), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # where the warp moves less than 1/20 px, keep the pixels exactly (no
    # resampling softness in the background around the face)
    still = dx * dx + dy * dy < np.float32(0.0025)
    dx[still] = 0.0
    dy[still] = 0.0
    fx, fy = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    out = frame.copy()
    out[y0:y1, x0:x1] = cv2.remap(frame, fx + dx, fy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out
