import numpy as np
from threading import Lock


class LandmarkStabilizer:
    """Reduces frame-to-frame landmark jitter in video.

    Per detected face we keep a small track (matched by bbox centre / size) and
    blend the current landmarks with the track's smoothed history using an
    *adaptive* weight: when the face is nearly still we smooth hard (kills
    jitter), when it moves fast we follow the current detection (no lag). This
    is an EMA with motion-gated alpha -- a lightweight One-Euro-style filter
    that is tolerant of frames arriving slightly out of order.

    Smoothing is applied to kps (5x2), landmark_2d_106 and the xy of
    landmark_3d_68 (z is left untouched). Identity/embedding are never changed.
    """

    def __init__(self, strength: float = 0.7, motion_frac: float = 0.05,
                 match_frac: float = 0.6, max_age: int = 8,
                 deadzone_frac: float = 0.0):
        # strength in [0,1]; higher = smoother. Maps to a floor on alpha.
        # The floor used to be clamped at 0.08, which silently capped the slider:
        # strength 0.92 and 0.98 both produced alpha_min = 0.08, so raising the
        # slider past ~0.92 did nothing at all. Clamped at 0.02 now, so the top
        # of the slider really is smoother (0.95 -> 0.05, 0.98 -> 0.02).
        self.alpha_min = float(np.clip(1.0 - strength, 0.02, 1.0))
        self.motion_frac = motion_frac
        self.match_frac = match_frac
        self.max_age = max_age
        # soft dead-zone: keypoint motion below deadzone_frac*face_size is treated
        # as detector noise and absorbed (alpha -> 0, i.e. the smoothed landmarks
        # freeze), which kills the "landmarks wobble while the head is still"
        # jitter. Real motion above the dead-zone is followed as before. 0 = off
        # (identical to the previous behaviour).
        self.deadzone_frac = max(0.0, float(deadzone_frac))
        self.tracks = []
        self._lock = Lock()

    def reset(self):
        with self._lock:
            self.tracks = []

    def _match(self, center, size):
        best, best_d = None, 1e18
        for t in self.tracks:
            d = float(np.hypot(*(center - t['center'])))
            if d < self.match_frac * size and d < best_d:
                best_d, best = d, t
        return best

    @staticmethod
    def _get(face, key):
        try:
            v = face[key]
        except Exception:
            v = getattr(face, key, None)
        return None if v is None else np.asarray(v)

    def stabilize(self, faces, refine_fn=None):
        """Smooth a frame's faces.

        `refine_fn` (optional, no args) is called in the MIDDLE of this method:
        after the bboxes have been smoothed but before the landmarks are. That
        ordering matters a lot for the hi-accuracy landmarker (2dfan4): it builds
        its 256x256 input crop straight from the bbox

            scale = 195 / max(bbox_w, bbox_h)
            tx,ty = (256 - (bbox_min+bbox_max) * scale) * 0.5

        so a jittering detector bbox makes the crop jitter, and a landmark CNN is
        not perfectly equivariant to input translation/scale -- slightly
        different framing gives non-linearly shifted predictions. The detector's
        bbox wobble therefore gets AMPLIFIED into landmark wobble before any
        smoothing can act on it (and the bbox is least stable at non-frontal
        poses, which is exactly where the swap looked most fragile). Feeding the
        landmarker a temporally smoothed bbox removes that amplification at the
        source; the landmarks are then smoothed as before.
        """
        if not faces:
            if refine_fn is not None:
                refine_fn()
            return faces
        with self._lock:
            for t in self.tracks:
                t['age'] += 1
            self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

            # ---- pass 1: match tracks, work out alpha, smooth the bbox -------
            pending = []
            for f in faces:
                bbox = np.asarray(f.bbox, dtype=np.float32)
                size = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])) + 1e-6
                center = np.array([(bbox[0] + bbox[2]) * 0.5,
                                   (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
                kps = self._get(f, 'kps')

                t = self._match(center, size)
                if t is None:
                    self.tracks.append({
                        'center': center, 'size': size, 'age': 0,
                        'bbox': bbox.copy(),
                        'kps': None if kps is None else kps.copy(),
                        'lm106': None, 'lm68': None,
                    })
                    pending.append((f, None, 1.0))
                    continue

                # adaptive alpha from how far the keypoints moved. kps is NOT
                # touched by the landmarker, so this is valid to compute now.
                if kps is not None and t.get('kps') is not None and t['kps'].shape == kps.shape:
                    motion = float(np.linalg.norm(kps - t['kps'], axis=1).mean())
                else:
                    motion = float(np.hypot(*(center - t['center'])))
                dead = self.deadzone_frac * size
                if dead > 0.0 and motion <= dead:
                    alpha = 0.0
                else:
                    motion_eff = motion - dead if dead > 0.0 else motion
                    raw = motion_eff / (self.motion_frac * size)
                    if dead > 0.0:
                        floor = self.alpha_min * min((motion - dead) / dead, 1.0)
                    else:
                        floor = self.alpha_min
                    alpha = float(np.clip(raw, floor, 1.0))

                # smooth the bbox with the same adaptive weight and write it back
                prev_bb = t.get('bbox')
                if prev_bb is not None and prev_bb.shape == bbox.shape:
                    new_bb = (alpha * bbox + (1.0 - alpha) * prev_bb).astype(np.float32)
                else:
                    new_bb = bbox
                t['bbox'] = new_bb.copy()
                try:
                    f['bbox'] = new_bb
                except Exception:
                    pass
                pending.append((f, t, alpha))

            # ---- landmarker runs here, on the SMOOTHED bboxes ----------------
            if refine_fn is not None:
                refine_fn()

            # ---- pass 2: smooth the landmarks --------------------------------
            for f, t, alpha in pending:
                kps = self._get(f, 'kps')
                lm106 = self._get(f, 'landmark_2d_106')
                lm68 = self._get(f, 'landmark_3d_68')
                if t is None:
                    # brand-new track: seed its landmark history, nothing to blend
                    for tr in self.tracks:
                        if tr['age'] == 0 and tr.get('lm68') is None and tr.get('lm106') is None:
                            tr['kps'] = None if kps is None else kps.copy()
                            tr['lm106'] = None if lm106 is None else lm106.copy()
                            tr['lm68'] = None if lm68 is None else lm68.copy()
                            break
                    continue
                beta = 1.0 - alpha

                def smooth(cur, prev):
                    if cur is None:
                        return None, prev
                    if prev is None or prev.shape != cur.shape:
                        return cur, cur.copy()
                    out = (alpha * cur + beta * prev).astype(cur.dtype)
                    return out, out.copy()

                new_kps, t['kps'] = smooth(kps, t['kps'])
                new_lm106, t['lm106'] = smooth(lm106, t['lm106'])

                new_lm68 = lm68
                if lm68 is not None:
                    if t.get('lm68') is None or t['lm68'].shape != lm68.shape:
                        t['lm68'] = lm68.copy()
                    else:
                        new_lm68 = lm68.copy()
                        new_lm68[:, :2] = (alpha * lm68[:, :2] +
                                           beta * t['lm68'][:, :2])
                        t['lm68'] = new_lm68.copy()

                bb = np.asarray(f.bbox, dtype=np.float32)
                t['center'] = np.array([(bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5], dtype=np.float32)
                t['size'] = float(max(bb[2] - bb[0], bb[3] - bb[1])) + 1e-6
                t['age'] = 0

                # write smoothed landmarks back onto the face object
                try:
                    if new_kps is not None:
                        f['kps'] = new_kps
                    if new_lm106 is not None:
                        f['landmark_2d_106'] = new_lm106
                    if new_lm68 is not None:
                        f['landmark_3d_68'] = new_lm68
                except Exception:
                    pass
        return faces

