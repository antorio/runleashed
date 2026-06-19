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
                 match_frac: float = 0.6, max_age: int = 8):
        # strength in [0,1]; higher = smoother. Maps to a floor on alpha.
        self.alpha_min = float(np.clip(1.0 - strength, 0.08, 1.0))
        self.motion_frac = motion_frac
        self.match_frac = match_frac
        self.max_age = max_age
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

    def stabilize(self, faces):
        if not faces:
            return faces
        with self._lock:
            for t in self.tracks:
                t['age'] += 1
            self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

            for f in faces:
                bbox = np.asarray(f.bbox, dtype=np.float32)
                size = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])) + 1e-6
                center = np.array([(bbox[0] + bbox[2]) * 0.5,
                                   (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)

                kps = self._get(f, 'kps')
                lm106 = self._get(f, 'landmark_2d_106')
                lm68 = self._get(f, 'landmark_3d_68')

                t = self._match(center, size)
                if t is None:
                    self.tracks.append({
                        'center': center, 'size': size, 'age': 0,
                        'kps': None if kps is None else kps.copy(),
                        'lm106': None if lm106 is None else lm106.copy(),
                        'lm68': None if lm68 is None else lm68.copy(),
                    })
                    continue

                # adaptive alpha from how far the keypoints moved
                if kps is not None and t['kps'] is not None and t['kps'].shape == kps.shape:
                    motion = float(np.linalg.norm(kps - t['kps'], axis=1).mean())
                else:
                    motion = float(np.hypot(*(center - t['center'])))
                alpha = float(np.clip(motion / (self.motion_frac * size),
                                      self.alpha_min, 1.0))
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
                    if t['lm68'] is None or t['lm68'].shape != lm68.shape:
                        t['lm68'] = lm68.copy()
                    else:
                        new_lm68 = lm68.copy()
                        new_lm68[:, :2] = (alpha * lm68[:, :2] +
                                           beta * t['lm68'][:, :2])
                        t['lm68'] = new_lm68.copy()

                t['center'], t['size'], t['age'] = center, size, 0

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
