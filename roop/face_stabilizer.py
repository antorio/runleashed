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
        self.alpha_min = float(np.clip(1.0 - strength, 0.08, 1.0))
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
                # soft dead-zone. Inside the zone (motion <= dead) the smoothed
                # landmarks freeze (alpha=0) -> still-head detector wobble is
                # killed. Just outside, the alpha floor ramps 0 -> alpha_min across
                # (dead, 2*dead] so slow real motion is picked up without a jump.
                # deadzone_frac=0 -> dead=0 -> falls straight to the original
                # clip(motion/(motion_frac*size), alpha_min, 1) formula.
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


class _OneEuro:
    """Textbook One-Euro filter for one scalar channel.

    cutoff = min_cutoff + beta * |dx_smoothed|  -- near-still signals get a low
    cutoff (heavy smoothing, kills jitter), fast signals raise the cutoff so the
    filter follows with minimal lag. freq is the assumed frame rate; since the
    parameters only enter through cutoff/freq ratios, an approximate constant
    (30) works across common video rates.
    """
    __slots__ = ('min_cutoff', 'beta', 'd_cutoff', 'freq', 'x_prev', 'dx_prev')

    def __init__(self, min_cutoff=1.0, beta=2.0, d_cutoff=0.5, freq=30.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.freq = float(freq)
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        x = float(x)
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        ad = self._alpha(self.d_cutoff, self.freq)
        self.dx_prev = ad * dx + (1.0 - ad) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self._alpha(cutoff, self.freq)
        self.x_prev = a * x + (1.0 - a) * self.x_prev
        return self.x_prev


class MatrixStabilizer:
    """One-Euro smoothing of the ALIGNMENT MATRIX itself (video only).

    The landmark smoother reduces point jitter, but estimate_norm still turns
    residual landmark noise into frame-to-frame flicker of the warp's rotation /
    scale / position -- visible as a subtle shimmer of the pasted face even when
    the head is still. Smoothing the similarity parameters of M kills that at
    the source: M (2x3, frame->crop) is decomposed into (theta, log s, tx, ty),
    each channel runs through a One-Euro filter (per face track), and M is
    rebuilt. One-Euro is motion-adaptive by construction: a still head gets
    heavy smoothing, a fast pan/turn is followed with negligible lag.

    Tracks are matched by face bbox centre (like LandmarkStabilizer). The
    translation channels are normalised by the crop size so one beta fits all.
    strength (reuses landmark_smoothing_strength) maps to min_cutoff.
    """

    def __init__(self, strength: float = 0.7, match_frac: float = 0.6,
                 max_age: int = 8, freq: float = 30.0):
        # strength -> min_cutoff on a log scale: 0 -> 4 Hz (barely smooths),
        # 0.7 -> ~0.6 Hz, 1 -> 0.25 Hz (calm). beta keeps real motion followed:
        # the cutoff rises with the (smoothed) parameter velocity, so pans and
        # head turns immediately reopen the filter.
        s = float(np.clip(strength, 0.0, 1.0))
        self.min_cutoff = float(4.0 * (0.04 ** s))
        self.beta = 10.0   # high beta = filter fully reopens on fast motion (no trail)
        self.match_frac = float(match_frac)
        self.max_age = int(max_age)
        self.freq = float(freq)
        self.tracks = []
        self._lock = Lock()

    def reset(self):
        with self._lock:
            self.tracks = []

    def __init__(self, strength: float = 0.7, match_frac: float = 0.6,
                 max_age: int = 8, freq: float = 30.0, deadband_px: float = 1.2):
        # strength -> min_cutoff on a log scale: 0 -> 4 Hz (barely smooths),
        # 0.7 -> ~0.4 Hz, 1 -> 0.16 Hz (calm). beta keeps real motion followed:
        # the cutoff rises with the (smoothed) parameter velocity, so pans and
        # head turns immediately reopen the filter.
        s = float(np.clip(strength, 0.0, 1.0))
        self.min_cutoff = float(4.0 * (0.04 ** s))
        self.beta = 10.0   # high beta = filter fully reopens on fast motion (no trail)
        self.match_frac = float(match_frac)
        self.max_age = int(max_age)
        self.freq = float(freq)
        # Deadband: if smoothing moves the crop corners by less than this many
        # pixels, DON'T touch M at all. Without it One-Euro re-warps every frame
        # by a sub-pixel amount even when the alignment is already steady, and
        # that constant sub-pixel resampling reads as a permanent micro-jitter
        # around the face. The deadband makes the stabilizer act only when there
        # is real flicker to remove and stay perfectly still otherwise.
        self.deadband_px = float(deadband_px)
        self.tracks = []
        self._lock = Lock()

    def _new_filters(self):
        return [_OneEuro(self.min_cutoff, self.beta, freq=self.freq) for _ in range(4)]

    def _match(self, center, size):
        best, best_d = None, 1e18
        for t in self.tracks:
            d = float(np.hypot(*(center - t['center'])))
            if d < self.match_frac * size and d < best_d:
                best_d, best = d, t
        return best

    @staticmethod
    def _corner_shift(M_a, M_b, crop_size):
        """Max displacement (px, in FRAME space) of the crop's four corners
        between two 2x3 forward (frame->crop) similarity matrices. This is the
        correct 'how much did the warp actually move' metric -- comparing raw
        matrix elements mixes radians, log-scale and pixels."""
        cs = float(crop_size)
        corners = np.array([[0, 0], [cs, 0], [0, cs], [cs, cs]], dtype=np.float64)
        # invert each 2x3 (frame->crop) to map crop corners back into the frame
        def inv_map(M, pts):
            A = M[:, :2]; b = M[:, 2]
            Ai = np.linalg.inv(A)
            return (pts - b) @ Ai.T
        try:
            pa = inv_map(M_a, corners)
            pb = inv_map(M_b, corners)
        except np.linalg.LinAlgError:
            return 1e9
        return float(np.linalg.norm(pa - pb, axis=1).max())

    def smooth(self, M, bbox, crop_size):
        """M: 2x3 similarity (frame->crop). Returns (M_out, changed).
        changed=False means the deadband held and M_out IS M (caller should keep
        the original crop, skipping the re-warp entirely)."""
        M = np.asarray(M, dtype=np.float64)
        s = float(np.hypot(M[0, 0], M[1, 0]))
        if not np.isfinite(s) or s <= 1e-9:
            return M.astype(np.float64), False
        theta = float(np.arctan2(M[1, 0], M[0, 0]))
        cs = float(max(crop_size, 1))
        params = [theta, float(np.log(s)), float(M[0, 2]) / cs, float(M[1, 2]) / cs]

        bb = np.asarray(bbox, dtype=np.float32)
        size = float(max(bb[2] - bb[0], bb[3] - bb[1])) + 1e-6
        center = np.array([(bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5], dtype=np.float32)

        with self._lock:
            for t in self.tracks:
                t['age'] += 1
            self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
            t = self._match(center, size)
            if t is None:
                t = {'center': center, 'size': size, 'age': 0,
                     'filters': self._new_filters(), 'theta_prev': theta}
                self.tracks.append(t)
            else:
                t['center'], t['size'], t['age'] = center, size, 0
            # unwrap theta relative to this track so a +-pi crossing does not
            # look like a huge jump to the filter
            dth = theta - t['theta_prev']
            while dth > np.pi:
                theta -= 2 * np.pi; dth = theta - t['theta_prev']
            while dth < -np.pi:
                theta += 2 * np.pi; dth = theta - t['theta_prev']
            t['theta_prev'] = theta
            params[0] = theta
            out = [f(p) for f, p in zip(t['filters'], params)]

        th, ls, tx, ty = out
        s2 = float(np.exp(ls))
        c, sn = np.cos(th) * s2, np.sin(th) * s2
        M_s = np.array([[c, -sn, tx * cs],
                        [sn,  c, ty * cs]], dtype=np.float64)

        # Deadband: if the smoothed warp barely differs from the raw one, keep
        # the raw M so no re-warp happens and there is zero added resampling.
        if self._corner_shift(M, M_s, cs) < self.deadband_px:
            return M, False
        return M_s, True
