"""Temporal smoother for the LivePortrait expression coefficients.

The Expression Restorer extracts the target expression FRESH every frame, with
no memory, so per-frame noise in that small control vector shows up as visible
wobble in the restored region (eyes/mouth/brows) -- and only there. This smooths
the *expression vector itself* (the cause), not the pixels (the symptom), which
is cheap, precise, and leaves everything outside the ER region untouched.

Design (independent, single knob):
  - Per-face tracks matched by landmark centre (like the landmark stabiliser),
    so multi-face and single-face both behave.
  - Motion-adaptive EMA that *self-calibrates*: the reference is a slow running
    average of the frame-to-frame expression change, so we never need to know
    LivePortrait's absolute expression scale. Calm frames (change <= the running
    average) are smoothed hard (alpha -> alpha_min); real expression bursts
    (change well above average, e.g. a blink or speech) are followed (alpha -> 1).
  - strength in [0,1] sets alpha_min. strength 0 -> passthrough (identical to no
    smoothing), higher -> calmer.
"""

from threading import Lock

import numpy as np


class ExpressionSmoother:
    def __init__(self, strength: float = 0.5, max_age: int = 12, match_frac: float = 0.6):
        self.set_strength(strength)
        self.max_age = int(max_age)
        self.match_frac = float(match_frac)
        self.tracks = []          # {'c':[x,y], 'size':float, 'exp':ndarray, 'mavg':float, 'age':int}
        self._lock = Lock()

    def set_strength(self, strength: float):
        # alpha_min = floor on how much of the NEW expression is taken during calm
        # moments. strength 0 -> alpha_min ~1 (passthrough); 1 -> heavy smoothing.
        self.alpha_min = float(np.clip(1.0 - float(strength), 0.05, 1.0))

    def smooth(self, applied, center, size):
        """Return a temporally-smoothed copy of `applied` for this face."""
        applied = np.asarray(applied, dtype=np.float32)
        if self.alpha_min >= 0.999:          # strength ~0 -> passthrough
            return applied
        with self._lock:
            t = self._match(center, size)
            prev = t['exp']
            if prev is None or prev.shape != applied.shape:
                t['exp'] = applied.copy()
                t['mavg'] = 0.0
                return applied
            motion = float(np.mean(np.abs(applied - prev)))
            # slow running average of motion = self-calibrating noise reference
            t['mavg'] = motion if t['mavg'] <= 0.0 else 0.9 * t['mavg'] + 0.1 * motion
            ratio = motion / max(t['mavg'], 1e-9)
            # calm (motion <= average, ratio<=1) -> alpha_min; burst (ratio>=2.5) -> 1
            ramp = float(np.clip((ratio - 1.0) / (2.5 - 1.0), 0.0, 1.0))
            alpha = self.alpha_min + (1.0 - self.alpha_min) * ramp
            out = (alpha * applied + (1.0 - alpha) * prev).astype(np.float32)
            t['exp'] = out
            return out

    def _match(self, center, size):
        center = np.asarray(center, dtype=np.float32).reshape(2)
        for tr in self.tracks:
            tr['age'] += 1
        self.tracks = [tr for tr in self.tracks if tr['age'] <= self.max_age]
        thr = self.match_frac * max(float(size), 1.0)
        best, bestd = None, None
        for tr in self.tracks:
            d = float(np.linalg.norm(tr['c'] - center))
            if d <= thr and (bestd is None or d < bestd):
                best, bestd = tr, d
        if best is None:
            best = {'c': center.copy(), 'size': float(size), 'exp': None, 'mavg': 0.0, 'age': 0}
            self.tracks.append(best)
        else:
            best['c'] = center.copy()
            best['size'] = float(size)
            best['age'] = 0
        return best
