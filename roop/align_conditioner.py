"""Align5Conditioner — the SINGLE shared implementation of the two alignment
conditioners, used by BOTH ProcessMgr.process_face and tools_jitter_probe so the
probe always measures exactly the code that renders (tested == shipped):

1. align5 smoothing: adaptive EMA on the 5 alignment points, right where the
   estimate() amplification happens. alpha follows FACE-CENTRE motion, so a
   still head is smoothed hard (kills the 7x-amplified corner wobble) while a
   panning head passes through raw (no lag).
2. sanity gate v2: learns the NORMAL 68->5-vs-kps disagreement as a baseline
   (with 2dfan4 there is a systematic ~0.09 offset that is NOT an error) and
   only falls back to detector-kps alignment on a real ANOMALY, with hysteresis
   so the alignment basis cannot chatter frame-to-frame.

Single-track state (matched to the dominant face). With multiple faces the
centre jumps between them every call -> alpha hits 1.0 -> smoothing safely
passes raw points through (no cross-face blending); the gate baseline mixes
faces, which is noted as a limitation rather than a hazard.
"""
import numpy as np


class Align5Conditioner:
    def __init__(self, alpha_min=0.12, motion_frac=0.04, gate_thr=0.08,
                 smoothing=True, gate=True, debug=False):
        self.alpha_min = float(alpha_min)
        self.motion_frac = float(motion_frac)
        self.gate_thr = float(gate_thr)
        self.smoothing = bool(smoothing)
        self.gate = bool(gate)
        self.debug = bool(debug)
        self.reset()

    def reset(self):
        self._prev = None
        self._centre = None
        self.baseline = None
        self.tripped = False
        self._exit_ok = 0
        # counters (probe/diagnostics)
        self.n_frames = 0
        self.n_fallback = 0
        self.n_flips = 0

    def condition(self, lmk5, kps5, face_size):
        """Returns (points, use_lmk).

        points   -- the (possibly smoothed) 5 alignment points to fit with when
                    use_lmk is True.
        use_lmk  -- False = gate says this frame's 68->5 set is anomalous;
                    caller should align with the detector kps instead.
        """
        self.n_frames += 1
        lmk5 = np.asarray(lmk5, dtype=np.float32)

        # ---- adaptive smoothing of the 5 align points ----
        if self.smoothing:
            centre = lmk5.mean(0)
            if self._prev is not None and self._prev.shape == lmk5.shape:
                centre_motion = float(np.linalg.norm(centre - self._centre))
                alpha = float(np.clip(centre_motion / (self.motion_frac * max(face_size, 1e-6)),
                                      self.alpha_min, 1.0))
                lmk5 = (alpha * lmk5 + (1.0 - alpha) * self._prev).astype(np.float32)
            self._prev = lmk5.copy()
            self._centre = lmk5.mean(0)
        else:
            self._prev = None
            self._centre = None

        # ---- sanity gate v2 (anomaly vs adaptive baseline, with hysteresis) ----
        use_lmk = True
        if self.gate and kps5 is not None:
            kps5 = np.asarray(kps5, dtype=np.float32).reshape(-1, 2)[:5]
            if kps5.shape == lmk5.shape:
                d_mean = float((np.linalg.norm(lmk5 - kps5, axis=1) /
                                max(face_size, 1e-6)).mean())
                if self.baseline is None:
                    self.baseline = max(d_mean, 0.01)
                was = self.tripped
                enter = d_mean > max(self.gate_thr * 2.0, 2.2 * self.baseline)
                leave = d_mean < max(self.gate_thr * 0.8, 1.6 * self.baseline)
                if not self.tripped and enter:
                    self.tripped = True
                    self._exit_ok = 0
                elif self.tripped:
                    self._exit_ok = self._exit_ok + 1 if leave else 0
                    if self._exit_ok >= 5:
                        self.tripped = False
                if not self.tripped:
                    # learn the baseline only from healthy frames
                    self.baseline = 0.95 * self.baseline + 0.05 * d_mean
                    if enter:
                        use_lmk = False           # single-frame anomaly
                else:
                    use_lmk = False               # held in fallback (hysteresis)
                if self.tripped != was:
                    self.n_flips += 1
                if not use_lmk:
                    self.n_fallback += 1
                    if self.debug:
                        print(f"[lmk-gate v2] fallback (mean={d_mean:.3f} "
                              f"baseline={self.baseline:.3f})")
        return lmk5, use_lmk
