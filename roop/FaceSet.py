import numpy as np
import roop.globals


class FaceSet:
    faces = []
    ref_images = []
    embedding_average = 'None'
    embeddings_backup = None

    def __init__(self):
        self.faces = []
        self.ref_images = []
        self.embeddings_backup = None

    def AverageEmbeddings(self):
        """Combine the embeddings of all uploaded source faces into one robust
        identity stored on faces[0]['embedding'] (the swapper renormalizes it
        via normed_embedding on use).

        Naive np.mean lets a single bad upload (blurry, extreme angle, or even a
        different person) drag the identity away from the real source. We instead
        (1) drop outliers by cosine distance from the group's median embedding,
        and (2) take a detector-confidence-weighted mean of the inliers. Modes
        ('mean' | 'median' | 'robust') and the outlier threshold are tunable in
        globals. Falls back to the original simple mean on any problem.
        """
        if len(self.faces) <= 1 or self.embeddings_backup is not None:
            return

        self.embeddings_backup = self.faces[0]['embedding']

        try:
            mode = getattr(roop.globals, 'faceset_average_mode', 'robust')
            thr = float(getattr(roop.globals, 'faceset_outlier_threshold', 0.6))

            raw = np.array([np.asarray(f.embedding, dtype=np.float32)
                            for f in self.faces], dtype=np.float32)
            scores = np.array([float(getattr(f, 'det_score', 1.0) or 1.0)
                               for f in self.faces], dtype=np.float32)

            if mode == 'mean':
                self.faces[0]['embedding'] = raw.mean(axis=0)
                return

            # unit-normalize for robust statistics / distances
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1e-6
            unit = raw / norms

            center = np.median(unit, axis=0)
            cnorm = np.linalg.norm(center)
            center_u = center / (cnorm if cnorm > 0 else 1e-6)

            if mode == 'median':
                # store the embedding closest to the median direction
                sims = unit @ center_u
                self.faces[0]['embedding'] = raw[int(np.argmax(sims))]
                return

            # mode == 'robust': reject outliers, then confidence-weighted mean
            dist = 1.0 - (unit @ center_u)          # cosine distance to median
            keep = dist <= thr
            if keep.sum() < 1:                       # threshold too tight -> keep all
                keep = np.ones(len(self.faces), dtype=bool)

            w = scores[keep].copy()
            if not np.isfinite(w).all() or w.sum() <= 0:
                w = np.ones(int(keep.sum()), dtype=np.float32)
            w = w / w.sum()

            self.faces[0]['embedding'] = (raw[keep] * w[:, None]).sum(axis=0)
        except Exception:
            # safe fallback: original behaviour
            embeddings = [face.embedding for face in self.faces]
            self.faces[0]['embedding'] = np.mean(embeddings, axis=0)
