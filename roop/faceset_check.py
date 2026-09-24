"""Faceset check: per-photo diagnostics for the photos of one person.

The swap uses ONE identity averaged over the faceset (FaceSet.AverageEmbeddings)
and, for "Face shape from source", the median jaw shape of its near-frontal,
mouth-closed photos. A few photos can pull both off: someone else's face, many
near-identical shots of one moment, tiny or blurry faces. This lists them.

Per photo:
- similarity: cosine of its identity embedding to the average of all the OTHER
  photos (leave-one-out), so a photo is not compared with itself.
- flags that suggest removing it ("Remove flagged"):
    other person?     cosine < 0.40 to the faceset's median identity (the rule
                      FaceSet 'robust' drops by; needs 3+ photos)
    unlike the others similarity 0.12+ below the faceset's median
    duplicate of #n   cosine > 0.95 to photo n, the best of its group (burst /
                      same video frame); the best one stays
    small             face < 128 px in the original photo (the embedding sees 112 px)
    blurry            sharpness < 40% of the faceset's median
- information only: head angles, mouth open, and whether face shape uses it
  (roop/face_shape.pick_shape_photos, the same choice the swap makes).
The thresholds are heuristics: check the flagged photos by eye.
"""
import cv2
import numpy as np

OTHER_PERSON = 0.40
UNLIKE_MARGIN = 0.12
DUPLICATE = 0.95
SMALL_FACE = 128
BLURRY = 0.40

REMOVABLE = ('other person?', 'unlike the others', 'duplicate', 'small', 'blurry')


def _get(face, key):
    try:
        return face.get(key)
    except Exception:
        return getattr(face, key, None)


def sharpness(image, bbox):
    """Laplacian variance of the face area at a fixed size (so faces of any
    size compare), or None."""
    if image is None or bbox is None:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    w, h = x1 - x0, y1 - y0
    x0, x1 = int(max(0, x0 + 0.1 * w)), int(min(image.shape[1], x1 - 0.1 * w))
    y0, y1 = int(max(0, y0 + 0.1 * h)), int(min(image.shape[0], y1 - 0.1 * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def photo_metrics(face, image, geometry=True):
    """Values of one photo that do not depend on the others (cache them)."""
    emb = _get(face, 'embedding')
    unit = None
    if emb is not None:
        emb = np.asarray(emb, np.float64).reshape(-1)
        n = np.linalg.norm(emb)
        unit = emb / n if n > 0 else None
    m = {'unit': unit, 'sharp': sharpness(image, _get(face, 'bbox')),
         'size': _get(face, 'source_size'), 'yaw': None, 'pitch': None, 'mouth': None}
    if geometry:
        try:
            from roop.face_shape import photo_geometry, source_landmarks
            lm = source_landmarks(face, image)
            if lm is not None:
                m['yaw'], m['pitch'], m['mouth'] = photo_geometry(lm)
        except Exception:
            pass
    return m


def check(metrics):
    """Rows (one per photo, in order): similarity to the others, flags,
    whether the photo is removable, and its part in the source shape ('yes' =
    used, 'fill' = used to reach the minimum, 'no', None = no 3D landmarks)."""
    from roop.face_shape import pick_shape_photos
    n = len(metrics)
    rows = [{'sim': None, 'flags': [], 'removable': False, 'shape': None} for _ in range(n)]
    units = [m['unit'] for m in metrics]
    idx = [i for i, u in enumerate(units) if u is not None]
    for i in range(n):
        if units[i] is None:
            rows[i]['flags'].append('no face data')
    if len(idx) >= 2:
        U = np.stack([units[i] for i in idx])
        total = U.sum(0)
        for k, i in enumerate(idx):
            rest = total - U[k]
            rows[i]['sim'] = float(U[k] @ rest / (np.linalg.norm(rest) + 1e-12))
    if len(idx) >= 3:
        # identity: the median direction (FaceSet 'robust'), which a few other
        # people cannot pull toward themselves the way an average can
        centre = np.median(np.stack([units[i] for i in idx]), axis=0)
        centre /= np.linalg.norm(centre) + 1e-12
        sims = [rows[i]['sim'] for i in idx]
        median = float(np.median(sims))
        for i in idx:
            if float(units[i] @ centre) < OTHER_PERSON:
                rows[i]['flags'].append('other person?')
            elif len(idx) >= 5 and rows[i]['sim'] < median - UNLIKE_MARGIN:
                rows[i]['flags'].append('unlike the others')
    sharp = [m['sharp'] for m in metrics if m['sharp'] is not None]
    sharp_med = float(np.median(sharp)) if len(sharp) >= 3 else None
    poor = [False] * n
    for i, m in enumerate(metrics):
        if m['size'] is not None and m['size'] < SMALL_FACE:
            rows[i]['flags'].append(f"small ({m['size']:.0f} px)")
            poor[i] = True
        if sharp_med and m['sharp'] is not None and m['sharp'] < BLURRY * sharp_med:
            rows[i]['flags'].append('blurry')
            poor[i] = True
    if len(idx) >= 2:
        # near-identical shots: keep the best of each group, flag the rest
        parent = {i: i for i in idx}
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        G = U @ U.T
        for a in range(len(idx)):
            for b in np.nonzero(G[a, a + 1:] > DUPLICATE)[0]:
                parent[root(idx[a + 1 + b])] = root(idx[a])
        groups = {}
        for i in idx:
            groups.setdefault(root(i), []).append(i)
        quality = lambda i: (not poor[i], metrics[i]['sharp'] or 0.0, metrics[i]['size'] or 0.0, -i)
        for members in groups.values():
            if len(members) > 1:
                keep = max(members, key=quality)
                for i in members:
                    if i != keep:
                        rows[i]['flags'].append(f'duplicate of #{keep + 1}')
    geometry = [None if m['yaw'] is None else (m['yaw'], m['pitch'], m['mouth']) for m in metrics]
    if any(g is not None for g in geometry):
        used, usable = pick_shape_photos(units, geometry)
        for rank, i in enumerate(used):
            rows[i]['shape'] = 'yes' if rank < usable else 'fill'
        for i, g in enumerate(geometry):
            if g is not None and rows[i]['shape'] is None:
                rows[i]['shape'] = 'no'
    for r in rows:
        r['removable'] = any(f.startswith(REMOVABLE) for f in r['flags'])
    return rows


def summary(rows, shape=True):
    """One line for the UI / console."""
    n = len(rows)
    count = lambda prefix: sum(1 for r in rows if any(f.startswith(prefix) for f in r['flags']))
    parts = []
    for label, prefix in (('another person?', 'other person?'), ('unlike the others', 'unlike the others'),
                          ('near-duplicate', 'duplicate'), ('small', 'small'), ('blurry', 'blurry')):
        c = count(prefix)
        if c:
            parts.append(f'{label} ×{c}')
    text = f'{n} faces: ' + (', '.join(parts) if parts else 'nothing flagged')
    sims = [r['sim'] for r in rows if r['sim'] is not None]
    if sims:
        text += f'; similarity to the rest: median {np.median(sims):.2f}'
        if len(sims) >= 3 and np.median(sims) < 0.30:
            text += ' (low: these photos do not look like one person)'
    if shape and any(r['shape'] is not None for r in rows):
        used = sum(1 for r in rows if r['shape'] in ('yes', 'fill'))
        text += f"; face shape uses {used} ({sum(1 for r in rows if r['shape'] == 'yes')} near-frontal with the mouth closed)"
    return text
