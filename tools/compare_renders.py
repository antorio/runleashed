"""Compare two rendered videos frame by frame.

    python tools/compare_renders.py A.mp4 B.mp4

Use it to check that a change which should not alter the picture really does
not: render the same clip with the same settings on two versions and compare.
Both files are H.264, so identical input frames give identical output frames;
any difference is reported per frame. As a reference for normal run-to-run
noise, compare two renders made with the SAME version.
"""
import argparse
import sys

import cv2
import numpy as np


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--show', type=int, default=10, help='how many of the most different frames to list')
    args = parser.parse_args()

    cap_a, cap_b = cv2.VideoCapture(args.a), cv2.VideoCapture(args.b)
    if not cap_a.isOpened() or not cap_b.isOpened():
        sys.exit('cannot open one of the videos')

    rows = []   # (frame, max abs diff, changed pixels %, psnr)
    index = 0
    while True:
        ok_a, fa = cap_a.read()
        ok_b, fb = cap_b.read()
        if not ok_a or not ok_b:
            break
        if fa.shape != fb.shape:
            sys.exit(f'frame {index}: different sizes {fa.shape} vs {fb.shape}')
        diff = cv2.absdiff(fa, fb)
        changed = np.count_nonzero(diff.max(axis=2))
        rows.append((index, int(diff.max()), 100.0 * changed / (fa.shape[0] * fa.shape[1]), psnr(fa, fb)))
        index += 1
    rest_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT)) - index
    rest_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT)) - index

    if not rows:
        sys.exit('no frames compared')
    identical = sum(1 for r in rows if r[1] == 0)
    print(f'frames compared : {len(rows)}' + (f'  (frames left unread: A {rest_a}, B {rest_b})' if rest_a > 0 or rest_b > 0 else ''))
    print(f'identical frames: {identical} ({100.0 * identical / len(rows):.1f}%)')
    if identical == len(rows):
        print('-> the two videos are identical')
        return
    finite = [r[3] for r in rows if np.isfinite(r[3])]
    print(f'PSNR of differing frames: min {min(finite):.1f} dB, median {float(np.median(finite)):.1f} dB '
          f'(above ~45 dB is not visible)')
    print(f'most different frames (frame: max diff, changed pixels, PSNR):')
    for f, d, pct, p in sorted(rows, key=lambda r: r[3])[:args.show]:
        print(f'  {f:6d}: {d:3d}, {pct:6.2f}%, {p:5.1f} dB')


if __name__ == '__main__':
    main()
