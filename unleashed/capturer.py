import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Optional
import cv2
import numpy as np

from unleashed.typing import Frame
import unleashed.globals

current_video_path = None
current_frame_total = 0
current_capture = None

# Frame cache for the preview and the face pickers. OpenCV seeks by decoding
# from the previous keyframe, so every request -- even the same frame again,
# i.e. every refresh after a setting change -- cost 0.1-0.9 s (1080p..4K).
# Recently read frames are kept, and a frame a few steps ahead of the decoder is
# reached by reading forward instead of seeking; either way the frame returned
# is the one the plain seek returns. Callers get a copy, so they may modify it.
current_position = None     # index the next read() returns; None = unknown
_frames = OrderedDict()     # frame index -> frame, for current_video_path only
_frames_bytes = 0
_CACHE_BYTES = 256 * 1024 * 1024
_READ_AHEAD = 30            # decoding further forward than this can be slower than a seek
_lock = threading.Lock()    # Gradio events can ask for frames concurrently

# Quick-seek copy. Even with the cache, every NEW frame costs a seek, i.e.
# decoding from the previous keyframe -- up to a few seconds apart in camera /
# web video. When a target video is selected, a copy with a keyframe every
# _SEEK_GOP frames is encoded in the background (ffmpeg, NVENC when available);
# once it is verified to hold the same frames, the preview reads from it. The
# face pickers (exact=True) and renders always read the original file.
current_source = None       # file the capture actually reads (video or its copy)
_SEEK_GOP = 8
_seek_copies = {}           # video path -> verified quick-seek copy
_seek_jobs = set()          # video paths with a copy being made


def get_image_frame(filename: str):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        print(f"Exception reading {filename}")
    return None


def get_video_frame(video_path: str, frame_number: int = 0, exact: bool = False) -> Optional[Frame]:
    """Frame frame_number (1-based) of the video. exact=False (preview) may
    read the quick-seek copy; exact=True (face pickers) always returns the
    original file's frame."""
    global current_video_path, current_capture, current_frame_total, current_position, current_source

    with _lock:
        source = video_path
        if not exact and getattr(unleashed.globals, 'preview_seek_copy', True):
            source = _seek_copies.get(video_path, video_path)
            if source != video_path and not os.path.isfile(source):
                _seek_copies.pop(video_path, None)      # the temp folder was cleaned
                source = video_path
        if video_path != current_video_path or current_capture is None:
            _release_video()
            current_capture = cv2.VideoCapture(source)
            current_source = source
            current_video_path = video_path
            current_frame_total = current_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            current_position = 0
        elif source != current_source:
            # switching between the video and its quick-seek copy: same frames,
            # keep the cache
            current_capture.release()
            current_capture = cv2.VideoCapture(source)
            current_source = source
            current_position = 0

        index = frame_number - 1
        if index < 0 or index >= current_frame_total:
            # outside the reported frame range: exactly the old clamped seek, uncached
            current_capture.set(cv2.CAP_PROP_POS_FRAMES, min(current_frame_total, index))
            current_position = None
            has_frame, frame = current_capture.read()
            return frame if has_frame else None

        cached = _frames.get(index)
        if cached is not None and (cached[1] or not exact):
            _frames.move_to_end(index)
            return cached[0].copy()

        ahead = None if current_position is None else index - current_position
        if ahead is not None and 0 <= ahead <= _READ_AHEAD:
            # a few frames ahead of the decoder: decode forward, no seek
            while ahead > 0 and current_capture.grab():
                ahead -= 1
        if ahead != 0:
            current_capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        has_frame, frame = current_capture.read()
        if not has_frame:
            current_position = None
            return None
        current_position = index + 1
        _remember(index, frame, source == video_path)
        return frame.copy()


def _remember(index, frame, exact):
    """Cache a frame; exact = read from the original file (not the copy)."""
    global _frames_bytes

    old = _frames.pop(index, None)
    if old is not None:
        _frames_bytes -= old[0].nbytes
    _frames[index] = (frame, exact)
    _frames_bytes += frame.nbytes
    while _frames_bytes > _CACHE_BYTES and len(_frames) > 1:
        _, oldest = _frames.popitem(last=False)
        _frames_bytes -= oldest[0].nbytes


def release_video():
    with _lock:
        _release_video()


def _release_video():
    global current_capture, current_video_path, current_position, current_source, _frames_bytes

    if current_capture is not None:
        current_capture.release()
        current_capture = None
    current_source = None
    # forget the path too: the old code kept it, so asking for the same video
    # again after a release used the released (None) capture and crashed
    current_video_path = None
    current_position = None
    _frames.clear()
    _frames_bytes = 0


def get_video_frame_total(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total


def forget_seek_copies():
    """The temp folder was emptied: read the original videos again."""
    global current_source
    with _lock:
        _seek_copies.clear()
        if current_capture is not None and current_source != current_video_path:
            _release_video()


def prepare_seek_copy(video_path):
    """Start making the quick-seek copy of a video in the background (once per
    video; a no-op when ffmpeg is missing or the copy exists)."""
    if not video_path or not os.path.isfile(video_path) or shutil.which('ffmpeg') is None:
        return
    if not getattr(unleashed.globals, 'preview_seek_copy', True):
        return
    with _lock:
        if video_path in _seek_copies or video_path in _seek_jobs:
            return
        _seek_jobs.add(video_path)
    threading.Thread(target=_make_seek_copy, args=(video_path,), name='seek-copy', daemon=True).start()


_encode_lock = threading.Lock()     # one encode at a time


def _make_seek_copy(video_path):
    try:
        with _encode_lock:
            _encode_seek_copy(video_path)
    finally:
        with _lock:
            _seek_jobs.discard(video_path)


def _encode_seek_copy(video_path):
    try:
        st = os.stat(video_path)
        key = hashlib.md5(f'{os.path.abspath(video_path)}|{st.st_size}|{st.st_mtime}'.encode()).hexdigest()[:16]
        # the app's temp folder (./temp), which Settings > Clean temp folder
        # and every start empty; tempfile's default may be the OS one
        folder = os.path.join(os.environ.get('TEMP') or tempfile.gettempdir(), 'unleashed_seek')
        os.makedirs(folder, exist_ok=True)
        out = os.path.join(folder, key + '.mp4')
        t0 = time.time()
        if not os.path.isfile(out):
            part = os.path.join(folder, key + '.part.mp4')
            gop = ['-g', str(_SEEK_GOP), '-bf', '0']
            encoders = [['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18'] + gop]
            try:
                if any('CUDA' in p for p in unleashed.globals.execution_providers):
                    encoders.insert(0, ['-c:v', 'h264_nvenc', '-preset', 'p1', '-rc', 'constqp', '-qp', '18'] + gop)
            except Exception:
                pass
            # every frame passed through as it is (no fps conversion): -fps_mode
            # since ffmpeg 5.1, -vsync before (removed in newer versions)
            syncs = [['-fps_mode', 'passthrough'], ['-vsync', '0']]
            done = False
            for enc in encoders:
                for sync in syncs:
                    # low priority: a render running at the same time comes first
                    cmd = (['nice', '-n', '10'] if shutil.which('nice') else []) + ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', video_path,
                           '-map', '0:v:0', '-an', '-sn', '-dn'] + sync + enc + ['-pix_fmt', 'yuv420p', part]
                    if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                        done = True
                        break
                if done:
                    break
            if not done:
                print('[preview] quick-seek copy could not be made; seeking the original video')
                return
            os.replace(part, out)
        if _same_frames(video_path, out):
            with _lock:
                _seek_copies[video_path] = out
            print(f'[preview] quick-seek copy ready ({time.time() - t0:.0f} s): frame changes no longer seek the original')
        else:
            print('[preview] quick-seek copy does not match the video frame for frame; seeking the original video')
    except Exception as e:
        print(f'[preview] quick-seek copy failed ({e}); seeking the original video')


def _same_frames(video_path, copy_path):
    """Same frame count, and three frames spread over the video look the same
    (a re-encode differs by a few levels at most)."""
    a, b = cv2.VideoCapture(video_path), cv2.VideoCapture(copy_path)
    try:
        n = int(a.get(cv2.CAP_PROP_FRAME_COUNT))
        if n < 1 or n != int(b.get(cv2.CAP_PROP_FRAME_COUNT)):
            return False
        for index in sorted({0, n // 3, (2 * n) // 3}):
            a.set(cv2.CAP_PROP_POS_FRAMES, index)
            b.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok_a, fa = a.read()
            ok_b, fb = b.read()
            if not (ok_a and ok_b) or fa.shape != fb.shape:
                return False
            if float(cv2.absdiff(fa, fb).mean()) > 3.0:
                return False
        return True
    finally:
        a.release()
        b.release()
