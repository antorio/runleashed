import threading
from collections import OrderedDict
from typing import Optional
import cv2
import numpy as np

from roop.typing import Frame

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


def get_image_frame(filename: str):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        print(f"Exception reading {filename}")
    return None


def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:
    global current_video_path, current_capture, current_frame_total, current_position

    with _lock:
        if video_path != current_video_path or current_capture is None:
            _release_video()
            current_capture = cv2.VideoCapture(video_path)
            current_video_path = video_path
            current_frame_total = current_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            current_position = 0

        index = frame_number - 1
        if index < 0 or index >= current_frame_total:
            # outside the reported frame range: exactly the old clamped seek, uncached
            current_capture.set(cv2.CAP_PROP_POS_FRAMES, min(current_frame_total, index))
            current_position = None
            has_frame, frame = current_capture.read()
            return frame if has_frame else None

        frame = _frames.get(index)
        if frame is not None:
            _frames.move_to_end(index)
            return frame.copy()

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
        _remember(index, frame)
        return frame.copy()


def _remember(index, frame):
    global _frames_bytes

    _frames[index] = frame
    _frames_bytes += frame.nbytes
    while _frames_bytes > _CACHE_BYTES and len(_frames) > 1:
        _, oldest = _frames.popitem(last=False)
        _frames_bytes -= oldest.nbytes


def release_video():
    with _lock:
        _release_video()


def _release_video():
    global current_capture, current_video_path, current_position, _frames_bytes

    if current_capture is not None:
        current_capture.release()
        current_capture = None
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
