
import os
import subprocess
import unleashed.globals
import unleashed.utilities as util

from typing import List
def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-y', '-loglevel', unleashed.globals.log_level]
    commands.extend(args)
    print ("Running ffmpeg")
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print("Running ffmpeg failed! Commandline:")
        print (" ".join(commands))
    return False



def cut_video(original_video: str, cut_video: str, start_frame: int, end_frame: int, reencode: bool):
    fps = util.detect_fps(original_video)
    start_time = start_frame / fps
    num_frames = end_frame - start_frame

    if reencode:
        run_ffmpeg(['-ss',  format(start_time, ".2f"), '-i', original_video, '-c:v', unleashed.globals.video_encoder, '-c:a', 'aac', '-frames:v', str(num_frames), cut_video])
    else:
        run_ffmpeg(['-ss',  format(start_time, ".2f"), '-i', original_video,  '-frames:v', str(num_frames), '-c:v' ,'copy','-c:a' ,'copy', cut_video])

def extract_frames(target_path : str, trim_frame_start, trim_frame_end, fps : float) -> bool:
    util.create_temp(target_path)
    temp_directory_path = util.get_temp_directory_path(target_path)
    commands = ['-i', target_path, '-q:v', '1', '-pix_fmt', 'rgb24', ]
    if trim_frame_start is not None and trim_frame_end is not None:
        commands.extend([ '-vf', 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(trim_frame_end) + ',fps=' + str(fps) ])
    out = os.path.join(temp_directory_path, '%06d.' + unleashed.globals.CFG.output_image_format)
    # every frame passed through as it is: -fps_mode since ffmpeg 5.1, -vsync
    # before (removed in ffmpeg 9)
    return run_ffmpeg(commands + ['-fps_mode', 'passthrough', out]) or run_ffmpeg(commands + ['-vsync', '0', out])


def create_video(target_path: str, dest_filename: str, fps: float = 24.0, temp_directory_path: str = None) -> bool:
    """The swapped frames as a video; False when ffmpeg failed."""
    if temp_directory_path is None:
        temp_directory_path = util.get_temp_directory_path(target_path)
    frames = os.path.join(temp_directory_path, f'%06d.{unleashed.globals.CFG.output_image_format}')
    vf = 'colorspace=bt709:iall=bt601-6-625:fast=1'
    first = os.path.join(temp_directory_path, f'000001.{unleashed.globals.CFG.output_image_format}')
    size = _image_size(first)
    if size and (size[0] % 2 or size[1] % 2):
        # yuv420p needs an even width and height (a 480x853 portrait failed
        # and left a 0-byte file); the in-memory writer does the same
        vf = 'scale=trunc(iw/2)*2:trunc(ih/2)*2,' + vf
    return run_ffmpeg(['-r', str(fps), '-i', frames, '-c:v', unleashed.globals.video_encoder, '-crf', str(unleashed.globals.video_quality),
                       '-pix_fmt', 'yuv420p', '-vf', vf, '-y', dest_filename])


def _image_size(path):
    """(width, height) of an image file, or None."""
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return (img.shape[1], img.shape[0]) if img is not None else None
    except Exception:
        return None


def create_gif_from_video(video_path: str, gif_path) -> bool:
    """The rendered video as a GIF at its own size; False when ffmpeg failed.
    (The old scale filter swapped width and height: every GIF came out
    enlarged by its aspect ratio, e.g. 1280x886 -> 1849x1280.)"""
    fps = util.detect_fps(video_path)
    return run_ffmpeg(['-i', video_path, '-vf', f'fps={fps},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                       '-loop', '0', gif_path])



def restore_audio(intermediate_video: str, original_video: str, trim_frame_start, trim_frame_end, final_video: str,
                  audio_codec: str = 'copy') -> bool:
    """The rendered video with the original's sound for the same range;
    False when ffmpeg failed. The video stream is copied untouched.

    No -shortest: it ended the file with the shorter stream, so a source
    whose sound stops early cut the rendered video (down to 0 frames). The
    sound is capped at the video's length instead (-t)."""
    fps = util.detect_fps(original_video)
    start = trim_frame_start or 0
    commands = ['-i', intermediate_video, '-ss', format(start / fps, ".2f")]
    if trim_frame_end is not None:
        commands.extend(['-to', format(trim_frame_end / fps, ".2f")])
    commands.extend(['-i', original_video, '-c:v', 'copy', '-c:a', audio_codec, '-map', '0:v:0', '-map', '1:a:0?'])
    if trim_frame_end is not None:
        commands.extend(['-t', format((trim_frame_end - start) / fps, ".3f")])
    commands.append(final_video)
    return run_ffmpeg(commands)
