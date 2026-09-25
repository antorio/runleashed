#!/usr/bin/env python3

import os
import sys
import shutil
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from typing import List
import platform
import signal
try:
    import torch        # CUDA device selection only; a CPU-only install (Intel Mac) has none
except ImportError:
    torch = None
import onnxruntime
import pathlib
import threading
import argparse

from time import time

import roop.globals
import roop.metadata
import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.capturer import get_video_frame_total

clip_text = None

call_display_ui = None

process_mgr = None


if 'ROCMExecutionProvider' in roop.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    roop.globals.headless = False

    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('--server_share', help='Public server', dest='server_share', action='store_true', default=False)
    program.add_argument('--cuda_device_id', help='Index of the cuda gpu to use', dest='cuda_device_id', type=int, default=0)
    roop.globals.startup_args = program.parse_args()
    # Always enable all processors when using GUI
    roop.globals.frame_processors = ['face_swapper', 'face_enhancer']


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    list_providers = [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]
    
    try:
        for i in range(len(list_providers)):
            if list_providers[i] == 'CUDAExecutionProvider':
                list_providers[i] = ('CUDAExecutionProvider', {'device_id': roop.globals.cuda_device_id})
                if torch is not None:
                    torch.cuda.set_device(roop.globals.cuda_device_id)
                break
    except:
        pass

    return list_providers
    


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in roop.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in roop.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))



def release_resources(keep_models=False) -> None:
    import gc
    global process_mgr

    if process_mgr is not None:
        # keep_models: the loaded model sessions are parked and picked up by the
        # next preview/render instead of being rebuilt (see ProcessMgr)
        process_mgr.release_resources(keep_models=keep_models)
        process_mgr = None

    gc.collect()
    # if 'CUDAExecutionProvider' in roop.globals.execution_providers and torch.cuda.is_available():
    #     with torch.cuda.device('cuda'):
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False

    # All model weights are hosted in ONE HuggingFace repo (mirrored there from
    # the original countfloyd/deepfake + facefusion-assets sources). HF's CDN is
    # reliable from Colab/datacenter IPs -- unlike GitHub release assets, which
    # rate-limit those IPs and were the cause of the "download stuck from Colab
    # but works in a browser" hangs. hf_hub_download also resumes + retries, so a
    # dropped connection no longer restarts from zero.
    HF_REPO = 'antorio/runleashed-models'
    models_root = util.resolve_relative_path('../models')

    # (filename in the HF repo root, local subdir under ../models)
    model_files = [
        ('inswapper_128.onnx', ''),
        ('GFPGANv1.4.onnx', ''),
        ('restoreformer_plus_plus.onnx', ''),
        ('xseg.onnx', ''),
        ('xseg_1.onnx', ''),
        ('xseg_2.onnx', ''),
        ('xseg_3.onnx', ''),
        ('bisenet_resnet_34.onnx', ''),
        ('2dfan4.onnx', ''),
        ('live_portrait_feature_extractor.onnx', 'liveportrait'),
        ('live_portrait_motion_extractor.onnx', 'liveportrait'),
        ('live_portrait_generator.onnx', 'liveportrait'),
        ('live_portrait_stitcher.onnx', 'liveportrait'),
        ('rd64-uni-refined.pth', 'CLIP'),
        ('CodeFormerv0.1.onnx', 'CodeFormer'),
    ]

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        hf_hub_download = None

    for filename, subdir in model_files:
        target_dir = os.path.join(models_root, subdir) if subdir else models_root
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(target_path):
            continue
        got = False
        if hf_hub_download is not None:
            try:
                # straight into ../models[/subdir]/<filename>, the path the
                # loaders expect (no second copy in the HF cache: ~2.5 GB)
                got_path = hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=target_dir)
                if os.path.abspath(got_path) != os.path.abspath(target_path):
                    shutil.copyfile(got_path, target_path)
                got = True
            except Exception as e:
                update_status(f'HF download failed for {filename} ({e}); falling back to direct URL')
        if not got:
            # Fallback: direct resolve URL from the same HF repo (still HF CDN).
            util.conditional_download(target_dir, [f'https://huggingface.co/{HF_REPO}/resolve/main/{filename}'])

    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True

def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)




def start() -> None:
    if roop.globals.headless:
        print('Headless mode currently unsupported - starting UI!')
        # faces = extract_face_images(roop.globals.source_path,  (False, 0))
        # roop.globals.INPUT_FACES.append(faces[roop.globals.source_face_index])
        # faces = extract_face_images(roop.globals.target_path,  (False, util.has_image_extension(roop.globals.target_path)))
        # roop.globals.TARGET_FACES.append(faces[roop.globals.target_face_index])
        # if 'face_enhancer' in roop.globals.frame_processors:
        #     roop.globals.selected_enhancer = 'GFPGAN'
       
    batch_process_regular(None, False, None)


def get_processing_plugins(masking_engine):
    processors = {  "faceswap": {}}

    # Expression restorer runs right after the swap.
    if getattr(roop.globals, 'expression_restorer', False):
        processors.update({"expression_restorer": {}})

    # Occlusion mask runs AFTER the expression restorer (so the ER never warps the
    # restored occluders like hands/hair). The mask/enhancer order is controlled
    # SOLELY by the user's 'mask_after_enhancer' toggle, for EVERY enhancer
    # (GFPGAN, CodeFormer, RestoreFormer++): off = mask before the enhancer;
    # on (the default in globals.py) = mask runs last.
    # Note: RestoreFormer++ / CodeFormer are codebook restorers that can colour-burn
    # when they enhance a crop that still contains a restored occluder (non-face
    # pixels). If you use them on footage WITH occluders and see colour burn, tick
    # 'mask after enhancer' so they only ever see the clean swapped face. We do NOT
    # force this anymore -- e.g. frontal / occluder-free clips don't need it, and
    # forcing it added slight jitter -- so the choice is yours.
    mask_after_enhancer = getattr(roop.globals, 'mask_after_enhancer', False)
    if masking_engine is not None and not mask_after_enhancer:
        processors.update({masking_engine: {}})

    if roop.globals.selected_enhancer == 'GFPGAN':
        processors.update({"gfpgan": {}})
    elif roop.globals.selected_enhancer == 'Codeformer':
        processors.update({"codeformer": {}})
    elif roop.globals.selected_enhancer == 'Restoreformer++':
        processors.update({"restoreformer++": {}})

    if masking_engine is not None and mask_after_enhancer:
        processors.update({masking_engine: {}})
    return processors


# One ProcessMgr serves the preview and the render. The lock keeps a preview
# from being re-initialised halfway through a frame, and a render from starting
# under a preview; render_active keeps any preview (and the settings it would
# apply) away from a running render, whose per-frame globals and ProcessMgr
# state (frame sequencer, stabiliser, tracks) must not change.
_swap_lock = threading.RLock()
render_active = False
mask_view_mgr = None
# Stop pressed before batch_process starts (the render still waits for a
# preview or loads its models): batch_process sets processing = True, which
# used to wipe that Stop out. Set by the UI's Stop, cleared at its Start.
stop_requested = False


def preview_locked(work):
    """Run work() for a preview unless a render is running (or starting).
    Returns (ran, result); never waits for a render."""
    if not _swap_lock.acquire(blocking=False):
        return False, None
    try:
        if render_active:
            return False, None
        return True, work()
    finally:
        _swap_lock.release()


def live_swap(frame, options):
    global process_mgr

    if frame is None:
        return frame

    with _swap_lock:
        if process_mgr is None:
            process_mgr = ProcessMgr(None)
        process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)
        newframe = process_mgr.process_frame(frame)
    if newframe is None:
        return frame
    return newframe


def mask_view(frame, options):
    """The frame with the area the swap replaces tinted green and occluders
    (kept original) cut out of it, no swap. Its own ProcessMgr, so the main
    one keeps its restorer / enhancer loaded."""
    global mask_view_mgr
    if frame is None:
        return frame
    with _swap_lock:
        if mask_view_mgr is None:
            mask_view_mgr = ProcessMgr(None)
        mask_view_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options, release_idle=False)
        newframe = mask_view_mgr.process_frame(frame)
    return frame if newframe is None else newframe


def batch_process_regular(swap_model, output_method, files:list[ProcessEntry], masking_engine:str, new_clip_text:str, use_new_method, imagemask, restore_original_mouth, restore_original_eyes, num_swap_steps, progress, selected_index = 0) -> None:
    global clip_text, process_mgr, render_active

    with _swap_lock:            # waits for a preview that is still running
        render_active = True
    try:
        _batch_process_regular(swap_model, output_method, files, masking_engine, new_clip_text, use_new_method, imagemask,
                               restore_original_mouth, restore_original_eyes, num_swap_steps, progress, selected_index)
    finally:
        render_active = False


def _batch_process_regular(swap_model, output_method, files, masking_engine, new_clip_text, use_new_method, imagemask, restore_original_mouth, restore_original_eyes, num_swap_steps, progress, selected_index):
    global clip_text, process_mgr

    with _swap_lock:
        release_resources(keep_models=True)
        limit_resources()
        if process_mgr is None:
            process_mgr = ProcessMgr(progress)
        mask = imagemask["layers"][0] if isinstance(imagemask, dict) else imagemask
        if len(roop.globals.INPUT_FACESETS) <= selected_index:
            selected_index = 0
        options = ProcessOptions(swap_model, get_processing_plugins(masking_engine), roop.globals.distance_threshold, roop.globals.blend_ratio,
                                  roop.globals.face_swap_mode, selected_index, new_clip_text, mask, num_swap_steps,
                                  roop.globals.subsample_size, False, restore_original_mouth, restore_original_eyes=restore_original_eyes)
        process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)
    batch_process(output_method, files, use_new_method)
    return

def batch_process(output_method, files:list[ProcessEntry], use_new_method) -> None:
    global clip_text, process_mgr

    roop.globals.processing = True
    if stop_requested:
        roop.globals.processing = False
        end_processing('Processing stopped!')
        return
    run_started = time()

    def fresh(path):
        # written by this run (a file of an earlier run with the same name,
        # e.g. an output template without {time}, does not count)
        return bool(path) and os.path.isfile(path) and os.path.getmtime(path) >= run_started - 1

    # limit threads for some providers
    max_threads = suggest_execution_threads()
    if max_threads == 1:
        roop.globals.execution_threads = 1

    imagefiles:list[ProcessEntry] = []
    videofiles:list[ProcessEntry] = []
           
    update_status('Sorting videos/images')

    used = set()
    def unique(path, marker=''):
        # two targets with the same file name (other folders, or the same second
        # in {time}) would write one output: only then add _2, _3, ...
        if path not in used:
            used.add(path)
            return path
        stem, ext = os.path.splitext(path)
        if marker and stem.endswith(marker):
            stem = stem[:-len(marker)]
        n = 2
        while f'{stem}_{n}{marker}{ext}' in used:
            n += 1
        path = f'{stem}_{n}{marker}{ext}'
        used.add(path)
        return path

    for index, f in enumerate(files):
        fullname = f.filename
        if util.has_image_extension(fullname):
            destination = util.get_destfilename_from_path(fullname, roop.globals.output_path, f'.{roop.globals.CFG.output_image_format}')
            destination = unique(util.replace_template(destination, index=index))
            pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
            f.finalname = destination
            imagefiles.append(f)

        elif util.is_video(fullname) or util.has_extension(fullname, ['gif']):
            destination = util.get_destfilename_from_path(fullname, roop.globals.output_path, f'__temp.{roop.globals.CFG.output_video_format}')
            f.finalname = unique(destination, '__temp')
            videofiles.append(f)



    if(len(imagefiles) > 0):
        update_status('Processing image(s)')
        # a painted keep-original mask belongs to the file it was painted on
        plain = [f for f in imagefiles if getattr(f, 'imagemask', None) is None]
        masked = [f for f in imagefiles if getattr(f, 'imagemask', None) is not None]
        if plain:
            process_mgr.set_manual_mask(None)
            process_mgr.run_batch([f.filename for f in plain], [f.finalname for f in plain], roop.globals.execution_threads)
        for f in masked:
            if not roop.globals.processing:
                break
            process_mgr.set_manual_mask(f.imagemask)
            process_mgr.run_batch([f.filename], [f.finalname], roop.globals.execution_threads)
        process_mgr.set_manual_mask(None)
        for f in imagefiles:
            # "Drop the frame" writes no file for an image
            f.completed = fresh(f.finalname)

    if(len(videofiles) > 0):
        for index,v in enumerate(videofiles):
            if not roop.globals.processing:
                end_processing('Processing stopped!')
                return
            fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
            if v.endframe == 0:
                v.endframe = get_video_frame_total(v.filename)

            is_streaming_only = output_method == "Virtual Camera"
            if is_streaming_only == False:
                update_status(f'Creating {os.path.basename(v.finalname)} with {fps} FPS...')

            start_processing = time()
            if hasattr(v, 'imagemask'):
                process_mgr.set_manual_mask(v.imagemask)
            if is_streaming_only == False and roop.globals.keep_frames or not use_new_method:
                util.create_temp(v.filename)
                update_status('Extracting frames...')
                ffmpeg.extract_frames(v.filename,v.startframe,v.endframe, fps)
                if not roop.globals.processing:
                    end_processing('Processing stopped!')
                    return

                temp_frame_paths = util.get_temp_frame_paths(v.filename)
                process_mgr.run_batch(temp_frame_paths, temp_frame_paths, roop.globals.execution_threads)
                if not roop.globals.processing:
                    end_processing('Processing stopped!')
                    return
                if roop.globals.wait_after_extraction:
                    extract_path = os.path.dirname(temp_frame_paths[0])
                    util.open_folder(extract_path)
                    input("Press any key to continue...")
                    print("Resorting frames to create video")
                    util.sort_rename_frames(extract_path)                                    
                
                ffmpeg.create_video(v.filename, v.finalname, fps)
                if not roop.globals.keep_frames:
                    util.delete_temp_frames(temp_frame_paths[0])
            else:
                if util.has_extension(v.filename, ['gif']):
                    skip_audio = True
                else:
                    skip_audio = roop.globals.skip_audio
                process_mgr.run_batch_inmem(output_method, v.filename, v.finalname, v.startframe, v.endframe, fps,roop.globals.execution_threads)
                
            if not roop.globals.processing:
                end_processing('Processing stopped!')
                return
            
            video_file_name = v.finalname
            if os.path.isfile(video_file_name):
                destination = ''
                if util.has_extension(v.filename, ['gif']):
                    # from finalname, which carries the _2 of a clashing name
                    # (replace_template drops its __temp); same name otherwise
                    gifname = os.path.splitext(v.finalname)[0] + '.gif'
                    destination = util.replace_template(gifname, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    update_status('Creating final GIF')
                    ffmpeg.create_gif_from_video(video_file_name, destination)
                    if os.path.isfile(destination):
                        os.remove(video_file_name)
                else:
                    skip_audio = roop.globals.skip_audio
                    destination = util.replace_template(video_file_name, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    if not skip_audio:
                        ffmpeg.restore_audio(video_file_name, v.filename, v.startframe, v.endframe, destination)
                        if os.path.isfile(destination):
                            os.remove(video_file_name)
                    else:
                        shutil.move(video_file_name, destination)
                # the finished file (finalname was the intermediate __temp file,
                # which is gone now): this is what the UI shows as the result
                if destination and fresh(destination):
                    v.finalname = destination
                    v.completed = True

            elif is_streaming_only == False:
                update_status(f'Failed processing {os.path.basename(v.finalname)}!')
            elapsed_time = time() - start_processing
            average_fps = (v.endframe - v.startframe) / elapsed_time
            update_status(f'\nProcessing {os.path.basename(destination)} took {elapsed_time:.2f} secs, {average_fps:.2f} frames/s')
    end_processing('Finished')


def end_processing(msg:str):
    update_status(msg)
    roop.globals.target_folder_path = None
    release_resources(keep_models=True)


def destroy() -> None:
    if roop.globals.target_path:
        util.clean_temp(roop.globals.target_path)
    release_resources()        
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    roop.globals.CFG = Settings('config.yaml')
    roop.globals.cuda_device_id = roop.globals.startup_args.cuda_device_id
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    if roop.globals.startup_args.server_share:
        roop.globals.CFG.server_share = True
    main.run()
