import glob
import mimetypes
import os
import platform
import shutil
import tempfile
import ssl
import subprocess
import sys
import urllib
try:
    import torch        # optional: CUDA checks and Clip2Seg; not installed on Intel Macs
except ImportError:
    torch = None
import gradio
import cv2
import zipfile
import traceback
import threading
import threading
import random

from typing import Union, Any
from contextlib import nullcontext

from pathlib import Path
from typing import List, Any
from tqdm import tqdm
from scipy.spatial import distance

import unleashed.template_parser as template_parser

import unleashed.globals

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"

THREAD_SEMAPHORE = threading.Semaphore()
NULL_CONTEXT  = nullcontext()


# monkey patch ssl for mac
if platform.system().lower() == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


# https://github.com/facefusion/facefusion/blob/master/facefusion
def detect_fps(target_path: str) -> float:
    fps = 24.0
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


# Gradio wants Images in RGB
def convert_to_gradio(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_gradio_preview(image):
    """Like convert_to_gradio, but downscales large frames to unleashed.globals.
    preview_max_height first. This shrinks the payload Gradio must send to the
    browser (especially over a gradio.live share tunnel), so the preview appears
    quickly. Preview only -- final renders are produced from the full-res frame.
    """
    if image is None:
        return None
    import unleashed.globals
    max_h = int(getattr(unleashed.globals, 'preview_max_height', 720) or 0)
    h = image.shape[0]
    if max_h > 0 and h > max_h:
        scale = max_h / float(h)
        new_w = max(1, int(round(image.shape[1] * scale)))
        image = cv2.resize(image, (new_w, max_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sort_rename_frames(path: str):
    filenames = os.listdir(path)
    filenames.sort()
    for i in range(len(filenames)):
        of = os.path.join(path, filenames[i])
        newidx = i + 1
        new_filename = os.path.join(
            path, f"{newidx:06d}." + unleashed.globals.CFG.output_image_format
        )
        os.rename(of, new_filename)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob(
        (
            os.path.join(
                glob.escape(temp_directory_path),
                f"*.{unleashed.globals.CFG.output_image_format}",
            )
        )
    )


def get_temp_directory_path(target_path: str) -> str:
    """The folder for the extracted frames of one video: in the app's temp
    folder (./temp, emptied at every start), not next to the video. There they
    were written to the user's folder (Drive on Colab) and the clean-up
    deleted that folder's whole 'temp' subfolder, user files included."""
    import hashlib
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    key = hashlib.md5(os.path.abspath(target_path).encode('utf-8', 'surrogateescape')).hexdigest()[:8]
    base = os.environ.get('TEMP') or tempfile.gettempdir()
    return os.path.join(base, 'frames', f'{target_name}_{key}')


def get_destfilename_from_path(
    srcfilepath: str, destfilepath: str, extension: str
) -> str:
    fn, ext = os.path.splitext(os.path.basename(srcfilepath))
    if "." in extension:
        return os.path.join(destfilepath, f"{fn}{extension}")
    return os.path.join(destfilepath, f"{fn}{extension}{ext}")


def replace_template(file_path: str, index: int = 0) -> str:
    fn, ext = os.path.splitext(os.path.basename(file_path))

    # Remove the "__temp" placeholder that was used as a temporary filename
    fn = fn.replace("__temp", "")

    template = unleashed.globals.CFG.output_template
    replaced_filename = template_parser.parse(
        template, {"index": str(index), "file": fn}
    )

    return os.path.join(unleashed.globals.output_path, f"{replaced_filename}{ext}")


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    if not unleashed.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path, ignore_errors=True)
    _remove_if_empty(os.path.dirname(temp_directory_path))


def delete_temp_frames(filename: str) -> None:
    """The frame folder of one video (the folder of this frame file) only."""
    folder = os.path.dirname(filename)
    shutil.rmtree(folder, ignore_errors=True)
    _remove_if_empty(os.path.dirname(folder))


def _remove_if_empty(folder):
    try:
        if os.path.isdir(folder) and not os.listdir(folder):
            os.rmdir(folder)
    except OSError:
        pass


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(("png", "jpg", "jpeg", "webp"))


def has_extension(filepath: str, extensions: List[str]) -> bool:
    return filepath.lower().endswith(tuple(extensions))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        if image_path.endswith(".webp"):
            return True
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(
            download_directory_path, os.path.basename(url)
        )
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get("Content-Length", 0))
            with tqdm(
                total=total,
                desc=f"Downloading {url}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


def get_local_files_from_folder(folder: str) -> List[str]:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return None
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]
    return files


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def get_device() -> str:
    if len(unleashed.globals.execution_providers) < 1:
        unleashed.globals.execution_providers = ["CPUExecutionProvider"]

    prov = unleashed.globals.execution_providers[0]
    if "CoreMLExecutionProvider" in prov:
        return "mps"
    if "CUDAExecutionProvider" in prov or "ROCMExecutionProvider" in prov:
        return "cuda"
    if "OpenVINOExecutionProvider" in prov:
        return "mkl"
    return "cpu"


def str_to_class(module_name, class_name) -> Any:
    from importlib import import_module

    class_ = None
    try:
        module_ = import_module(module_name)
        try:
            class_ = getattr(module_, class_name)()
        except AttributeError:
            print(f"Class {class_name} does not exist")
    except ImportError:
        print(f"Module {module_name} does not exist")
    return class_

def is_installed(name:str) -> bool:
    return shutil.which(name);

# Taken from https://stackoverflow.com/a/68842705
def get_platform() -> str:
    if sys.platform == "linux":
        try:
            proc_version = open("/proc/version").read()
            if "Microsoft" in proc_version:
                return "wsl"
        except:
            pass
    return sys.platform

def open_with_default_app(filename:str):
    if filename == None:
        return
    platform = get_platform()
    if platform == "darwin":
        subprocess.call(("open", filename))
    elif platform in ["win64", "win32"]:        os.startfile(filename.replace("/", "\\"))
    elif platform == "wsl":
        subprocess.call("cmd.exe /C start".split() + [filename])
    else:  # linux variants
        subprocess.call("xdg-open", filename)


def zip(files, zipname):
    with zipfile.ZipFile(zipname, "w") as zip_file:
        for f in files:
            zip_file.write(f, os.path.basename(f))


def unzip(zipfilename: str, target_path: str):
    with zipfile.ZipFile(zipfilename, "r") as zip_file:
        zip_file.extractall(target_path)


def mkdir_with_umask(directory):
    oldmask = os.umask(0)
    # mode needs octal
    os.makedirs(directory, mode=0o775, exist_ok=True)
    os.umask(oldmask)


def open_folder(path: str):
    platform = get_platform()
    try:
        if platform == "darwin":
            subprocess.call(("open", path))
        elif platform in ["win64", "win32"]:
            open_with_default_app(path)
        elif platform == "wsl":
            subprocess.call("cmd.exe /C start".split() + [path])
        else:  # linux variants
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        traceback.print_exc()
        pass
        # import webbrowser
        # webbrowser.open(url)


def create_version_html() -> str:
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    versions_html = f"""
python: <span title="{sys.version}">{python_version}</span>
•
torch: {getattr(torch, '__long_version__', torch.__version__) if torch is not None else 'not installed'}
•
gradio: {gradio.__version__}
"""
    return versions_html


def compute_cosine_distance(emb1, emb2) -> float:
    return distance.cosine(emb1, emb2)

def has_cuda_device():
    return torch is not None and torch.cuda is not None and torch.cuda.is_available()


def print_cuda_info():
    try:
        print(f'Number of CUDA devices: {torch.cuda.device_count()} Currently used Id: {torch.cuda.current_device()} Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    except:
       print('No CUDA device found!')

def folder_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def clean_temp_folder() -> int:
    """Empty the app's own temp folder (./temp): uploads cached by Gradio,
    quick-seek video copies, faceset thumbnails, leftovers. Never the OS temp
    folder ('Use OS temp folder' on). Returns the bytes freed."""
    import unleashed.globals
    path = os.environ.get('TEMP')
    if unleashed.globals.CFG.use_os_temp_folder or not path or not os.path.isdir(path):
        return 0
    before = folder_size(path)
    clean_dir(path)
    try:
        from unleashed.capturer import forget_seek_copies
        forget_seek_copies()
    except Exception:
        pass
    return max(0, before - folder_size(path))


def clean_dir(path: str):
    contents = os.listdir(path)
    for item in contents:
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(e)
            

def conditional_thread_semaphore() -> Union[Any, Any]:
    if 'DmlExecutionProvider' in unleashed.globals.execution_providers or 'ROCMExecutionProvider' in unleashed.globals.execution_providers:
        return THREAD_SEMAPHORE
    return NULL_CONTEXT

def shuffle_array(arr):
  """
  Shuffles the given array in place using the Fisher-Yates shuffle algorithm.

  Args:
    arr: The array to be shuffled.

  Returns:
    None. The array is shuffled in place.
  """
  for i in range(len(arr) - 1, 0, -1):
    j = random.randint(0, i)
    arr[i], arr[j] = arr[j], arr[i]


def tuned_execution_providers():
    """Return unleashed.globals.execution_providers with the CUDA options from
    unleashed.globals merged in (see the notes there for the trade-offs):
      - cudnn_conv_algo_search       (globals default: EXHAUSTIVE)
      - cudnn_conv_use_max_workspace (globals default: True; False saves GBs of
        GPU memory per session when many models are resident, at a speed cost)
    We deliberately do NOT force arena_extend_strategy=kSameAsRequested anymore:
    it can fragment the arena and OOM (onnxruntime issue #14474); the default
    kNextPowerOfTwo reuses blocks better. Safe for non-CUDA providers."""
    import unleashed.globals
    provs = unleashed.globals.execution_providers or []
    search = getattr(unleashed.globals, 'cudnn_conv_algo_search', 'HEURISTIC')
    maxws = '1' if getattr(unleashed.globals, 'cudnn_conv_use_max_workspace', False) else '0'
    extra = {'cudnn_conv_algo_search': search, 'cudnn_conv_use_max_workspace': maxws}
    out = []
    for p in provs:
        if isinstance(p, (tuple, list)) and len(p) == 2 and 'CUDA' in str(p[0]):
            opts = dict(p[1]) if isinstance(p[1], dict) else {}
            for k, v in extra.items():
                opts.setdefault(k, v)
            out.append((p[0], opts))
        elif isinstance(p, str) and 'CUDA' in p:
            out.append((p, dict(extra)))
        else:
            out.append(p)
    return out
