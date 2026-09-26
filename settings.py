import os
import tempfile

import yaml

# The output options are also kept next to the results (the output folder:
# Google Drive on Colab), so a new Colab session, which starts from a fresh
# clone and config_colab.yaml, keeps them. Machine options (provider, threads,
# server, output folder) stay in config.yaml only.
OVERLAY_FILE = 'unleashed_settings.yaml'
PORTABLE = {
    'output_template': str, 'output_image_format': str, 'output_video_format': str,
    'output_video_codec': str, 'video_quality': int, 'clear_output': bool, 'path_start': str,
    'faceset_average_mode': str, 'faceset_outlier_threshold': float,
}


def _number(value, default, kind=int):
    """A number from the config (a hand edit may leave null or text)."""
    try:
        return kind(value)
    except (TypeError, ValueError):
        return default


def _write_yaml(path, data):
    """Write through a temporary file and a rename: an exit in the middle of
    the write can no longer leave an empty config (which starts on defaults,
    e.g. a public link)."""
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix='.settings_', suffix='.yaml', dir=folder)
    try:
        with os.fdopen(fd, 'w') as f:
            yaml.dump(data, f)
        # mkstemp makes the file owner-only (0600): keep the old file's mode,
        # or the usual one for a new file (best effort: a Drive mount may
        # not take a chmod)
        try:
            try:
                mode = os.stat(path).st_mode & 0o777
            except OSError:
                mode = 0o644
            os.chmod(tmp, mode)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _port(value):
    """A server port from the config: 0 (= Gradio's default) unless valid.
    An emptied port box once saved 'server_port: null' and the app could not
    start again."""
    try:
        port = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return port if 0 < port < 65536 else 0


class Settings:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load()

    def default_get(_, data, name, default):
        value = default
        try:
            value = data.get(name, default)
        except:
            pass
        return value


    def load(self):
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except:
            data = None

        self.selected_theme = self.default_get(data, 'selected_theme', "Default")
        self.server_name = self.default_get(data, 'server_name', "") or ""
        self.server_port = _port(self.default_get(data, 'server_port', 0))
        self.server_share = self.default_get(data, 'server_share', True)
        self.server_share_saved = self.server_share     # what save() writes (a --server_share start does not)
        self.output_image_format = self.default_get(data, 'output_image_format', 'png')
        self.output_video_format = self.default_get(data, 'output_video_format', 'mp4')
        self.output_video_codec = self.default_get(data, 'output_video_codec', 'libx264')
        self.video_quality = _number(self.default_get(data, 'video_quality', 18), 18)
        self.clear_output = self.default_get(data, 'clear_output', False)
        self.max_threads = _number(self.default_get(data, 'max_threads', 8), 8)
        self.memory_limit = _number(self.default_get(data, 'memory_limit', 0), 0)
        self.provider = self.default_get(data, 'provider', 'cuda')
        # what the config says, saved back as it is: a session without a GPU
        # uses 'cpu' in memory but must not write it into the config
        self.provider_configured = self.provider
        self.force_cpu = self.default_get(data, 'force_cpu', False)
        self.output_template = self.default_get(data, 'output_template', '{file}_{time}')
        self.output_folder = self.default_get(data, 'output_folder', '/content/drive/MyDrive/c')
        # no longer offered (uploads stayed in ./temp while the cleaning was
        # switched off): always off
        self.use_os_temp_folder = False
        self.output_show_video = self.default_get(data, 'output_show_video', True)
        self.launch_browser = self.default_get(data, 'launch_browser', True)
        self.allowed_paths = self.default_get(data, 'allowed_paths', ['/content/drive/'])
        # Face Swap: the source / target path boxes start with this folder
        # (only where it exists, so a local machine gets empty boxes)
        self.path_start = self.default_get(data, 'path_start', '/content/drive/MyDrive/c/')
        # Settings > Faceset: how the photos of a faceset make one identity
        self.faceset_average_mode = self.default_get(data, 'faceset_average_mode', 'robust')
        try:
            self.faceset_outlier_threshold = float(self.default_get(data, 'faceset_outlier_threshold', 0.6))
        except (TypeError, ValueError):
            self.faceset_outlier_threshold = 0.6


    def load_overlay(self, folder):
        """The output options saved next to the results (see OVERLAY_FILE), on
        top of config.yaml. Unknown or mistyped entries are ignored."""
        if not folder:
            return False
        try:
            with open(os.path.join(folder, OVERLAY_FILE), 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            return False
        if not isinstance(data, dict):
            return False
        for key, kind in PORTABLE.items():
            if key not in data:
                continue
            value = data[key]
            if kind is float and isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(self, key, float(value))
            elif kind is int and isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(self, key, int(value))
            elif isinstance(value, kind):
                setattr(self, key, value)
        return True


    def save(self, output_folder=None):
        """config.yaml, and the output options next to the results in
        output_folder (when given)."""
        data = {
            'selected_theme': self.selected_theme,
            'server_name': self.server_name,
            'server_port': _port(self.server_port),
            # the --server_share command line flag is for that start only
            'server_share': getattr(self, 'server_share_saved', self.server_share),
            'output_image_format' : self.output_image_format,
            'output_video_format' : self.output_video_format,
            'output_video_codec' : self.output_video_codec,
            'video_quality' : self.video_quality,
            'clear_output' : self.clear_output,
            'max_threads' : self.max_threads,
            'memory_limit' : self.memory_limit,
            'provider' : getattr(self, 'provider_configured', self.provider),
            'force_cpu' : self.force_cpu,
            'output_template' : self.output_template,
            'output_folder' : self.output_folder,
            'use_os_temp_folder' : self.use_os_temp_folder,
            'output_show_video' : self.output_show_video,
            'launch_browser' : self.launch_browser,
            'allowed_paths' : list(self.allowed_paths or []),
            'path_start' : self.path_start,
            'faceset_average_mode': self.faceset_average_mode,
            'faceset_outlier_threshold': self.faceset_outlier_threshold,
        }
        _write_yaml(self.config_file, data)
        if output_folder and os.path.isdir(output_folder):
            try:
                _write_yaml(os.path.join(output_folder, OVERLAY_FILE), {k: data[k] for k in PORTABLE})
            except OSError as e:
                print(f'[settings] not saved next to the results ({e})')
