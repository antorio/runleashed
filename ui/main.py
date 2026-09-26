import os
import time
import gradio as gr
import unleashed.globals
import unleashed.metadata
import unleashed.utilities as util
import ui.globals as uii

from ui.tabs.faceswap_tab import faceswap_tab
from ui.tabs.facemgr_tab import facemgr_tab
from ui.tabs.settings_tab import settings_tab
from ui.theme import unleashed_theme, unleashed_css, unleashed_js

unleashed.globals.keep_fps = None
unleashed.globals.keep_frames = None
unleashed.globals.skip_audio = None
unleashed.globals.use_batch = None


def prepare_environment():
    _out = getattr(unleashed.globals.CFG, 'output_folder', None)
    candidate = str(_out).strip() if (_out and str(_out).strip()) else None
    fallback = os.path.abspath(os.path.join(os.getcwd(), "output"))
    chosen = None
    if candidate:
        try:
            os.makedirs(candidate, exist_ok=True)
            chosen = candidate
        except OSError:
            # Configured folder isn't available here (e.g. a Colab '/content'
            # path on a local machine). Fall back to ./output instead of crashing.
            print(f"[unleashed] output folder '{candidate}' is not writable here; "
                  f"falling back to '{fallback}'.")
    if chosen is None:
        os.makedirs(fallback, exist_ok=True)
        chosen = fallback
    unleashed.globals.output_path = chosen
    if not unleashed.globals.CFG.use_os_temp_folder:
        os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(os.path.join(os.getcwd(), "temp"))
    os.makedirs(os.environ.get("TEMP", os.path.abspath(os.path.join(os.getcwd(), "temp"))), exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = os.environ.get("TEMP", os.path.abspath(os.path.join(os.getcwd(), "temp")))
    os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'

def run():
    from unleashed.core import decode_execution_providers, set_display_ui

    prepare_environment()
    # the output options saved next to the results (a new Colab session starts
    # from config_colab.yaml): see settings.py
    if unleashed.globals.CFG.load_overlay(unleashed.globals.output_path):
        print(f'[settings] output options from {unleashed.globals.output_path}')
    unleashed.globals.faceset_average_mode = unleashed.globals.CFG.faceset_average_mode
    unleashed.globals.faceset_outlier_threshold = unleashed.globals.CFG.faceset_outlier_threshold
    # nothing in ./temp outlives a run (uploads, video copies, thumbnails):
    # start with it empty
    freed = util.clean_temp_folder()
    if freed:
        print(f'[temp] emptied {os.environ.get("TEMP")}: {freed / 1e6:.0f} MB from the last run')

    set_display_ui(show_msg)
    if unleashed.globals.CFG.provider == "cuda" and util.has_cuda_device() == False:
       unleashed.globals.CFG.provider = "cpu"

    unleashed.globals.execution_providers = decode_execution_providers([unleashed.globals.CFG.provider])
    gputype = util.get_device()
    if gputype == 'cuda':
        util.print_cuda_info()
        
    print(f'Using provider {unleashed.globals.execution_providers} - Device:{gputype}')

    # Download (first run) and build the face analyser in the background while
    # the UI starts, so the first preview / "Use face from the frame" finds it
    # ready. The analyser lock makes any early click wait for it instead of
    # starting a second download.
    def _warm_up():
        try:
            from unleashed.face_util import get_face_analyser
            get_face_analyser()
            print('[load] face analyser ready')
        except Exception as e:
            print(f'[load] face analyser not ready at startup ({e}); it will be built on first use')
    import threading
    threading.Thread(target=_warm_up, name='analyser-warmup', daemon=True).start()
    
    run_server = True
    uii.ui_restart_server = False

    while run_server:
        server_name = unleashed.globals.CFG.server_name
        if server_name is None or len(server_name) < 1:
            server_name = None
        try:
            server_port = int(unleashed.globals.CFG.server_port or 0)
        except (TypeError, ValueError):
            server_port = 0
        if not 0 < server_port < 65536:
            server_port = None
        # a copy: the additions below are for this launch, not for config.yaml
        allowed_paths = list(unleashed.globals.CFG.allowed_paths or [])
        # Always allow the actual output dir and the working dir so local runs can
        # serve uploads/results (Colab's /content/drive entry stays harmless).
        for _p in (unleashed.globals.output_path, os.getcwd()):
            if _p and _p not in allowed_paths:
                allowed_paths.append(_p)
        ssl_verify = False if server_name == '0.0.0.0' else True
        with gr.Blocks(title=f'{unleashed.metadata.name} v{unleashed.metadata.version}',
                       theme=unleashed_theme, css=unleashed_css, js=unleashed_js) as ui:
            # (no delete_cache: at Restart Server it deleted every upload while
            # the Face Swap list still named them; ./temp is emptied at start)
            # ---- header: bold title + version, then env badges inline ----
            with gr.Row(variant='compact', elem_id="app_header"):
                gr.Markdown(f"# {unleashed.metadata.name} v{unleashed.metadata.version}")
                gr.HTML(util.create_version_html(), elem_id="versions")
            faceswap_load = faceswap_tab()
            facemgr_load = facemgr_tab()
            settings_tab()
            gr.HTML('<div class="rl-footer">Use via API</div>')
            faceswap_load(ui)
            facemgr_load(ui)
        launch_browser = unleashed.globals.CFG.launch_browser

        uii.ui_restart_server = False
        try:
            ui.queue().launch(inbrowser=launch_browser, server_name=server_name, server_port=server_port, share=unleashed.globals.CFG.server_share, ssl_verify=ssl_verify, prevent_thread_lock=True, show_error=True, allowed_paths=allowed_paths)
        except Exception as e:
            print(f'Exception {e} when launching Gradio Server!')
            uii.ui_restart_server = True
            run_server = False
        try:
            while uii.ui_restart_server == False:
                time.sleep(1.0)

        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()


def show_msg(msg: str):
    gr.Info(msg)
