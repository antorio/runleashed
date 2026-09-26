import os
import threading

import gradio as gr

import unleashed.globals
import ui.globals
from unleashed import template_parser

image_formats = ['jpg', 'png', 'webp']
video_formats = ['avi', 'mkv', 'mp4', 'webm']
video_codecs = ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']
providerlist = None

settings_controls = []

# ----------------------------------------------------------------------------- saving
# Every change is saved (config.yaml, and the output options also next to the
# results, see settings.py), shortly after the last one: a slider being dragged
# sends many values.
_save_timer = None
_save_lock = threading.Lock()          # the timer
_write_lock = threading.Lock()         # one save at a time (a slow Drive write)


def _save_now():
    with _write_lock:
        try:
            unleashed.globals.CFG.save(unleashed.globals.output_path)
        except Exception as e:
            print(f'[settings] not saved: {e}')


def _persist():
    global _save_timer
    with _save_lock:
        if _save_timer is not None:
            _save_timer.cancel()
        _save_timer = threading.Timer(0.5, _save_now)
        _save_timer.daemon = True
        _save_timer.start()


def _flush_pending_save():
    """At exit: a change made in the last half second is saved too."""
    global _save_timer
    with _save_lock:
        pending, _save_timer = _save_timer, None
    if pending is not None and pending.is_alive():
        pending.cancel()
        _save_now()


import atexit
atexit.register(_flush_pending_save)


# ----------------------------------------------------------------------------- layout

def settings_tab():
    from unleashed.core import suggest_execution_providers
    global providerlist
    settings_controls.clear()   # avoid stale-control accumulation across server restarts
    CFG = unleashed.globals.CFG

    providerlist = suggest_execution_providers()
    with gr.Tab("Settings"):
        gr.Markdown("### Settings\nChanges apply right away and are saved: in `config.yaml`, and the output "
                    "options also next to your results (so a new Colab session keeps them). Provider and the "
                    "server options take effect at the next start.")
        with gr.Row():
            # ---------------- COLUMN 1: Performance + Output ----------------
            with gr.Column():
                with gr.Accordion("Performance", open=True):
                    settings_controls.append(gr.Dropdown(providerlist, label="Provider", info="Used from the next app start",
                                                         value=lambda: unleashed.globals.CFG.provider, elem_id='provider', interactive=True))
                    max_threads = gr.Slider(1, 32, value=lambda: unleashed.globals.CFG.max_threads, label="Max. Number of Threads", step=1.0, interactive=True)
                    memory_limit = gr.Slider(0, 128, value=lambda: unleashed.globals.CFG.memory_limit, label="Max. Memory to use (GB)",
                                             info='0 = no limit (not applied on macOS)', step=1.0, interactive=True)
                    settings_controls.append(gr.Checkbox(label="Force CPU for Face Analyser", value=lambda: unleashed.globals.CFG.force_cpu,
                                                         elem_id='force_cpu', interactive=True))
                with gr.Accordion("Output", open=True):
                    output_template = gr.Textbox(label="Filename Output Template", lines=1, placeholder='{file}_{time}',
                                                 info=f"Placeholders: {template_parser.PLACEHOLDERS} (the extension is added). "
                                                      "Enter to apply.", value=lambda: unleashed.globals.CFG.output_template)
                    output_folder_box = gr.Textbox(label="Output Folder", lines=1,
                                                   info="Where results are saved (on Colab each new session starts from config_colab.yaml's). Enter to apply.",
                                                   value=lambda: unleashed.globals.output_path or CFG.output_folder)
                    path_start_box = gr.Textbox(label="Face Swap path boxes start with", lines=1, placeholder='/content/drive/MyDrive/c/',
                                                info="Filled in when this folder exists (next page load). Enter to apply.",
                                                value=lambda: getattr(unleashed.globals.CFG, 'path_start', ''))
                    settings_controls.append(gr.Dropdown(image_formats, label="Image Output Format", value=lambda: unleashed.globals.CFG.output_image_format,
                                                         elem_id='output_image_format', interactive=True))
                    settings_controls.append(gr.Dropdown(video_codecs, label="Video Codec", value=lambda: unleashed.globals.CFG.output_video_codec,
                                                         elem_id='output_video_codec', interactive=True))
                    settings_controls.append(gr.Dropdown(video_formats, label="Video Output Format", value=lambda: unleashed.globals.CFG.output_video_format,
                                                         elem_id='output_video_format', interactive=True))
                    video_quality = gr.Slider(0, 100, value=lambda: unleashed.globals.CFG.video_quality, label="Video Quality (crf)", step=1.0, interactive=True)
            # ---------------- COLUMN 2: Expression Restorer + Faceset ----------------
            with gr.Column():
                with gr.Accordion("Expression Restorer — experimental / debug (this session only)", open=True):
                    expr_global_controls = []
                    expr_global_controls.append(gr.Checkbox(label="Full LivePortrait pipeline (experimental)", value=lambda a='expression_full_pipeline': getattr(unleashed.globals, a), elem_id='expression_full_pipeline', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Stitching model (experimental)", value=lambda a='expression_stitching': getattr(unleashed.globals, a), elem_id='expression_stitching', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Serialize (stable at high threads)", value=lambda a='expression_serialize': getattr(unleashed.globals, a), elem_id='expression_serialize', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Debug log [expr-delta] (A/B in console)", value=lambda a='expression_debug': getattr(unleashed.globals, a), elem_id='expression_debug', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Profile timings (per-stage ms in console)", info="Prints [timing] per processor each frame -- use briefly to see which stage is slow, then turn off.", value=lambda a='profile_timings': getattr(unleashed.globals, a), elem_id='profile_timings', interactive=True))
                    expr_cal_scale = gr.Slider(0.80, 1.20, value=lambda a='expression_lp_cal_scale': getattr(unleashed.globals, a), step=0.005, label="Full pipeline · cal scale", info='paste-back size correction (1.0 = none)', interactive=True)
                    expr_cal_dx = gr.Slider(-0.15, 0.15, value=lambda a='expression_lp_cal_dx': getattr(unleashed.globals, a), step=0.005, label="Full pipeline · cal shift X", info='horizontal nudge (fraction of crop)', interactive=True)
                    expr_cal_dy = gr.Slider(-0.15, 0.15, value=lambda a='expression_lp_cal_dy': getattr(unleashed.globals, a), step=0.005, label="Full pipeline · cal shift Y", info='vertical nudge (fraction of crop)', interactive=True)
                    expr_cal_rot = gr.Slider(-10.0, 10.0, value=lambda a='expression_lp_cal_rot': getattr(unleashed.globals, a), step=0.5, label="Full pipeline · cal rotation (°)", info='paste-back rotation correction', interactive=True)
                with gr.Accordion("Faceset", open=True):
                    faceset_mode = gr.Dropdown(["robust", "median", "mean"], label="Faceset average mode",
                                               info="How the photos of a faceset make one identity: robust = the mean "
                                                    "without the photos far from the others, mean = all photos, "
                                                    "median = the one photo closest to the others. Loaded sources follow "
                                                    "at the next preview / Start.",
                                               value=lambda: unleashed.globals.faceset_average_mode, interactive=True)
                    faceset_threshold = gr.Slider(0.0, 1.0, value=lambda: unleashed.globals.faceset_outlier_threshold, step=0.05,
                                                  label="Faceset outlier threshold",
                                                  info='robust: lower = drops more photos (at the lowest, only the closest photo)',
                                                  interactive=True)
            # ---------------- COLUMN 3: Interface & maintenance ----------------
            with gr.Column():
                with gr.Accordion("Interface & maintenance", open=True):
                    settings_controls.append(gr.Checkbox(label="Public Server", info="gradio.live link, from the next start",
                                                         value=lambda: unleashed.globals.CFG.server_share, elem_id='server_share', interactive=True))
                    settings_controls.append(gr.Checkbox(label='Delete earlier results at Start',
                                                         info='Only the files earlier renders wrote to the output folder '
                                                              '(never your targets or photos)',
                                                         value=lambda: unleashed.globals.CFG.clear_output, elem_id='clear_output', interactive=True))
                    input_server_name = gr.Textbox(label="Server Name", lines=1, info="Blank = local only. From the next start.",
                                                   value=lambda: unleashed.globals.CFG.server_name)
                    input_server_port = gr.Number(label="Server Port", precision=0, info="0 = default. From the next start.",
                                                  value=lambda: unleashed.globals.CFG.server_port)
                    button_apply_restart = gr.Button("Restart Server")
                    button_clean_temp = gr.Button("Clean temp folder")
                    gr.Markdown("Empties the app's temp folder (uploads, video copies, thumbnails). "
                                "It is also emptied at every start. Uploaded files have to be added again.")

    # Expression Restorer debug controls -> unleashed.globals (this session only)
    for c in expr_global_controls:
        c.select(fn=on_option_changed)
    expr_cal_scale.release(fn=lambda v, n='expression_lp_cal_scale': on_global_value_changed(v, n), inputs=[expr_cal_scale])
    expr_cal_dx.release(fn=lambda v, n='expression_lp_cal_dx': on_global_value_changed(v, n), inputs=[expr_cal_dx])
    expr_cal_dy.release(fn=lambda v, n='expression_lp_cal_dy': on_global_value_changed(v, n), inputs=[expr_cal_dy])
    expr_cal_rot.release(fn=lambda v, n='expression_lp_cal_rot': on_global_value_changed(v, n), inputs=[expr_cal_rot])

    # Faceset (saved)
    faceset_mode.select(fn=on_faceset_mode, inputs=[faceset_mode])
    # always_last: a slider sends a value per step, and with the default "once"
    # the page dropped the steps made while one was on its way (the last value
    # shown was not the one saved)
    faceset_threshold.input(fn=on_faceset_threshold, inputs=[faceset_threshold], trigger_mode='always_last')

    # text boxes: applied on Enter or when leaving the box (not at every keystroke:
    # the output folder box made a folder for each partial path)
    for ev in (output_template.submit, output_template.blur):
        ev(fn=on_output_template, inputs=[output_template], outputs=[output_template], show_api=False)
    for ev in (output_folder_box.submit, output_folder_box.blur):
        ev(fn=on_output_folder, inputs=[output_folder_box], outputs=[output_folder_box], show_api=False)
    for ev in (path_start_box.submit, path_start_box.blur):
        ev(fn=on_path_start, inputs=[path_start_box], outputs=[path_start_box], show_api=False)
    for ev in (input_server_name.submit, input_server_name.blur):
        ev(fn=on_server_name, inputs=[input_server_name], outputs=[input_server_name], show_api=False)
    # (a Number has no blur event: typing is stored, Enter also tidies the box)
    input_server_port.input(fn=on_server_port_typed, inputs=[input_server_port], show_api=False,
                            trigger_mode='always_last')
    input_server_port.submit(fn=on_server_port, inputs=[input_server_port], outputs=[input_server_port], show_api=False)

    # dropdowns, checkboxes and sliders of the config
    for s in settings_controls:
        s.select(fn=on_settings_changed, trigger_mode='always_last')
    max_threads.input(fn=lambda a, b='max_threads': on_settings_changed_misc(a, b), inputs=[max_threads],
                      trigger_mode='always_last')
    memory_limit.input(fn=lambda a, b='memory_limit': on_settings_changed_misc(a, b), inputs=[memory_limit],
                       trigger_mode='always_last')
    video_quality.input(fn=lambda a, b='video_quality': on_settings_changed_misc(a, b), inputs=[video_quality],
                        trigger_mode='always_last')

    from ui.tabs import faceswap_tab as fs
    # in the Face Swap tab's group: it changes that tab's target list
    button_clean_temp.click(fn=clean_temp, outputs=fs.refresh_outputs(), show_api=False,
                            concurrency_id='fs_state', concurrency_limit=1)
    button_apply_restart.click(restart)


# ----------------------------------------------------------------------------- handlers

def on_global_value_changed(new_val, attribname):
    if hasattr(unleashed.globals, attribname):
        setattr(unleashed.globals, attribname, new_val)


def on_option_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(unleashed.globals, attribname):
            setattr(unleashed.globals, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(unleashed.globals, attribname):
            setattr(unleashed.globals, attribname, evt.value)
            return
    raise gr.Error(f'Unhandled Setting for {evt.target}')


def on_faceset_mode(mode):
    if mode in ('robust', 'median', 'mean'):
        unleashed.globals.faceset_average_mode = mode
        unleashed.globals.CFG.faceset_average_mode = mode
        _persist()


def on_faceset_threshold(value):
    try:
        value = min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return
    unleashed.globals.faceset_outlier_threshold = value
    unleashed.globals.CFG.faceset_outlier_threshold = value
    _persist()


def on_output_template(text):
    """A file name with placeholders; refused (the old one stays) when empty,
    with an unknown placeholder (it used to fail the render after the video
    was done) or with a folder separator."""
    current = unleashed.globals.CFG.output_template
    text = (text or '').strip()
    if text == current:
        return current
    if not text:
        gr.Warning('The template cannot be empty (default: {file}_{time})')
        return current
    unknown = template_parser.unknown_placeholders(text)
    if unknown:
        gr.Warning(f'Unknown placeholder {{{unknown[0]}}}: use {template_parser.PLACEHOLDERS}')
        return current
    if any(c in text for c in '/\\'):
        gr.Warning('The template is a file name: no / or \\ (the folder is the Output Folder)')
        return current
    unleashed.globals.CFG.output_template = text
    _persist()
    return text


def on_output_folder(folder):
    """Where results go. Applied once (Enter / leaving the box), made when
    applied; the folder in use stays when the new one cannot be used."""
    from unleashed import core
    current = unleashed.globals.output_path or unleashed.globals.CFG.output_folder
    folder = os.path.expanduser((folder or '').strip())
    if not folder or os.path.normpath(folder) == os.path.normpath(current or ''):
        return current
    if core.render_active or unleashed.globals.processing:
        gr.Warning('A render is running: change the output folder when it is done')
        return current
    if not os.path.isabs(folder):
        gr.Warning('Type the full path of the folder (e.g. /content/drive/MyDrive/results)')
        return current
    if os.path.dirname(os.path.normpath(folder)) == os.path.normpath(folder):
        gr.Warning('Not the root of the disk: choose a folder for the results')
        return current
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as e:
        gr.Warning(f'This folder cannot be used: {e}')
        return current
    if not os.access(folder, os.W_OK):
        gr.Warning(f'This folder cannot be written to: {folder}')
        return current
    unleashed.globals.output_path = folder
    unleashed.globals.CFG.output_folder = folder
    _persist()
    gr.Info(f'Results now go to {folder}')
    return folder


def on_path_start(folder):
    folder = (folder or '').strip()
    unleashed.globals.CFG.path_start = folder
    _persist()
    return folder


def on_server_name(name):
    name = (name or '').strip()
    unleashed.globals.CFG.server_name = name
    _persist()
    return name


def on_server_port_typed(value):
    """While typing: stored, the box left as it is (no output)."""
    on_server_port(value, quiet=True)


def on_server_port(value, quiet=False):
    try:
        port = int(value or 0)
    except (TypeError, ValueError):
        port = 0
    if not 0 <= port < 65536:
        if not quiet:
            gr.Warning('A port is a number from 1 to 65535 (0 = default)')
        port = 0
    unleashed.globals.CFG.server_port = port
    _persist()
    return port


def on_settings_changed_misc(new_val, attribname):
    if hasattr(unleashed.globals.CFG, attribname):
        if attribname in ('max_threads', 'memory_limit', 'video_quality'):
            try:
                new_val = int(round(float(new_val)))
            except (TypeError, ValueError):
                return
        setattr(unleashed.globals.CFG, attribname, new_val)
        _persist()
    else:
        print("Didn't find attrib!")


def on_settings_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    CFG = unleashed.globals.CFG
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(CFG, attribname):
            setattr(CFG, attribname, evt.selected)
            if attribname == 'server_share':
                CFG.server_share_saved = evt.selected
            _persist()
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(CFG, attribname):
            setattr(CFG, attribname, evt.value)
            if attribname == 'provider':
                CFG.provider_configured = evt.value
            _persist()
            return
    raise gr.Error(f'Unhandled Setting for {evt.target}')


def clean_temp():
    """Empty ./temp. Face Swap targets uploaded into it leave that tab's list
    (sources and targets added by path stay) and the tab is refreshed (the
    returned values fill faceswap_tab.refresh_outputs())."""
    from ui.main import prepare_environment
    from unleashed.utilities import clean_temp_folder
    from ui.tabs import faceswap_tab as fs, faceswap_state

    if fs._rendering() or unleashed.globals.processing:
        gr.Warning('A render is running: its files are in the temp folder. Clean it when the render is done.')
        return fs.refresh_values()
    if unleashed.globals.CFG.use_os_temp_folder:
        gr.Warning('"use_os_temp_folder" is on in config.yaml: the system temp folder is not cleaned by the app.')
        return fs.refresh_values()
    freed = clean_temp_folder()
    prepare_environment()
    gone = faceswap_state.reset_after_temp_clean()
    text = f'Temp folder emptied: {freed / 1e6:.0f} MB freed.'
    if gone:
        text += (f' {len(gone)} uploaded target file{"s were" if len(gone) != 1 else " was"} in it and left the '
                 f'Face Swap list ({", ".join(gone[:3])}{" ..." if len(gone) > 3 else ""}): upload again to use.')
    gr.Info(text)
    return fs.refresh_values()


def restart():
    ui.globals.ui_restart_server = True
