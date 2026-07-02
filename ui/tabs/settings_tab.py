import shutil
import os
import gradio as gr
import roop.globals
import ui.globals
from roop.utilities import clean_dir

available_themes = ["Default", "gradio/glass", "gradio/monochrome", "gradio/seafoam", "gradio/soft", "gstaff/xkcd", "freddyaboulton/dracula_revamped", "ysharma/steampunk"]
image_formats = ['jpg','png', 'webp']
video_formats = ['avi','mkv', 'mp4', 'webm']
video_codecs = ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']
providerlist = None

settings_controls = []

def settings_tab():
    from roop.core import suggest_execution_providers
    global providerlist
    settings_controls.clear()   # avoid stale-control accumulation across server restarts

    providerlist = suggest_execution_providers()
    with gr.Tab("Settings"):
        gr.Markdown("### Settings\nGlobal preferences — applied to every swap and saved automatically.")
        accuracy_controls = []      # checkboxes/dropdowns -> roop.globals via elem_id
        accuracy_sliders = []       # (slider, globals_attr) -> on_global_value_changed
        with gr.Row():
            # ---------------- COLUMN 1: Performance + Output ----------------
            with gr.Column():
                with gr.Accordion("Performance", open=True):
                    settings_controls.append(gr.Dropdown(providerlist, label="Provider", value=roop.globals.CFG.provider, elem_id='provider', interactive=True))
                    max_threads = gr.Slider(1, 32, value=roop.globals.CFG.max_threads, label="Max. Number of Threads", step=1.0, interactive=True)
                    memory_limit = gr.Slider(0, 128, value=roop.globals.CFG.memory_limit, label="Max. Memory to use (Gb)", info='0 meaning no limit', step=1.0, interactive=True)
                    settings_controls.append(gr.Checkbox(label="Force CPU for Face Analyser", value=roop.globals.CFG.force_cpu, elem_id='force_cpu', interactive=True))
                    chk_det_size = gr.Checkbox(label="Use default Det-Size", value=True, elem_id='default_det_size', interactive=True)
                with gr.Accordion("Output", open=True):
                    output_template = gr.Textbox(label="Filename Output Template", info="(file extension is added automatically)", lines=1, placeholder='{file}_{time}', value=roop.globals.CFG.output_template)
                    output_folder_box = gr.Textbox(label="Output Folder", info="Where results are saved", lines=1, placeholder='/content/drive/MyDrive/c', value=getattr(roop.globals.CFG, 'output_folder', '/content/drive/MyDrive/c'))
                    settings_controls.append(gr.Dropdown(image_formats, label="Image Output Format", value=roop.globals.CFG.output_image_format, elem_id='output_image_format', interactive=True))
                    settings_controls.append(gr.Dropdown(video_codecs, label="Video Codec", value=roop.globals.CFG.output_video_codec, elem_id='output_video_codec', interactive=True))
                    settings_controls.append(gr.Dropdown(video_formats, label="Video Output Format", value=roop.globals.CFG.output_video_format, elem_id='output_video_format', interactive=True))
                    video_quality = gr.Slider(0, 100, value=roop.globals.CFG.video_quality, label="Video Quality (crf)", step=1.0, interactive=True)
            # ---------------- COLUMN 2: Expression Restorer ----------------
            with gr.Column():
                with gr.Accordion("Expression Restorer — experimental / debug", open=True):
                    expr_global_controls = []
                    expr_global_controls.append(gr.Checkbox(label="Full LivePortrait pipeline (experimental)", value=lambda a='expression_full_pipeline': getattr(roop.globals, a), elem_id='expression_full_pipeline', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Stitching model (experimental)", value=lambda a='expression_stitching': getattr(roop.globals, a), elem_id='expression_stitching', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Serialize (stable at high threads)", value=lambda a='expression_serialize': getattr(roop.globals, a), elem_id='expression_serialize', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Debug log [expr-delta] (A/B in console)", value=lambda a='expression_debug': getattr(roop.globals, a), elem_id='expression_debug', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Profile timings (per-stage ms in console)", info="Prints [timing] per processor each frame -- use briefly to see which stage is slow, then turn off.", value=lambda a='profile_timings': getattr(roop.globals, a), elem_id='profile_timings', interactive=True))
                    expr_cal_scale = gr.Slider(0.80, 1.20, value=lambda a='expression_lp_cal_scale': getattr(roop.globals, a), step=0.005, label="Full pipeline · cal scale", info='paste-back size correction (1.0 = none)', interactive=True)
                    expr_cal_dx = gr.Slider(-0.15, 0.15, value=lambda a='expression_lp_cal_dx': getattr(roop.globals, a), step=0.005, label="Full pipeline · cal shift X", info='horizontal nudge (fraction of crop)', interactive=True)
                    expr_cal_dy = gr.Slider(-0.15, 0.15, value=lambda a='expression_lp_cal_dy': getattr(roop.globals, a), step=0.005, label="Full pipeline · cal shift Y", info='vertical nudge (fraction of crop)', interactive=True)
                    expr_cal_rot = gr.Slider(-10.0, 10.0, value=lambda a='expression_lp_cal_rot': getattr(roop.globals, a), step=0.5, label="Full pipeline · cal rotation (°)", info='paste-back rotation correction', interactive=True)
                with gr.Accordion("Faceset", open=True):
                    accuracy_controls.append(gr.Dropdown(["robust", "median", "mean"], label="Faceset average mode", info="how multi-image source identity is blended", value=lambda a='faceset_average_mode': getattr(roop.globals, a), elem_id='faceset_average_mode', interactive=True))
                    _ot = gr.Slider(0.0, 1.0, value=lambda a='faceset_outlier_threshold': getattr(roop.globals, a), step=0.05, label="Faceset outlier threshold", info='lower = stricter outlier drop', interactive=True)
                    accuracy_sliders.append((_ot, 'faceset_outlier_threshold'))
            # ---------------- COLUMN 3: Interface & maintenance ----------------
            with gr.Column():
                with gr.Accordion("Interface & maintenance", open=True):
                    themes = gr.Dropdown(available_themes, label="Theme", info="Change needs complete restart", value=roop.globals.CFG.selected_theme)
                    settings_controls.append(gr.Checkbox(label="Public Server", value=roop.globals.CFG.server_share, elem_id='server_share', interactive=True))
                    settings_controls.append(gr.Checkbox(label='Clear output folder before each run', value=roop.globals.CFG.clear_output, elem_id='clear_output', interactive=True))
                    settings_controls.append(gr.Checkbox(label='Use OS temp folder', value=roop.globals.CFG.use_os_temp_folder, elem_id='use_os_temp_folder', interactive=True))
                    settings_controls.append(gr.Checkbox(label='Show video in browser (re-encodes output)', value=roop.globals.CFG.output_show_video, elem_id='output_show_video', interactive=True))
                    input_server_name = gr.Textbox(label="Server Name", lines=1, info="Leave blank to run locally", value=roop.globals.CFG.server_name)
                    input_server_port = gr.Number(label="Server Port", precision=0, info="Leave at 0 to use default", value=roop.globals.CFG.server_port)
                    with gr.Row():
                        button_apply_settings = gr.Button("Apply Settings", variant='primary')
                        button_apply_restart = gr.Button("Restart Server")
                    button_clean_temp = gr.Button("Clean temp folder")

    for c in expr_global_controls:
        c.select(fn=on_option_changed)
    expr_cal_scale.release(fn=lambda v, n='expression_lp_cal_scale': on_global_value_changed(v, n), inputs=[expr_cal_scale])
    expr_cal_dx.release(fn=lambda v, n='expression_lp_cal_dx': on_global_value_changed(v, n), inputs=[expr_cal_dx])
    expr_cal_dy.release(fn=lambda v, n='expression_lp_cal_dy': on_global_value_changed(v, n), inputs=[expr_cal_dy])
    expr_cal_rot.release(fn=lambda v, n='expression_lp_cal_rot': on_global_value_changed(v, n), inputs=[expr_cal_rot])

    chk_det_size.select(fn=on_option_changed)

    # New accuracy / quality controls -> roop.globals (live, reset to globals.py on restart)
    for c in accuracy_controls:
        c.select(fn=on_option_changed)
    for _sl, _nm in accuracy_sliders:
        _sl.release(fn=lambda v, n=_nm: on_global_value_changed(v, n), inputs=[_sl])
    output_folder_box.change(fn=on_output_folder_changed, inputs=[output_folder_box])

    # Settings
    for s in settings_controls:
        s.select(fn=on_settings_changed)
    max_threads.input(fn=lambda a,b='max_threads':on_settings_changed_misc(a,b), inputs=[max_threads])
    memory_limit.input(fn=lambda a,b='memory_limit':on_settings_changed_misc(a,b), inputs=[memory_limit])
    video_quality.input(fn=lambda a,b='video_quality':on_settings_changed_misc(a,b), inputs=[video_quality])

    button_clean_temp.click(fn=clean_temp)
    button_apply_settings.click(apply_settings, inputs=[themes, input_server_name, input_server_port, output_template, max_threads, memory_limit, video_quality])
    button_apply_restart.click(restart)


def on_global_value_changed(new_val, attribname):
    if hasattr(roop.globals, attribname):
        setattr(roop.globals, attribname, new_val)


def on_output_folder_changed(folder):
    folder = (folder or '').strip()
    if not folder:
        return
    roop.globals.output_path = folder
    if roop.globals.CFG is not None:
        roop.globals.CFG.output_folder = folder
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception:
        pass


def on_option_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.value)
            return
    raise gr.Error(f'Unhandled Setting for {evt.target}')


def on_settings_changed_misc(new_val, attribname):
    if hasattr(roop.globals.CFG, attribname):
        if attribname in ('max_threads', 'memory_limit', 'video_quality'):
            try:
                new_val = int(round(float(new_val)))
            except (TypeError, ValueError):
                pass
        setattr(roop.globals.CFG, attribname, new_val)
    else:
        print("Didn't find attrib!")
        


def on_settings_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.value)
            return
            
    raise gr.Error(f'Unhandled Setting for {evt.target}')

def clean_temp():
    from ui.main import prepare_environment
    
    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs = []
    if not roop.globals.CFG.use_os_temp_folder:
        clean_dir(os.environ["TEMP"])
    prepare_environment()
    gr.Info('Temp Files removed')
    return None,None,None,None


def apply_settings(themes, input_server_name, input_server_port, output_template, max_threads, memory_limit, video_quality):
    from ui.main import show_msg

    roop.globals.CFG.selected_theme = themes
    roop.globals.CFG.server_name = input_server_name
    roop.globals.CFG.server_port = input_server_port
    roop.globals.CFG.output_template = output_template
    # Make the button authoritative for the sliders too: read what the user sees
    # right now, so threads/memory/quality no longer depend on the live .input event.
    roop.globals.CFG.max_threads = int(round(float(max_threads)))
    roop.globals.CFG.memory_limit = int(round(float(memory_limit)))
    roop.globals.CFG.video_quality = int(round(float(video_quality)))
    roop.globals.CFG.save()
    show_msg('Settings saved')


def restart():
    ui.globals.ui_restart_server = True
