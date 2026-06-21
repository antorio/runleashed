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

    providerlist = suggest_execution_providers()
    with gr.Tab("⚙ Settings"):
        gr.Markdown("### Settings\nGlobal preferences — applied to every swap and saved automatically.")
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Performance", open=True):
                    settings_controls.append(gr.Dropdown(providerlist, label="Provider", value=roop.globals.CFG.provider, elem_id='provider', interactive=True))
                    max_threads = gr.Slider(1, 32, value=roop.globals.CFG.max_threads, label="Max. Number of Threads", info='default: 3', step=1.0, interactive=True)
                    memory_limit = gr.Slider(0, 128, value=roop.globals.CFG.memory_limit, label="Max. Memory to use (Gb)", info='0 meaning no limit', step=1.0, interactive=True)
                    settings_controls.append(gr.Checkbox(label="Force CPU for Face Analyser", value=roop.globals.CFG.force_cpu, elem_id='force_cpu', interactive=True))
                    chk_det_size = gr.Checkbox(label="Use default Det-Size", value=True, elem_id='default_det_size', interactive=True)
            with gr.Column():
                with gr.Accordion("Output", open=True):
                    output_template = gr.Textbox(label="Filename Output Template", info="(file extension is added automatically)", lines=1, placeholder='{file}_{time}', value=roop.globals.CFG.output_template)
                    settings_controls.append(gr.Dropdown(image_formats, label="Image Output Format", info='default: png', value=roop.globals.CFG.output_image_format, elem_id='output_image_format', interactive=True))
                    settings_controls.append(gr.Dropdown(video_codecs, label="Video Codec", info='default: libx264', value=roop.globals.CFG.output_video_codec, elem_id='output_video_codec', interactive=True))
                    settings_controls.append(gr.Dropdown(video_formats, label="Video Output Format", info='default: mp4', value=roop.globals.CFG.output_video_format, elem_id='output_video_format', interactive=True))
                    video_quality = gr.Slider(0, 100, value=roop.globals.CFG.video_quality, label="Video Quality (crf)", info='default: 14', step=1.0, interactive=True)
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
        with gr.Row():
            with gr.Accordion("Expression Restorer — live tuning (no restart needed)", open=False):
                expr_global_controls = []
                with gr.Row():
                    expr_global_controls.append(gr.Checkbox(label="Pose lock (translation)", value=roop.globals.expression_pose_lock, elem_id='expression_pose_lock', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Pose lock — scale", value=roop.globals.expression_pose_lock_scale, elem_id='expression_pose_lock_scale', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Pose lock — rotation", value=roop.globals.expression_pose_lock_rotation, elem_id='expression_pose_lock_rotation', interactive=True))
                with gr.Row():
                    expr_global_controls.append(gr.Checkbox(label="Full LivePortrait pipeline (experimental)", value=roop.globals.expression_full_pipeline, elem_id='expression_full_pipeline', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Stitching model (experimental)", value=roop.globals.expression_stitching, elem_id='expression_stitching', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Serialize (stable at high threads)", value=roop.globals.expression_serialize, elem_id='expression_serialize', interactive=True))
                with gr.Row():
                    expr_power = gr.Slider(0.0, 5.0, value=roop.globals.expression_power, step=0.1, label="Expression power", info='amplify expression (default 2.0)', interactive=True)
                    expr_border = gr.Slider(0.0, 0.5, value=roop.globals.expression_blend_border, step=0.02, label="Blend border", info='edge feather (default 0.2)', interactive=True)

    for c in expr_global_controls:
        c.select(fn=on_option_changed)
    expr_power.release(fn=lambda v, n='expression_power': on_global_value_changed(v, n), inputs=[expr_power])
    expr_border.release(fn=lambda v, n='expression_blend_border': on_global_value_changed(v, n), inputs=[expr_border])

    chk_det_size.select(fn=on_option_changed)

    # Settings
    for s in settings_controls:
        s.select(fn=on_settings_changed)
    max_threads.input(fn=lambda a,b='max_threads':on_settings_changed_misc(a,b), inputs=[max_threads])
    memory_limit.input(fn=lambda a,b='memory_limit':on_settings_changed_misc(a,b), inputs=[memory_limit])
    video_quality.input(fn=lambda a,b='video_quality':on_settings_changed_misc(a,b), inputs=[video_quality])

    button_clean_temp.click(fn=clean_temp)
    button_apply_settings.click(apply_settings, inputs=[themes, input_server_name, input_server_port, output_template])
    button_apply_restart.click(restart)


def on_global_value_changed(new_val, attribname):
    if hasattr(roop.globals, attribname):
        setattr(roop.globals, attribname, new_val)


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


def apply_settings(themes, input_server_name, input_server_port, output_template):
    from ui.main import show_msg

    roop.globals.CFG.selected_theme = themes
    roop.globals.CFG.server_name = input_server_name
    roop.globals.CFG.server_port = input_server_port
    roop.globals.CFG.output_template = output_template
    roop.globals.CFG.save()
    show_msg('Settings saved')


def restart():
    ui.globals.ui_restart_server = True
