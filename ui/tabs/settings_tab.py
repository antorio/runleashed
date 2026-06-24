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
    with gr.Tab("Settings"):
        gr.Markdown("### Settings\nGlobal preferences — applied to every swap and saved automatically.")
        accuracy_controls = []      # checkboxes/dropdowns -> roop.globals via elem_id
        accuracy_sliders = []       # (slider, globals_attr) -> on_global_value_changed
        with gr.Row():
            # ---------------- COLUMN 1: Performance + Output ----------------
            with gr.Column():
                with gr.Accordion("Performance", open=True):
                    settings_controls.append(gr.Dropdown(providerlist, label="Provider", value=roop.globals.CFG.provider, elem_id='provider', interactive=True))
                    max_threads = gr.Slider(1, 32, value=roop.globals.CFG.max_threads, label="Max. Number of Threads", info='default: 8', step=1.0, interactive=True)
                    memory_limit = gr.Slider(0, 128, value=roop.globals.CFG.memory_limit, label="Max. Memory to use (Gb)", info='0 meaning no limit', step=1.0, interactive=True)
                    settings_controls.append(gr.Checkbox(label="Force CPU for Face Analyser", value=roop.globals.CFG.force_cpu, elem_id='force_cpu', interactive=True))
                    chk_det_size = gr.Checkbox(label="Use default Det-Size", value=True, elem_id='default_det_size', interactive=True)
                with gr.Accordion("Output", open=True):
                    output_template = gr.Textbox(label="Filename Output Template", info="(file extension is added automatically)", lines=1, placeholder='{file}_{time}', value=roop.globals.CFG.output_template)
                    output_folder_box = gr.Textbox(label="Output Folder", info="Where results are saved", lines=1, placeholder='/content/drive/MyDrive/c', value=getattr(roop.globals.CFG, 'output_folder', '/content/drive/MyDrive/c'))
                    settings_controls.append(gr.Dropdown(image_formats, label="Image Output Format", info='default: png', value=roop.globals.CFG.output_image_format, elem_id='output_image_format', interactive=True))
                    settings_controls.append(gr.Dropdown(video_codecs, label="Video Codec", info='default: libx264', value=roop.globals.CFG.output_video_codec, elem_id='output_video_codec', interactive=True))
                    settings_controls.append(gr.Dropdown(video_formats, label="Video Output Format", info='default: mp4', value=roop.globals.CFG.output_video_format, elem_id='output_video_format', interactive=True))
                    video_quality = gr.Slider(0, 100, value=roop.globals.CFG.video_quality, label="Video Quality (crf)", info='default: 18', step=1.0, interactive=True)
                with gr.Accordion("Alignment & detection — live tuning", open=False):
                    accuracy_controls.append(gr.Checkbox(label="Use landmark alignment (68pt + RANSAC)", info="Main fix for extreme yaw/pitch. Off = detector raw kps.", value=roop.globals.use_landmark_alignment, elem_id='use_landmark_alignment', interactive=True))
                    accuracy_controls.append(gr.Dropdown(["off", "fallback", "always"], label="Multi-angle detection", info="fallback = only rotate when 0° finds nothing", value=roop.globals.multi_angle_detection_mode, elem_id='multi_angle_detection_mode', interactive=True))
                    accuracy_controls.append(gr.Checkbox(label="Color transfer (LAB) toward target", info="Match swap colour to target lighting", value=roop.globals.use_color_transfer, elem_id='use_color_transfer', interactive=True))
            # ---------------- COLUMN 2: Expression Restorer ----------------
            with gr.Column():
                with gr.Accordion("Expression Restorer — live tuning (no restart needed)", open=True):
                    expr_global_controls = []
                    expr_global_controls.append(gr.Checkbox(label="Pose lock (translation)", value=roop.globals.expression_pose_lock, elem_id='expression_pose_lock', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Pose lock — scale", value=roop.globals.expression_pose_lock_scale, elem_id='expression_pose_lock_scale', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Pose lock — rotation", value=roop.globals.expression_pose_lock_rotation, elem_id='expression_pose_lock_rotation', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Full LivePortrait pipeline (experimental)", value=roop.globals.expression_full_pipeline, elem_id='expression_full_pipeline', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Stitching model (experimental)", value=roop.globals.expression_stitching, elem_id='expression_stitching', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Serialize (stable at high threads)", value=roop.globals.expression_serialize, elem_id='expression_serialize', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Debug log [expr-delta] (A/B in console)", value=roop.globals.expression_debug, elem_id='expression_debug', interactive=True))
                    expr_global_controls.append(gr.Checkbox(label="Pose gate (skip restorer at extreme angles)", value=roop.globals.expression_pose_gate, elem_id='expression_pose_gate', interactive=True))
                    expr_power = gr.Slider(0.0, 5.0, value=roop.globals.expression_power, step=0.1, label="Expression power", info='amplify expression (default 2.0)', interactive=True)
                    expr_border = gr.Slider(0.0, 0.5, value=roop.globals.expression_blend_border, step=0.02, label="Blend border", info='edge feather (default 0.2)', interactive=True)
                    expr_gate_soft = gr.Slider(10.0, 90.0, value=roop.globals.expression_pose_gate_soft, step=1.0, label="Pose gate · start fade (°)", info='max(|pitch|,|yaw|) where restorer begins fading', interactive=True)
                    expr_gate_hard = gr.Slider(10.0, 90.0, value=roop.globals.expression_pose_gate_hard, step=1.0, label="Pose gate · full skip (°)", info='angle where restorer is skipped (clean swap)', interactive=True)
                    expr_cal_scale = gr.Slider(0.80, 1.20, value=roop.globals.expression_lp_cal_scale, step=0.005, label="Full pipeline · cal scale", info='paste-back size correction (1.0 = none)', interactive=True)
                    expr_cal_dx = gr.Slider(-0.15, 0.15, value=roop.globals.expression_lp_cal_dx, step=0.005, label="Full pipeline · cal shift X", info='horizontal nudge (fraction of crop)', interactive=True)
                    expr_cal_dy = gr.Slider(-0.15, 0.15, value=roop.globals.expression_lp_cal_dy, step=0.005, label="Full pipeline · cal shift Y", info='vertical nudge (fraction of crop)', interactive=True)
                    expr_cal_rot = gr.Slider(-10.0, 10.0, value=roop.globals.expression_lp_cal_rot, step=0.5, label="Full pipeline · cal rotation (°)", info='paste-back rotation correction', interactive=True)
                with gr.Accordion("Face mask & paste-back — live tuning", open=False):
                    accuracy_controls.append(gr.Checkbox(label="Convex-hull face matte", info="Follows face contour (less jaw/neck/bg bleed). Off = rectangle.", value=roop.globals.use_face_hull_mask, elem_id='use_face_hull_mask', interactive=True))
                    accuracy_controls.append(gr.Checkbox(label="Occlusion mask after enhancer", info="Mask always runs after the ER. On = also after the enhancer, so occluders (hands/hair) stay un-enhanced.", value=roop.globals.mask_after_enhancer, elem_id='mask_after_enhancer', interactive=True))
                    _hf = gr.Slider(0.0, 1.5, value=roop.globals.face_hull_forehead, step=0.05, label="Hull forehead extend", info='cover forehead (default 0.6)', interactive=True)
                    _hd = gr.Slider(0.0, 0.5, value=roop.globals.face_hull_dilate, step=0.01, label="Hull dilate", info='grow matte outward (default 0.10)', interactive=True)
                    _hm = gr.Slider(0.0, 0.5, value=roop.globals.face_hull_min_area, step=0.01, label="Hull min area (degeneracy guard)", info='fallback to ellipse below this at extreme angles (default 0.22)', interactive=True)
                    accuracy_sliders.append((_hf, 'face_hull_forehead'))
                    accuracy_sliders.append((_hd, 'face_hull_dilate'))
                    accuracy_sliders.append((_hm, 'face_hull_min_area'))
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
                with gr.Accordion("Stabilization & faceset — live tuning", open=False):
                    accuracy_controls.append(gr.Checkbox(label="Landmark smoothing (video)", info="Reduce per-frame jitter in video", value=roop.globals.landmark_smoothing, elem_id='landmark_smoothing', interactive=True))
                    _ls = gr.Slider(0.0, 1.0, value=roop.globals.landmark_smoothing_strength, step=0.05, label="Smoothing strength", info='higher = smoother (default 0.7)', interactive=True)
                    accuracy_sliders.append((_ls, 'landmark_smoothing_strength'))
                    accuracy_controls.append(gr.Dropdown(["robust", "median", "mean"], label="Faceset average mode", info="how multi-image source identity is blended", value=roop.globals.faceset_average_mode, elem_id='faceset_average_mode', interactive=True))
                    _ot = gr.Slider(0.0, 1.0, value=roop.globals.faceset_outlier_threshold, step=0.05, label="Faceset outlier threshold", info='lower = stricter outlier drop (default 0.6)', interactive=True)
                    accuracy_sliders.append((_ot, 'faceset_outlier_threshold'))

    for c in expr_global_controls:
        c.select(fn=on_option_changed)
    expr_power.release(fn=lambda v, n='expression_power': on_global_value_changed(v, n), inputs=[expr_power])
    expr_border.release(fn=lambda v, n='expression_blend_border': on_global_value_changed(v, n), inputs=[expr_border])
    expr_gate_soft.release(fn=lambda v, n='expression_pose_gate_soft': on_global_value_changed(v, n), inputs=[expr_gate_soft])
    expr_gate_hard.release(fn=lambda v, n='expression_pose_gate_hard': on_global_value_changed(v, n), inputs=[expr_gate_hard])
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
    button_apply_settings.click(apply_settings, inputs=[themes, input_server_name, input_server_port, output_template])
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
