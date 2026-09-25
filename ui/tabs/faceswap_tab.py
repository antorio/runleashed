"""Face Swap tab: layout and wiring. The state and logic live in
ui/tabs/faceswap_state.py (S).

Layout, left to right in the order you work (compact for a 1920x1080 screen):
  1 Source faces · 2 Target files · 3 Faces to replace  |  preview on top,
  then frame (◀ ▶, arrow keys) · range · view · Ready + Start / Stop · status ·
  results  |  settings in accordions (Swap, Expression, Occlusion, Enhance,
  Detection & tracking, Video output).

How the preview follows the settings: every setting's .change bumps a hidden
counter in the browser (js only, no server call); the counter's .change runs
one preview with the current panel values (always_last, one at a time), so a
burst of changes gives one or two previews. Server-side changes (sources,
targets, people, painting) return the counter + 1 to get the same effect.
Settings are applied at each preview and once at Start, never during a render.
"""
import os
import time
import traceback

import gradio as gr
import numpy as np

import roop.globals
import roop.utilities as util
from roop.capturer import get_image_frame, get_video_frame
from ui.tabs import faceswap_state as S

G = roop.globals
INTERNAL = dict(show_api=False)


def _info(message):
    """Short-lived confirmation (the toasts sit over the settings column's top)."""
    gr.Info(message, duration=5)

BUMP_JS = '(n) => (n || 0) + 1'
# ◀ ▶ in the browser, moving the slider itself: Gradio reads a js event's
# inputs asynchronously, so a click that came before the previous one had been
# applied read the same frame and was lost (a server round trip was worse)
STEP_JS = '''(n) => {
    const el = document.querySelector('#frame_slider input[type=range]');
    if (el) {
        const v = Math.max(Number(el.min) || 1, Math.min(Number(el.max), Math.round(Number(el.value)) + %d));
        if (v !== Number(el.value)) {
            el.value = v;
            el.dispatchEvent(new Event('input', {bubbles: true}));
        }
    }
    return (n || 0) + 1;
}'''

settings = {}              # key -> component
C = {}                     # other components by name (for the load / cross-tab refresh)
_preview = {'size': None, 'view': None}      # what the preview image currently shows
last_results = []          # files of the last run
selected_result = None


def _s(key, comp):
    settings[key] = comp
    return comp


def _vals(data):
    """Panel values by key from an event's {component: value} input."""
    return {k: data[c] for k, c in settings.items() if c in data}


# The galleries' highlight is set by the server (the source in use, the file
# shown, the person picked). Gradio's Gallery then fires `select` as if it had
# been clicked, but only when its highlight actually changes; processed after a
# newer action, that echo would move the choice back. So the server follows
# what each gallery highlights in the browser (Gallery.svelte: a new value
# keeps the highlight while it is in range and drops it after the gallery was
# empty) and notes an echo only for a highlight that really changes. A real
# click on another thumbnail clears the notes; notes also expire.
_shown = {'src': None, 'tgt': None, 'people': None}
_emptied = {'src': True, 'tgt': True, 'people': True}
_echo = {'src': [], 'tgt': [], 'people': []}
ECHO_SECONDS = 30


def _gallery(kind, items):
    """gr.Gallery(value=items), following what the browser will highlight."""
    n = len(items)
    if n == 0:
        _shown[kind], _emptied[kind] = None, True
    elif _emptied[kind]:
        _shown[kind], _emptied[kind] = None, False
    elif _shown[kind] is not None and _shown[kind] >= n:
        _shown[kind] = None
    return gr.Gallery(value=items)


def _page_loaded():
    """A new page: its galleries start empty-handed (nothing highlighted) and
    no editor is open."""
    for kind in _shown:
        _shown[kind], _emptied[kind] = None, True
        _echo[kind] = []
    _painting.update(tid=None, frame=None)


def _highlight(kind, index):
    if index is not None and index != _shown[kind]:
        _echo[kind] = [e for e in _echo[kind] if time.time() - e[1] < ECHO_SECONDS][-3:] + [(index, time.time())]
    _shown[kind] = index
    return gr.Gallery(selected_index=index)


def _is_echo(kind, index):
    _shown[kind] = index                        # the browser highlights it now
    notes = [e for e in _echo[kind] if time.time() - e[1] < ECHO_SECONDS]
    hit = next((e for e in notes if e[0] == index), None)
    if hit is not None:
        notes.remove(hit)
        _echo[kind] = notes
        return True
    _echo[kind] = []
    return False


def src_highlight():
    return _highlight('src', S.active_source_index() if G.INPUT_FACESETS else None)


def tgt_highlight():
    return _highlight('tgt', S.selected_target_index())


def people_highlight():
    return _highlight('people', S.selected_person if G.TARGET_FACES else None)


# ============================================================================ layout

def faceswap_tab():
    S.load_saved_defaults()
    V = S.values
    video = (S.target() or {}).get('kind') in ('video', 'gif')
    t = S.target()

    with gr.Tab("Face Swap"):
        tick = gr.Number(value=0, visible=False)
        C['tick'] = tick
        with gr.Row(elem_id="swap_row", equal_height=False):

            # --------------------------------------------------------------- left: set up
            with gr.Column(scale=3, min_width=280, elem_id="fs_left"):
                gr.Markdown("### 1 · Source faces", elem_classes="fs-step")
                C['src_gal'] = gr.Gallery(value=S.sources_gallery(), show_label=False, columns=3, allow_preview=False, preview=False,
                                          interactive=False, object_fit="cover", height="150px", elem_id="src_gal")
                C['src_info'] = gr.Markdown(S.source_info(), visible=bool(S.source_info()), elem_classes="fs-line")
                have_src = bool(G.INPUT_FACESETS)
                with gr.Row(elem_classes="fs-buttons"):
                    C['btn_src_remove'] = gr.Button("Remove this source", size="sm", min_width=80, interactive=have_src)
                    C['btn_src_clear'] = gr.Button("Remove all", size="sm", min_width=80, interactive=have_src)
                    C['btn_src_undo'] = gr.Button(S.source_undo_label() or "Undo", size="sm", min_width=80,
                                                  interactive=S.source_undo_label() is not None)
                with gr.Row(elem_classes="fs-buttons"):
                    C['btn_src_combine'] = gr.Button(S.combine_label(), size="sm", visible=len(S.same_person_photos()) >= 2)
                    C['btn_src_shuffle'] = gr.Button("Shuffle order", size="sm",
                                                     visible=S.MODES[V['mode']] == 'all_input' and len(G.INPUT_FACESETS) > 1)
                C['src_drop'] = gr.Files(show_label=False, file_count="multiple", file_types=None, elem_id="src_drop")
                with gr.Row(equal_height=True, elem_classes="fs-path"):
                    C['src_path'] = gr.Textbox(show_label=False, container=False, scale=5, max_lines=1,
                                               placeholder="or a path: .fsz, photo or folder (Enter)")
                    C['btn_src_path'] = gr.Button("Add", size="sm", scale=1, min_width=60)

                gr.Markdown("### 2 · Target files", elem_classes="fs-step")
                C['tgt_gal'] = gr.Gallery(value=S.targets_gallery(), show_label=False, columns=3, allow_preview=False, preview=False,
                                          interactive=False, object_fit="contain", height="150px", elem_id="tgt_gal")
                have_tgt = bool(S.targets)
                with gr.Row(elem_classes="fs-buttons"):
                    C['btn_tgt_remove'] = gr.Button("Remove this file", size="sm", min_width=80, interactive=have_tgt)
                    C['btn_tgt_clear'] = gr.Button("Remove all", size="sm", min_width=80, interactive=have_tgt)
                    C['btn_tgt_undo'] = gr.Button(S.undo_label() or "Undo", size="sm", min_width=80, interactive=S.undo_label() is not None)
                C['tgt_drop'] = gr.Files(show_label=False, file_count="multiple", file_types=None, elem_id="tgt_drop")
                with gr.Row(equal_height=True, elem_classes="fs-path"):
                    C['tgt_path'] = gr.Textbox(show_label=False, container=False, scale=5, max_lines=1,
                                               placeholder="or a path: file or folder (Enter)")
                    C['btn_tgt_path'] = gr.Button("Add", size="sm", scale=1, min_width=60)

                gr.Markdown("### 3 · Faces to replace", elem_classes="fs-step")
                mode = _s('mode', gr.Dropdown(list(S.MODES), value=V['mode'], show_label=False, container=False,
                                              elem_id="mode_dd"))
                with gr.Column(visible=S.MODES[V['mode']] == 'selected', elem_id="people_col") as people_col:
                    C['people_col'] = people_col
                    C['people_hint'] = gr.Markdown("Click faces in the preview to add them.", visible=not G.TARGET_FACES,
                                                   elem_classes="fs-line")
                    C['people_gal'] = gr.Gallery(value=S.people_gallery(), show_label=False, columns=4, allow_preview=False,
                                                 preview=False, interactive=False, object_fit="cover", height="110px",
                                                 elem_id="people_gal")
                    with gr.Row(elem_id="people_tools"):
                        C['btn_person_remove'] = gr.Button("Remove this person", size="sm", scale=1, min_width=120,
                                                           interactive=bool(G.TARGET_FACES))
                        _s('tolerance', gr.Slider(0.01, 1.0, value=V['tolerance'], step=0.01, scale=2,
                                                  label="Match tolerance (higher = looser)"))

            # --------------------------------------------------------------- centre: preview on top, controls below
            with gr.Column(scale=5, min_width=460, elem_id="center_stage"):
                C['preview'] = gr.Image(label="Preview", interactive=False, format="jpeg", height="56vh",
                                        show_download_button=False, show_fullscreen_button=True, elem_id="preview_img")
                C['result_video'] = gr.Video(label="Result", visible=False, height="56vh", interactive=False)
                with gr.Column(visible=False) as editor_col:
                    C['editor_col'] = editor_col
                    C['editor'] = gr.ImageEditor(type="numpy", sources=(), transforms=(), layers=False, height="56vh",
                                                 brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", color_mode="fixed", default_size=40),
                                                 eraser=gr.Eraser(default_size=40), show_download_button=False,
                                                 label="Paint in white what stays original (all frames), then Done",
                                                 elem_id="mask_editor")
                    with gr.Row(elem_classes="fs-buttons"):
                        C['btn_paint_done'] = gr.Button("Done", variant="primary", size="sm")
                        C['btn_paint_clear'] = gr.Button("Clear", size="sm")
                        C['btn_paint_cancel'] = gr.Button("Cancel", size="sm")
                # the timeline right under the picture, like a player
                with gr.Row(equal_height=True, visible=video, elem_id="frame_bar") as frame_row:
                    C['frame_row'] = frame_row
                    C['btn_prev'] = gr.Button("◀", size="sm", scale=0, min_width=40, elem_id="frame_prev")
                    C['frame'] = gr.Slider(1, max(2, (t or {}).get('frames', 2)), value=1, step=1, label="Frame  (← → keys)",
                                           scale=6, elem_id="frame_slider")
                    C['btn_next'] = gr.Button("▶", size="sm", scale=0, min_width=40, elem_id="frame_next")
                with gr.Row(equal_height=True, visible=video, elem_id="range_bar") as range_row:
                    C['range_row'] = range_row
                    C['btn_start_here'] = gr.Button("From here", size="sm", scale=0, min_width=90)
                    C['btn_end_here'] = gr.Button("To here", size="sm", scale=0, min_width=80)
                    C['btn_whole'] = gr.Button("Whole video", size="sm", scale=0, min_width=100)
                    C['range_md'] = gr.Markdown(S.range_text(), elem_classes="fs-line", elem_id="range_line")
                with gr.Row(equal_height=True, elem_id="view_bar") as view_bar:
                    C['view_bar'] = view_bar
                    C['view'] = gr.Radio(S.VIEWS, value="Swapped", show_label=False, container=False, scale=6, elem_id="view_radio")
                    C['auto'] = gr.Checkbox(value=True, label="Auto-update", container=False, scale=0, min_width=120)
                    C['btn_refresh'] = gr.Button("Refresh", size="sm", scale=0, min_width=80)
                with gr.Row(equal_height=True, elem_id="run_bar"):
                    C['ready_md'] = gr.Markdown(S.readiness()[1], elem_id="ready_line")
                    C['btn_start'] = gr.Button("▶ Start", variant="primary", scale=0, min_width=120)
                    C['btn_stop'] = gr.Button("⏹ Stop", variant="secondary", scale=0, min_width=90,
                                              interactive=S.run_lock.locked())
                C['status_md'] = gr.Markdown("", elem_id="status_line")
                C['results'] = gr.Files(label="Results of the last run", interactive=False, visible=False, elem_id="results")

            # --------------------------------------------------------------- right: settings
            with gr.Column(scale=3, min_width=300, elem_id="fs_settings"):
                with gr.Row(elem_classes="fs-buttons"):
                    C['btn_save_def'] = gr.Button("Save my defaults", size="sm", min_width=90)
                    C['btn_load_def'] = gr.Button("Load my defaults", size="sm", min_width=90, interactive=S.has_saved_defaults())
                    C['btn_reset_def'] = gr.Button("Factory defaults", size="sm", min_width=90)

                with gr.Accordion(f"Swap · {S.summary('swap')}", open=True) as acc_swap:
                    with gr.Row():
                        _s('resolution', gr.Dropdown(S.RESOLUTIONS, value=V['resolution'], label="Resolution"))
                        _s('passes', gr.Slider(1, 5, value=V['passes'], step=1, label="Passes (1 = normal)"))
                    with gr.Row():
                        _s('identity', gr.Slider(0.0, 1.0, value=V['identity'], step=0.05, label="Identity boost (0 = off)"))
                        _s('face_shape', gr.Slider(0.0, 1.0, value=V['face_shape'], step=0.05, label="Source jaw / chin shape (0 = off)"))

                with gr.Accordion(f"Expression · {S.summary('expression')}", open=True) as acc_expr:
                    er = _s('er', gr.Checkbox(value=V['er'], label="Restore target expression (LivePortrait)"))
                    with gr.Column(visible=V['er']) as er_col:
                        _s('er_strength', gr.Slider(0, 500, value=V['er_strength'], step=1, label="Strength (100 = as the target)"))
                        with gr.Row(elem_id="expr_checks"):
                            _s('er_eyes', gr.Checkbox(value=V['er_eyes'], label="Eyes / blink"))
                            _s('er_mouth', gr.Checkbox(value=V['er_mouth'], label="Mouth"))
                            _s('er_brows', gr.Checkbox(value=V['er_brows'], label="Brows"))
                        with gr.Accordion("Fine-tune", open=False):
                            _s('er_smoothing', gr.Slider(0.0, 1.0, value=V['er_smoothing'], step=0.05, label="Expression smoothing (video)"))
                            _s('er_pose_lock', gr.Checkbox(value=V['er_pose_lock'], label="Pose lock (no head drift)"))
                            _s('er_pose_gate', gr.Checkbox(value=V['er_pose_gate'], label="Skip at extreme head angles"))
                            _s('er_structure', gr.Checkbox(value=V['er_structure'], label="Expression only (keep the swapped face shape)"))
                    with gr.Row(elem_classes="fs-checks"):
                        _s('keep_mouth', gr.Checkbox(value=V['keep_mouth'], label="Paste back target's mouth"))
                        _s('keep_eyes', gr.Checkbox(value=V['keep_eyes'], label="Paste back target's eyes"))

                with gr.Accordion(f"Occlusion · {S.summary('occlusion')}", open=False) as acc_occ:
                    engine = _s('mask_engine', gr.Dropdown(list(S.MASK_ENGINES), value=V['mask_engine'],
                                                           label="Mask model (hands, hair … stay original)"))
                    with gr.Column(visible=V['mask_engine'] == 'Clip2Seg (by text)') as clip_col:
                        _s('mask_objects', gr.Textbox(value=V['mask_objects'], label="Objects to keep (comma separated)",
                                                      max_lines=1))
                    C['paint_md'] = gr.Markdown(S.mask_info(), visible=bool(S.mask_info()), elem_classes="fs-line")
                    with gr.Row(elem_classes="fs-buttons"):
                        C['btn_paint'] = gr.Button("Paint areas that stay original", size="sm")
                        C['btn_paint_remove'] = gr.Button("Remove painting", size="sm",
                                                          interactive=(t or {}).get('mask') is not None)
                    with gr.Accordion("Edges & blending", open=False):
                        with gr.Row():
                            _s('erosion', gr.Slider(1, 3, value=V['erosion'], step=1, label="Edge erosion"))
                            _s('blur', gr.Slider(4, 100, value=V['blur'], step=1, label="Edge blur"))
                        _s('to_chin', gr.Checkbox(value=V['to_chin'], label="Extend swap to chin"))
                        _s('aligned_edges', gr.Checkbox(value=V['aligned_edges'], label="Face-aligned edges (tilted faces)"))
                        with gr.Row():
                            _s('crop_top', gr.Slider(0.0, 0.99, value=V['crop_top'], step=0.01, label="Crop top"))
                            _s('crop_bottom', gr.Slider(0.0, 0.99, value=V['crop_bottom'], step=0.01, label="Crop bottom"))
                        with gr.Row():
                            _s('crop_left', gr.Slider(0.0, 0.99, value=V['crop_left'], step=0.01, label="Crop left"))
                            _s('crop_right', gr.Slider(0.0, 0.99, value=V['crop_right'], step=0.01, label="Crop right"))
                        _s('color_transfer', gr.Checkbox(value=V['color_transfer'], label="Match colours to target"))

                with gr.Accordion(f"Enhance · {S.summary('enhance')}", open=False) as acc_enh:
                    enh = _s('enhancer', gr.Dropdown(list(S.ENHANCERS), value=V['enhancer'], label="Enhancer"))
                    with gr.Column(visible=V['enhancer'] != 'None') as enh_col:
                        _s('enhancer_blend', gr.Slider(0.0, 1.0, value=V['enhancer_blend'], step=0.01, label="Strength"))
                        _s('mask_after_enhancer', gr.Checkbox(value=V['mask_after_enhancer'], label="Occlusion mask after enhancer"))

                with gr.Accordion(f"Detection & tracking · {S.summary('detection')}", open=False) as acc_det:
                    with gr.Row():
                        _s('det_thresh', gr.Slider(0.10, 0.90, value=V['det_thresh'], step=0.01, label="Detection confidence"))
                        _s('det_size', gr.Dropdown([320, 640, 1024], value=V['det_size'], label="Detection size"))
                    C['multi_angle'] = _s('multi_angle', gr.Dropdown(list(S.MULTI_ANGLE), value=V['multi_angle'], label="Rotated faces"))
                    with gr.Column(visible=S.MULTI_ANGLE[V['multi_angle']] == 'always') as upright_col:
                        C['upright_col'] = upright_col
                        _s('upright', gr.Slider(0.0, 1.0, value=V['upright'], step=0.05, label="Upright priority"))
                    _s('autorotate', gr.Checkbox(value=V['autorotate'], label="Auto-rotate lying faces"))
                    lmk = _s('lmk_align', gr.Checkbox(value=V['lmk_align'], label="68-point alignment"))
                    with gr.Column(visible=V['lmk_align']) as lmk_col:
                        _s('hi_lmk', gr.Checkbox(value=V['hi_lmk'], label="High-accuracy landmarks (2dfan4)"))
                        C['lmk_gate'] = _s('lmk_gate', gr.Checkbox(value=V['lmk_gate'], label="Landmark sanity gate"))
                        with gr.Column(visible=V['lmk_gate']) as gate_col:
                            C['gate_col'] = gate_col
                            _s('lmk_gate_thr', gr.Slider(0.0, 0.50, value=V['lmk_gate_thr'], step=0.005, label="Gate threshold"))
                    sm = _s('smoothing', gr.Checkbox(value=V['smoothing'], label="Landmark smoothing (video)"))
                    with gr.Column(visible=V['smoothing']) as sm_col:
                        with gr.Row():
                            _s('smoothing_strength', gr.Slider(0.0, 1.0, value=V['smoothing_strength'], step=0.05, label="Strength"))
                            _s('smoothing_deadzone', gr.Slider(0.0, 0.03, value=V['smoothing_deadzone'], step=0.001,
                                                               label="Still-face threshold (0 = off)"))

                with gr.Accordion(f"Video output · {S.summary('video')}", open=False) as acc_vid:
                    method = _s('method', gr.Radio([(S.METHOD_MEMORY, S.METHOD_MEMORY),
                                                    (f"{S.METHOD_EXTRACT} (no smoothing)", S.METHOD_EXTRACT)],
                                                   value=V['method'], label="Processing"))
                    with gr.Column(visible=V['method'] == S.METHOD_EXTRACT) as keep_col:
                        _s('keep_frames', gr.Checkbox(value=V['keep_frames'], label="Keep extracted frames (until the next start)"))
                    _s('no_face', gr.Dropdown(list(S.NO_FACE), value=V['no_face'], label="When no face gets swapped in a frame"))
                    _s('skip_audio', gr.Checkbox(value=V['skip_audio'], label="No audio"))
                    C['out_fps'] = gr.Number(value=(t or {}).get('out_fps', 0), label="Output fps, this video (0 = original)",
                                             precision=2, minimum=0, visible=video)

        C.update(acc_swap=acc_swap, acc_expr=acc_expr, acc_occ=acc_occ, acc_enh=acc_enh, acc_det=acc_det, acc_vid=acc_vid)

    _wire(tick, er, er_col, engine, clip_col, enh, enh_col, lmk, lmk_col, sm, sm_col, method, keep_col, mode)
    return register_load


# ============================================================================ wiring

def _wire(tick, er, er_col, engine, clip_col, enh, enh_col, lmk, lmk_col, sm, sm_col, method, keep_col, mode):
    # state handlers run one at a time; only the slow ones (adding files) show
    # a progress overlay, which lands on every output (the readiness line too)
    slow = dict(concurrency_id="fs_state", concurrency_limit=1, **INTERNAL)
    one = dict(slow, show_progress="hidden")

    # every setting change refreshes the preview (browser-side bump, coalesced)
    for key, comp in settings.items():
        if key == 'mask_objects':
            comp.submit(None, [tick], [tick], js=BUMP_JS, **INTERNAL)
            comp.blur(None, [tick], [tick], js=BUMP_JS, **INTERNAL)
        else:
            comp.change(None, [tick], [tick], js=BUMP_JS, **INTERNAL)
    for comp in (C['view'], C['auto']):
        comp.change(None, [tick], [tick], js=BUMP_JS, **INTERNAL)
    C['frame'].release(None, [tick], [tick], js=BUMP_JS, **INTERNAL)

    # children shown only when their parent is on (browser only)
    show = lambda cond: f"(v) => ({{__type__: 'update', visible: {cond}}})"
    er.change(None, [er], [er_col], js=show('v'), **INTERNAL)
    engine.change(None, [engine], [clip_col], js=show("v === 'Clip2Seg (by text)'"), **INTERNAL)
    enh.change(None, [enh], [enh_col], js=show("v !== 'None'"), **INTERNAL)
    lmk.change(None, [lmk], [lmk_col], js=show('v'), **INTERNAL)
    sm.change(None, [sm], [sm_col], js=show('v'), **INTERNAL)
    method.change(None, [method], [keep_col], js=show(f"v === '{S.METHOD_EXTRACT}'"), **INTERNAL)
    always = next(k for k, v in S.MULTI_ANGLE.items() if v == 'always')
    C['multi_angle'].change(None, [C['multi_angle']], [C['upright_col']], js=show(f"v === '{always}'"), **INTERNAL)
    C['lmk_gate'].change(None, [C['lmk_gate']], [C['gate_col']], js=show('v'), **INTERNAL)

    # preview
    preview_inputs = set(settings.values()) | {C['view'], C['frame'], C['auto'], tick}
    preview_outputs = [C['preview'], C['result_video'], C['ready_md'], C['acc_swap'], C['acc_expr'], C['acc_occ'],
                       C['acc_enh'], C['acc_det'], C['acc_vid']]
    tick.change(on_preview, preview_inputs, preview_outputs, trigger_mode="always_last", concurrency_id="fs_preview",
                concurrency_limit=1, show_progress="hidden", **INTERNAL)
    C['btn_refresh'].click(lambda d: on_preview(d, force=True), preview_inputs, preview_outputs, concurrency_id="fs_preview",
                           concurrency_limit=1, show_progress="hidden", **INTERNAL)
    people_out = [C['people_gal'], C['people_hint'], C['btn_person_remove'], C['ready_md'], tick]
    C['preview'].select(on_preview_click, [C['frame'], C['view'], tick], people_out, **one)

    # sources
    det = [settings['det_thresh'], settings['det_size']]
    src_out = [C['src_gal'], C['src_info'], C['btn_src_combine'], C['btn_src_shuffle'], C['btn_src_undo'],
               C['btn_src_remove'], C['btn_src_clear'], C['ready_md'], tick]
    fix_src = dict(fn=src_highlight,
                   inputs=None, outputs=C['src_gal'], show_progress="hidden", **INTERNAL)
    C['src_drop'].upload(on_src_upload, [C['src_drop'], tick] + det, [C['src_drop']] + src_out, **slow).then(**fix_src)
    for ev in (C['btn_src_path'].click, C['src_path'].submit):          # the Add button or Enter
        ev(on_src_path, [C['src_path'], tick] + det, src_out, **slow).then(**fix_src)
    # the mode renumbers the source captions (One source per face): keep the highlight
    mode.change(on_mode, [mode], [C['people_col'], C['src_gal'], C['src_info'], C['btn_src_shuffle'],
                                  C['ready_md']], **one).then(**fix_src)
    C['src_gal'].select(on_src_select, [tick], [C['src_info'], C['btn_src_combine'], C['ready_md'], tick], **one)
    C['btn_src_remove'].click(on_src_remove, [tick], src_out, **one).then(**fix_src)
    C['btn_src_clear'].click(on_src_clear, [tick], src_out, **one).then(**fix_src)
    C['btn_src_undo'].click(on_src_undo, [tick], src_out, **one).then(**fix_src)
    C['btn_src_combine'].click(on_src_combine, [tick], src_out, **slow).then(**fix_src)
    C['btn_src_shuffle'].click(on_src_shuffle, [tick], src_out, **one).then(**fix_src)

    # targets
    tgt_out = [C['tgt_gal'], C['frame_row'], C['range_row'], C['frame'], C['range_md'], C['out_fps'],
               C['paint_md'], C['btn_paint_remove'], C['btn_tgt_undo'], C['btn_tgt_remove'], C['btn_tgt_clear'],
               C['ready_md'], tick]
    fix_tgt = dict(fn=tgt_highlight, inputs=None, outputs=C['tgt_gal'],
                   show_progress="hidden", **INTERNAL)
    C['tgt_drop'].upload(on_tgt_upload, [C['tgt_drop'], tick], [C['tgt_drop']] + tgt_out, **slow).then(**fix_tgt)
    for ev in (C['btn_tgt_path'].click, C['tgt_path'].submit):
        ev(on_tgt_path, [C['tgt_path'], tick], tgt_out, **slow).then(**fix_tgt)
    C['tgt_gal'].select(on_tgt_select, [tick], tgt_out, **one)
    C['btn_tgt_remove'].click(on_tgt_remove, [tick], tgt_out, **one).then(**fix_tgt)
    C['btn_tgt_clear'].click(on_tgt_clear, [tick], tgt_out, **one).then(**fix_tgt)
    C['btn_tgt_undo'].click(on_tgt_undo, [tick], tgt_out, **one).then(**fix_tgt)
    C['btn_prev'].click(None, [tick], [tick], js=STEP_JS % -1, **INTERNAL)
    C['btn_next'].click(None, [tick], [tick], js=STEP_JS % 1, **INTERNAL)
    range_out = [C['range_md'], C['tgt_gal'], C['ready_md']]
    C['btn_start_here'].click(lambda f: on_range('start', f), [C['frame']], range_out, **one).then(**fix_tgt)
    C['btn_end_here'].click(lambda f: on_range('end', f), [C['frame']], range_out, **one).then(**fix_tgt)
    C['btn_whole'].click(lambda f: on_range('whole', f), [C['frame']], range_out, **one).then(**fix_tgt)
    C['out_fps'].input(on_out_fps, [C['out_fps']], None, **one)

    # people
    fix_people = dict(fn=people_highlight, inputs=None,
                      outputs=C['people_gal'], show_progress="hidden", **INTERNAL)
    C['people_gal'].select(on_person_select, None, None, **one)
    C['btn_person_remove'].click(on_person_remove, [tick], people_out, **one).then(**fix_people)

    # painted keep-original mask
    paint_out = [C['editor_col'], C['preview'], C['editor'], C['paint_md'], C['btn_paint_remove'], C['tgt_gal'],
                 C['acc_occ'], C['view_bar'], C['frame_row'], C['range_row'], tick]
    C['btn_paint'].click(on_paint_open, [C['frame'], tick], paint_out, **one)
    C['btn_paint_done'].click(on_paint_done, [C['editor'], C['frame'], tick], paint_out, **one).then(**fix_tgt)
    C['btn_paint_clear'].click(on_paint_clear, [C['frame'], tick], paint_out, **one)
    C['btn_paint_cancel'].click(on_paint_cancel, [tick], paint_out, **one)
    C['btn_paint_remove'].click(on_paint_remove, [tick], paint_out, **one).then(**fix_tgt)

    # run
    run_inputs = set(settings.values()) | {tick}
    start = C['btn_start'].click(on_start_check, run_inputs, [C['btn_start'], C['btn_stop'], C['status_md'], C['results']],
                                 show_progress="hidden", **INTERNAL)
    render = start.success(on_render, None, [C['status_md']], show_progress="full", **INTERNAL)
    render.then(on_render_done, [tick], [C['btn_start'], C['btn_stop'], C['status_md'], C['results'], C['view'],
                                        C['ready_md'], tick], show_progress="hidden", **INTERNAL)
    C['btn_stop'].click(on_stop, None, [C['btn_stop'], C['status_md']], queue=False, **INTERNAL)
    C['results'].select(on_result_select, [C['results'], tick], [C['view'], tick], **one)

    # defaults
    C['btn_save_def'].click(on_save_defaults, set(settings.values()), [C['btn_load_def']], **one)
    keys = list(settings)
    C['btn_load_def'].click(lambda n: on_load_defaults(keys, n), [tick], [settings[k] for k in keys] + [tick], **one)
    C['btn_reset_def'].click(lambda n: on_factory_defaults(keys, n), [tick], [settings[k] for k in keys] + [tick], **one)


def register_load(ui):
    """Page load: show the server's state (sources, targets, people, the last
    applied settings), not the build-time values."""
    keys = list(settings)
    outputs = [settings[k] for k in keys] + refresh_outputs()
    def on_load():
        _page_loaded()
        return [S.values[k] for k in keys] + refresh_values()
    ui.load(on_load, None, outputs, show_progress="hidden", **INTERNAL).then(
        lambda: [src_highlight(), tgt_highlight()],
        None, [C['src_gal'], C['tgt_gal']], show_progress="hidden", **INTERNAL)


def refresh_outputs():
    """Components refresh_values() fills (also used by other tabs)."""
    return [C['src_gal'], C['src_info'], C['btn_src_combine'], C['btn_src_shuffle'], C['btn_src_undo'],
            C['btn_src_remove'], C['btn_src_clear'],
            C['tgt_gal'], C['frame_row'], C['range_row'], C['frame'], C['range_md'], C['out_fps'],
            C['paint_md'], C['btn_paint_remove'], C['btn_tgt_undo'], C['btn_tgt_remove'], C['btn_tgt_clear'],
            C['people_col'], C['people_gal'], C['people_hint'], C['btn_person_remove'],
            C['ready_md'], C['btn_start'], C['btn_stop'], C['tick']]


def refresh_values():
    # Start stays clickable here: this page may be a reload during a render,
    # and only the page that pressed Start hears when it ends (a second Start
    # is refused; a Stop with nothing running says so). The preview tick is a
    # new value (milliseconds): the page's own tick counts up by one.
    return (_src_updates()[:7] + _tgt_updates()[:11] +
            [gr.Column(visible=S.MODES[S.values['mode']] == 'selected')] + _people_updates() +
            [gr.Markdown(S.readiness()[1]), gr.Button(interactive=True), gr.Button(interactive=_rendering()),
             int(time.time() * 1000)])


# ============================================================================ updates

def _src_info_update():
    text = S.source_info()
    return gr.Markdown(value=text, visible=bool(text))


def _src_updates():
    n = len(G.INPUT_FACESETS)
    return [_gallery('src', S.sources_gallery()), _src_info_update(),
            gr.Button(value=S.combine_label(), visible=len(S.same_person_photos()) >= 2),
            gr.Button(visible=S.MODES[S.values['mode']] == 'all_input' and n > 1),
            gr.Button(value=S.source_undo_label() or "Undo", interactive=S.source_undo_label() is not None),
            gr.Button(interactive=n > 0), gr.Button(interactive=n > 0),
            gr.Markdown(S.readiness()[1])]


def _tgt_updates():
    t = S.target()
    video = t is not None and t['kind'] != 'image'
    frames = t['frames'] if video else 2
    return [_gallery('tgt', S.targets_gallery()),
            gr.Row(visible=video), gr.Row(visible=video),
            gr.Slider(minimum=1, maximum=max(2, frames), value=t['start'] if video else 1),
            gr.Markdown(S.range_text()),
            gr.Number(value=t['out_fps'] if video else 0, visible=video),
            _paint_md_update(), _paint_remove_update(),
            gr.Button(value=S.undo_label() or "Undo", interactive=S.undo_label() is not None),
            gr.Button(interactive=bool(S.targets)), gr.Button(interactive=bool(S.targets)),
            gr.Markdown(S.readiness()[1])]


def _people_updates():
    have = bool(G.TARGET_FACES)
    return [_gallery('people', S.people_gallery()), gr.Markdown(visible=not have), gr.Button(interactive=have)]


def _messages(messages):
    problems = [m for m in messages if 'not added' in m or 'skipped' in m or 'not found' in m.lower()
                or 'cannot' in m or 'flagged' in m or 'Review' in m or m.startswith(('Not ', 'No ', 'Type '))]
    if problems:
        gr.Warning('\n'.join(problems[:8]) + (f'\n... and {len(problems) - 8} more' if len(problems) > 8 else ''))
    elif messages:
        _info('\n'.join(messages[:6]) + (f'\n... and {len(messages) - 6} more' if len(messages) > 6 else ''))


def _apply_detection(thresh, size):
    """Source loading and face picking detect with the panel's values (not
    while a render runs: it reads them per frame)."""
    from roop import core
    if not core.render_active:
        S.values['det_thresh'], S.values['det_size'] = thresh, size
        G.det_thresh, G.det_size = float(thresh), int(size)


# ============================================================================ handlers: sources

def on_mode(mode):
    S.values['mode'] = mode
    n = len(G.INPUT_FACESETS)
    return [gr.Column(visible=S.MODES[mode] == 'selected'),
            _gallery('src', S.sources_gallery()), _src_info_update(),
            gr.Button(visible=S.MODES[mode] == 'all_input' and n > 1), gr.Markdown(S.readiness()[1])]


def on_src_upload(files, tick, thresh, size, progress=gr.Progress()):
    if files:
        _apply_detection(thresh, size)
        _messages(S.add_sources([f.name if hasattr(f, 'name') else str(f) for f in files], progress))
    return [None] + _src_updates() + [(tick or 0) + 1]


def on_src_path(path, tick, thresh, size):
    _apply_detection(thresh, size)
    _messages(S.add_source_path(path))
    return _src_updates() + [(tick or 0) + 1]


def on_src_select(evt: gr.SelectData, tick):
    if evt is None or _is_echo('src', evt.index) or evt.index == S.active_source_index():
        return [gr.skip()] * 4
    S.select_source(evt.index)
    return [_src_info_update(), gr.Button(value=S.combine_label(), visible=len(S.same_person_photos()) >= 2),
            gr.Markdown(S.readiness()[1]), (tick or 0) + 1]


def on_src_remove(tick):
    if not S.remove_active_source():
        gr.Warning('No source to remove')
    return _src_updates() + [(tick or 0) + 1]


def on_src_clear(tick):
    S.clear_sources()
    return _src_updates() + [(tick or 0) + 1]


def on_src_undo(tick):
    label = S.undo_sources()
    if label:
        _info(f'Undone: {label}')
    return _src_updates() + [(tick or 0) + 1]


def on_src_combine(tick):
    _info(S.combine_photo_sources())
    return _src_updates() + [(tick or 0) + 1]


def on_src_shuffle(tick):
    S.shuffle_sources()
    return _src_updates() + [(tick or 0) + 1]


# ============================================================================ handlers: targets

def on_tgt_upload(files, tick, progress=gr.Progress()):
    if files:
        _messages(S.add_targets([f.name if hasattr(f, 'name') else str(f) for f in files], progress))
    return [None] + _tgt_updates() + [(tick or 0) + 1]


def on_tgt_path(path, tick):
    _messages(S.add_target_path(path))
    return _tgt_updates() + [(tick or 0) + 1]


def on_tgt_select(evt: gr.SelectData, tick):
    if evt is None or _is_echo('tgt', evt.index) or evt.index == S.selected_target_index():
        return [gr.skip()] * 13
    S.select_target(evt.index)
    return _tgt_updates() + [(tick or 0) + 1]


def on_tgt_remove(tick):
    if not S.remove_selected_target():
        gr.Warning('No file to remove')
    return _tgt_updates() + [(tick or 0) + 1]


def on_tgt_clear(tick):
    if S.targets:
        S.clear_targets()
    return _tgt_updates() + [(tick or 0) + 1]


def on_tgt_undo(tick):
    label = S.undo_targets()
    if label:
        _info(f'Undone: {label}')
    return _tgt_updates() + [(tick or 0) + 1]


def on_range(which, frame):
    if which == 'whole':
        S.reset_range()
    else:
        note = S.set_range(which, frame)
        if note:
            _info(note)
    return [gr.Markdown(S.range_text()), _gallery('tgt', S.targets_gallery()), gr.Markdown(S.readiness()[1])]


def on_out_fps(value):
    S.set_out_fps(value)


# ============================================================================ handlers: people

def on_person_select(evt: gr.SelectData):
    if evt is not None and not _is_echo('people', evt.index):
        S.selected_person = evt.index


def on_person_remove(tick):
    if not S.remove_selected_person():
        gr.Warning('No person to remove')
    return _people_updates() + [gr.Markdown(S.readiness()[1]), (tick or 0) + 1]


def on_preview_click(evt: gr.SelectData, frame, view, tick):
    if S.MODES[S.values['mode']] != 'selected':
        return [gr.skip()] * 5                             # clicking the preview only picks in Specific people
    if view not in ('Original', 'Swapped') or not _preview['size'] or evt is None:
        _info('Pick people in the Original or Swapped view')
        return [gr.skip()] * 5
    x, y = evt.index[0], evt.index[1]
    msg = S.pick_person_at(frame, x, y, _preview['size'])
    (_info if msg == 'Person added' else gr.Warning)(msg)
    return _people_updates() + [gr.Markdown(S.readiness()[1]), (tick or 0) + 1]


# ============================================================================ handlers: painted mask

_painting = {'tid': None, 'frame': None}      # the file and frame the open editor paints on


def _current_frame(frame_num, tid=None):
    t = S.target(tid)
    if t is None or not os.path.isfile(t['path']):
        return None
    if t['kind'] == 'image':
        return get_image_frame(t['path'])
    return get_video_frame(t['path'], int(frame_num or 1))


def _editor_background(frame_num, tid=None):
    frame = _current_frame(frame_num, tid)
    return None if frame is None else util.convert_to_gradio_preview(frame)


def _paint_md_update():
    text = S.mask_info()
    return gr.Markdown(value=text, visible=bool(text))


def _paint_remove_update():
    return gr.Button(interactive=(S.target() or {}).get('mask') is not None)


def _paint_view(open_editor, editor_value=gr.skip()):
    """The editor in the preview's place; view, frame and range controls do
    nothing for it, so they hide until Done / Cancel."""
    t = S.target()
    video = t is not None and t['kind'] != 'image'
    return [gr.Column(visible=open_editor), gr.Image(visible=not open_editor), editor_value,
            _paint_md_update(), _paint_remove_update(), _gallery('tgt', S.targets_gallery()),
            gr.Accordion(label=f"Occlusion · {S.summary('occlusion')}"),
            gr.Row(visible=not open_editor), gr.Row(visible=video and not open_editor),
            gr.Row(visible=video and not open_editor)]


def on_paint_open(frame, tick):
    bg = _editor_background(frame)
    if bg is None:
        gr.Warning('Add a target file first')
        return _paint_view(False) + [tick]
    t = S.target()
    _painting.update(tid=t['id'], frame=int(frame or 1))
    layer = t['mask'] if t['mask'] is not None and t['mask'].shape[:2] == bg.shape[:2] else None
    return _paint_view(True, {'background': bg, 'layers': [layer] if layer is not None else [], 'composite': None}) + [tick]


def on_paint_done(editor, frame, tick):
    # on the file the editor was opened for, even if another one was clicked meanwhile
    tid = _painting['tid']
    t = S.target(tid) if tid is not None else None
    painted = t is not None and S.set_target_mask(editor, _painting['frame'], tid)
    _painting.update(tid=None, frame=None)
    if t is None:
        gr.Warning('The file was removed while painting: nothing saved')
    else:
        _info(f"Painted areas saved for {t['name']}" if painted else f"Nothing painted: no mask on {t['name']}")
    return _paint_view(False) + [(tick or 0) + 1]


def on_paint_clear(frame, tick):
    bg = _editor_background(_painting['frame'] or frame, _painting['tid'])
    return _paint_view(True, {'background': bg, 'layers': [], 'composite': None}) + [tick]


def on_paint_cancel(tick):
    _painting.update(tid=None, frame=None)
    return _paint_view(False) + [(tick or 0) + 1]


def on_paint_remove(tick):
    S.clear_target_mask()
    _painting.update(tid=None, frame=None)
    return _paint_view(False) + [(tick or 0) + 1]


# ============================================================================ preview

def _label(t, frame_num, text):
    where = t['name']
    if t['kind'] != 'image':
        clock = S.clock(frame_num, t['fps'])
        where += f' · {clock}' if clock else f' · frame {frame_num}'
    return f'{where} — {text}' if text else where


def _headers():
    return [gr.Accordion(label=f"Swap · {S.summary('swap')}"), gr.Accordion(label=f"Expression · {S.summary('expression')}"),
            gr.Accordion(label=f"Occlusion · {S.summary('occlusion')}"), gr.Accordion(label=f"Enhance · {S.summary('enhance')}"),
            gr.Accordion(label=f"Detection & tracking · {S.summary('detection')}"), gr.Accordion(label=f"Video output · {S.summary('video')}")]


def on_preview(data, force=False):
    from roop import core
    vals = _vals(data)
    view, frame_num, auto = data[C['view']], int(data[C['frame']] or 1), data[C['auto']]

    def work():
        S.apply_settings(vals)
        return _render_view(view, frame_num, auto or force)

    try:
        ran, out = core.preview_locked(work)
    except Exception as e:
        traceback.print_exc()
        return [gr.Image(label=f'Preview failed: {e}'), gr.Video(visible=False), gr.skip()] + [gr.skip()] * 6
    if not ran:
        return [gr.Image(label='Preview paused while rendering'), gr.skip(), gr.skip()] + [gr.skip()] * 6
    image, video = out
    if _painting['tid'] is not None:
        image, video = gr.skip(), gr.skip()        # the editor stands in for the preview until Done / Cancel
    return [image, video, gr.Markdown(S.readiness()[1])] + _headers()


def _render_view(view, frame_num, swap_now):
    """(image update, video update) for the preview area."""
    from roop import core
    if view == 'Result':
        return _result_view()
    t = S.target()
    if t is None:
        _preview['size'] = None
        return gr.Image(value=None, label='Preview — add a target file', visible=True), gr.Video(visible=False)
    if not os.path.isfile(t['path']):
        return gr.Image(value=None, label='Preview — this file is gone (temp folder cleaned?): add it again', visible=True), gr.Video(visible=False)
    t_start = time.perf_counter()
    frame = get_image_frame(t['path']) if t['kind'] == 'image' else get_video_frame(t['path'], frame_num)
    if frame is None:
        return gr.Image(value=None, label=_label(t, frame_num, 'frame could not be read'), visible=True), gr.Video(visible=False)

    note = ''
    shown = frame
    t_swap = time.perf_counter()
    if view == 'Mask':
        shown = core.mask_view(frame, S.build_options(mask_view=True))
        plugin, warn = S.mask_plugin()
        note = 'green = swapped' + (f' ({warn})' if warn else '' if plugin else ' (no occlusion mask)')
    elif view in ('Swapped', 'Side by side'):
        if not swap_now:
            return gr.skip(), gr.skip()                   # auto-update off: Refresh does it
        if not G.INPUT_FACESETS:
            note = 'add a source face'
        elif S.MODES[S.values['mode']] == 'selected' and not G.TARGET_FACES:
            pass                                          # the pick hint below says what to do
        else:
            swapped = core.live_swap(frame.copy(), S.build_options())
            if swapped is None or np.array_equal(swapped, frame):
                note = f"no face swapped ({S.values['mode']})"
            if view == 'Side by side':
                shown = np.hstack([frame, swapped if swapped is not None else frame])
                note = note or 'original | swapped'
            else:
                shown = swapped if swapped is not None else frame
    image = util.convert_to_gradio_preview(shown)
    if view != 'Original':
        # server-side time; if the browser shows it much later, the rest is the tunnel / network
        t_done = time.perf_counter()
        print(f'[preview] frame {frame_num}: load {(t_swap - t_start) * 1000:.0f} ms | {view.lower()} '
              f'{(t_done - t_swap) * 1000:.0f} ms | total {(t_done - t_start) * 1000:.0f} ms', flush=True)
    _preview['size'] = (image.shape[1], image.shape[0]) if view in ('Original', 'Swapped') else None
    _preview['view'] = view
    pick = ('click faces in the preview' if S.MODES[S.values['mode']] == 'selected' and not G.TARGET_FACES
            and view in ('Original', 'Swapped') else '')
    note = ' · '.join(x for x in (note, pick) if x)
    return gr.Image(value=image, label=_label(t, frame_num, note), visible=True), gr.Video(visible=False)


def _result_view():
    path = selected_result or (last_results[0] if last_results else None)
    if not path or not os.path.isfile(path):
        return gr.Image(value=None, label='Result — nothing rendered yet in this session', visible=True), gr.Video(visible=False)
    name = os.path.basename(path)
    if util.is_video(path) and G.CFG.output_show_video:
        return gr.Image(visible=False), gr.Video(value=path, label=f'Result — {name}', visible=True)
    frame = get_video_frame(path, 1, exact=True) if (util.is_video(path) or path.lower().endswith('.gif')) else get_image_frame(path)
    if frame is None:
        return gr.Image(value=None, label=f'Result — {name} could not be read', visible=True), gr.Video(visible=False)
    return gr.Image(value=util.convert_to_gradio_preview(frame), label=f'Result — {name}', visible=True), gr.Video(visible=False)


# ============================================================================ run

_pending = {}
_starting = 0.0             # when a Start was accepted (0: none); the render clears it


def _rendering():
    # the render takes run_lock right after the check; a Start whose render
    # never came (page closed in between) stops counting after 30 s
    return S.run_lock.locked() or (_starting > 0 and time.time() - _starting < 30)



def on_start_check(data):
    """Refuse clearly before anything happens; otherwise hand the panel values
    to the render. A refused Start leaves a running render and its buttons alone."""
    if _rendering():
        raise gr.Error('A render is already running')
    if _painting['tid'] is not None:
        raise gr.Error('Press Done or Cancel on the painting first')
    S.values.update(_vals(data))
    ready, text = S.readiness()
    if not ready:
        raise gr.Error(text.replace('To start: ', 'Cannot start yet: '))
    missing = [t['name'] for t in S.targets if not os.path.isfile(t['path'])]
    if missing:
        raise gr.Error('These files are gone (temp folder cleaned?): ' + ', '.join(missing[:5]) + '. Add them again.')
    if any(t['kind'] != 'image' for t in S.targets) and not util.is_installed('ffmpeg'):
        raise gr.Error('ffmpeg is not installed: videos cannot be rendered')
    global _starting
    from roop import core
    _pending.clear()
    _pending.update(_vals(data))
    core.stop_requested = False
    _starting = time.time()
    return [gr.Button(interactive=False), gr.Button(interactive=True), gr.Markdown('Starting ...'), gr.Files(visible=False)]


def on_render(progress=gr.Progress()):
    global last_results, selected_result, _starting
    from roop import core
    from ui.main import prepare_environment
    if not S.run_lock.acquire(blocking=False):
        return gr.Markdown('A render is already running')
    started = time.time()
    try:
        prepare_environment()
        entries = S.process_entries()
        with core._swap_lock:                     # waits for a preview; none can start after this
            S.apply_settings(_pending)
            core.render_active = True
            # everything the render uses, read here: the panel's handlers
            # (Factory defaults, source edits) may change S.values from now on
            v = dict(S.values)
            plugin, warn = S.mask_plugin()
            source = S.active_source_index()
        try:
            if G.CFG.clear_output:
                S.clear_output_media()
            G.video_encoder = G.CFG.output_video_codec
            G.video_quality = G.CFG.video_quality
            G.max_memory = G.CFG.memory_limit if G.CFG.memory_limit > 0 else None
            if warn:
                gr.Warning(warn)
            core.batch_process_regular(S.SWAP_MODEL, "File", entries, plugin, v['mask_objects'],
                                       v['method'] == S.METHOD_MEMORY, None, bool(v['keep_mouth']),
                                       bool(v['keep_eyes']), int(v['passes']), progress, source)
        finally:
            core.render_active = False
        last_results = [e.finalname for e in entries if getattr(e, 'completed', False) and os.path.isfile(e.finalname)]
        selected_result = last_results[0] if last_results else None
        secs = time.time() - started
        done = f'{len(last_results)} of {len(entries)} file{"s" if len(entries) != 1 else ""}'
        stopped = not G.processing
        G.processing = False
        if stopped and len(last_results) < len(entries):
            partial = [os.path.basename(e.finalname) for e in entries
                       if not getattr(e, 'completed', False) and e.finalname and os.path.isfile(e.finalname)]
            text = (f'**Stopped** after {secs:.0f} s: {done} finished, saved in `{G.output_path}`.' if last_results
                    else f'**Stopped** after {secs:.0f} s, before a file was finished.')
            if partial:
                text += f' The interrupted part is left as `{partial[0]}` (no sound; the next run overwrites it).'
            return gr.Markdown(text)
        return gr.Markdown(f'**Done** in {secs:.0f} s: {done} saved in `{G.output_path}`.')
    except Exception as e:
        traceback.print_exc()
        G.processing = False
        return gr.Markdown(f'**Render failed:** {e}')
    finally:
        core.stop_requested = False
        _starting = 0.0
        S.run_lock.release()


def on_render_done(tick):
    running = _rendering()
    view = gr.Radio(value='Result') if last_results else gr.skip()
    return [gr.Button(interactive=not running), gr.Button(interactive=running), gr.skip(),
            gr.Files(value=last_results or None, visible=bool(last_results)), view,
            gr.Markdown(S.readiness()[1]), (tick or 0) + 1]


def on_stop():
    from roop import core
    if not _rendering():
        return [gr.Button(interactive=False), gr.Markdown('Nothing is rendering.')]
    core.stop_requested = True        # also when the render is still starting
    G.processing = False
    return [gr.Button(interactive=False), gr.Markdown('Stopping: the frames being processed finish first ...')]


def on_result_select(evt: gr.SelectData, files, tick):
    global selected_result
    if evt is None or not files:
        return [gr.skip(), gr.skip()]
    f = files[evt.index]
    selected_result = getattr(f, 'name', None) or str(f)
    return [gr.Radio(value='Result'), (tick or 0) + 1]


# ============================================================================ defaults

def on_save_defaults(data):
    S.values.update(_vals(data))
    try:
        path = S.save_defaults()
    except OSError as e:
        raise gr.Error(f'Could not save: {e}')
    _info('Saved as your defaults (loaded at every start)')
    return gr.Button(interactive=True)


def on_load_defaults(keys, tick):
    if not S.load_my_defaults():
        gr.Warning('No saved defaults yet: set the panel up and press "Save my defaults"')
        return [gr.skip()] * (len(keys) + 1)
    _info('Your saved defaults are back')
    return [S.values[k] for k in keys] + [(tick or 0) + 1]


def on_factory_defaults(keys, tick):
    S.factory_defaults()
    _info('Factory defaults loaded (your saved defaults are kept)' if S.has_saved_defaults() else 'Factory defaults loaded')
    return [S.values[k] for k in keys] + [(tick or 0) + 1]


# ============================================================================ used by other tabs

def add_faceset_to_sources(path):
    """Face Management 'Use in Face Swap'. Returns refresh_values() for
    refresh_outputs()."""
    if path and os.path.isfile(path):
        _messages(S.add_sources([path]))
    else:
        gr.Warning('Save the faceset first')
    return refresh_values()
