import os
import re
import shutil
import tempfile
import time
import cv2
import numpy as np
import gradio as gr
import roop.utilities as util
import roop.globals
from roop import faceset_check
from roop.face_util import extract_face_images, get_all_faces
from roop.capturer import get_video_frame, get_video_frame_total
from typing import List, Tuple, Optional
from roop.typing import Frame, Face

selected_face_index = -1
# The faces in the list, in gallery order: {'image': the 512 cut-out saved in
# the .fsz, 'face': the face found in it, 'thumb': RGB copy for the gallery,
# 'metrics': faceset_check.photo_metrics, computed on the first check}.
entries = []
_seen_inputs = set()        # the Input Files list at the last change: Gradio sends the whole list each time
_video_path = None          # video shown in "Cut face from video frame"
_saved_name = None          # the .fsz just written: its own change event must not reload it
_rows = None                # last check result for the current list, None = not checked / stale
current_video_fps = 0

REPORT_HEADERS = ["#", "Similarity", "Yaw", "Pitch", "Mouth", "Face px", "Face shape", "Flags"]


def facemgr_tab() -> None:
    with gr.Tab("Face Management"):
        gr.Markdown("# Create blending facesets\nCollect several reference faces of one person into a single .fsz faceset — more angles blend into a stronger swap.")
        with gr.Row():
            with gr.Column(scale=3, min_width=420):
                with gr.Row():
                    fb_files = gr.Files(label='Input Files', file_count="multiple", file_types=["image", "video"], interactive=True)
                    fb_facesetfile = gr.Files(label='Faceset', file_count='single', file_types=['.fsz'], interactive=True)
                videoimagefst = gr.Image(label="Cut face from video frame", height=512, interactive=False, visible=True, format="jpeg")
                with gr.Row():
                    frame_num_fst = gr.Slider(1, 1, value=1, label="Frame Number", info='0:00:00', step=1.0, interactive=False)
                    fb_cutfromframe = gr.Button("Use faces from this frame", variant='secondary', interactive=False)
            with gr.Column(scale=2, min_width=320):
                faces = gr.Gallery(label="Faces in this Faceset", allow_preview=True, preview=True, height=512, object_fit="scale-down")
                with gr.Row():
                    fb_remove = gr.Button("Remove selected", variant='secondary')
                    fb_update = gr.Button("Create/Update Faceset file", variant='primary')
                    fb_clear = gr.Button("Clear all", variant='stop')
                with gr.Row():
                    fb_check = gr.Button("Check faceset", variant='secondary')
                    fb_remove_flagged = gr.Button("Remove flagged", variant='secondary')
        report_md = gr.Markdown()
        report_df = gr.Dataframe(headers=REPORT_HEADERS, interactive=False, wrap=True, visible=False)

    report = [report_md, report_df]
    video = [videoimagefst, frame_num_fst, fb_cutfromframe]
    # every event here reads and changes the same list: one at a time
    one = dict(concurrency_id='facemgr', concurrency_limit=1)
    fb_facesetfile.change(fn=on_faceset_changed, inputs=[fb_facesetfile], outputs=[faces] + report, **one)
    fb_files.change(fn=on_fb_files_changed, inputs=[fb_files], outputs=[faces] + video + report, **one)
    fb_update.click(fn=on_update_clicked, outputs=[fb_facesetfile], **one)
    fb_remove.click(fn=on_remove_clicked, outputs=[faces] + report, **one)
    fb_clear.click(fn=on_clear_clicked, outputs=[faces, fb_files, fb_facesetfile] + video + report, **one)
    fb_cutfromframe.click(fn=on_cutfromframe_clicked, inputs=[frame_num_fst], outputs=[faces] + report, **one)
    frame_num_fst.release(fn=on_frame_num_fst_changed, inputs=[frame_num_fst], outputs=[videoimagefst], **one)
    faces.select(fn=on_face_selected, **one)
    fb_check.click(fn=on_check_clicked, outputs=[faces] + report, **one)
    fb_remove_flagged.click(fn=on_remove_flagged_clicked, outputs=[faces] + report, **one)


# ----------------------------------------------------------------------------- list

def _add(face, image):
    entries.append({'image': image, 'face': face, 'thumb': util.convert_to_gradio(image), 'metrics': None})


def _changed():
    """The list changed: the last check no longer applies. The selection stays
    while its index still exists, as the gallery keeps showing it (it sends no
    new select event for that)."""
    global _rows, selected_face_index
    _rows = None
    if selected_face_index >= len(entries):
        selected_face_index = -1


def _gallery():
    items = []
    for i, e in enumerate(entries):
        caption = str(i + 1)
        if _rows is not None and _rows[i]['removable']:
            caption += ' ⚠ ' + ', '.join(f for f in _rows[i]['flags'] if f.startswith(faceset_check.REMOVABLE))
        items.append((e['thumb'], caption))
    return items


def _report():
    """(markdown, dataframe) for the current check, hidden when there is none."""
    if _rows is None:
        return gr.Markdown(value=''), gr.Dataframe(visible=False)
    table = []
    for i, (e, r) in enumerate(zip(entries, _rows)):
        m = e['metrics'] or {}
        fmt = lambda v, f: '' if v is None else f.format(v)
        table.append([i + 1, fmt(r['sim'], '{:.2f}'), fmt(m.get('yaw'), '{:.0f}°'), fmt(m.get('pitch'), '{:.0f}°'),
                      '' if m.get('mouth') is None else ('open' if m['mouth'] > _mouth_open() else 'closed'),
                      fmt(m.get('size'), '{:.0f}'), r['shape'] or '',
                      ', '.join(r['flags'])])
    text = ('**' + faceset_check.summary(_rows) + '.** ⚠ = suggested for removal (Remove flagged); the thresholds '
            'are heuristics, so check those photos by eye. Similarity = to the average of the other photos. '
            'Face shape: yes = used for "Face shape from source" (near-frontal, mouth closed), '
            'fill = used to reach 3 photos, no = not used.')
    return gr.Markdown(value=text), gr.Dataframe(value=table, headers=REPORT_HEADERS, visible=True)


def _mouth_open():
    from roop.face_shape import MOUTH_OPEN
    return MOUTH_OPEN


def _outputs():
    return [_gallery(), *_report()]


# ----------------------------------------------------------------------------- events

def on_faceset_changed(faceset, progress=gr.Progress()):
    """Open a .fsz for editing: its faces REPLACE the list (one face per PNG)."""
    global _saved_name

    if faceset is None:
        return _outputs()
    filename = faceset.name
    if _saved_name is not None and os.path.basename(filename) == _saved_name:
        _saved_name = None          # the file Create/Update just wrote: the list already is its content
        return _outputs()
    _saved_name = None
    if not filename.lower().endswith('fsz'):
        return _outputs()

    had_faces = len(entries) > 0
    entries.clear()
    _changed()
    folder = tempfile.mkdtemp(prefix='faceset_')
    try:
        util.unzip(filename, folder)
        pngs = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
        pngs.sort(key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', f)])
        missing = []
        for k, file in enumerate(pngs):
            progress(k / max(len(pngs), 1), desc="Retrieving faces from Faceset File")
            path = os.path.join(folder, file)
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                missing.append(file)
                continue
            if image.shape[:2] == (512, 512):
                # a faceset cut-out: keep it as it is, with its main face
                found = get_all_faces(image)
                if not found:
                    missing.append(file)
                    continue
                _add(max(found, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])), image)
            else:
                # another tool's faceset (whole photos): cut out each face
                found = extract_face_images(path, (False, 0), 0.5)
                if not found:
                    missing.append(file)
                for face, image in found:
                    _add(face, image)
        if missing:
            gr.Warning(f"No face detected in {len(missing)} image(s) of the faceset: {', '.join(missing[:5])}")
        if had_faces:
            gr.Info('The faceset file replaced the faces that were in the list')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return _outputs()


def on_fb_files_changed(inputfiles, progress=gr.Progress()):
    """Add the faces of NEW input files (Gradio sends the whole list each time;
    files already added are skipped). A video goes to the frame cutter."""
    global _video_path, current_video_fps

    video_image, slider, cut_button = gr.update(), gr.update(), gr.update()
    current = [f.name for f in inputfiles or []]
    # only files that were not in the list at the last change; a file taken
    # out of Input Files and put back counts as new again
    new = [p for p in current if p not in _seen_inputs]
    _seen_inputs.clear()
    _seen_inputs.update(current)
    added = 0
    for k, source_path in enumerate(new):
        progress(k / max(len(new), 1), desc="Retrieving faces from images")
        if util.has_image_extension(source_path):
            roop.globals.source_path = source_path
            for face, image in extract_face_images(source_path, (False, 0), 0.5):
                _add(face, image)
                added += 1
        elif util.is_video(source_path) or source_path.lower().endswith('gif'):
            _video_path = source_path
            total_frames = get_video_frame_total(source_path)
            current_video_fps = util.detect_fps(source_path)
            cut_button = gr.Button(interactive=True)
            video_image, slider = display_video_frame(source_path, 1, total_frames)
    if added:
        _changed()
    return [_gallery(), video_image, slider, cut_button, *_report()]


def display_video_frame(filename: str, frame_num: int, total: int=0) -> Tuple[gr.Image, gr.Slider]:
    global current_video_fps

    current_frame = get_video_frame(filename, frame_num, exact=True)
    if current_video_fps == 0:
        current_video_fps = 1
    secs = (frame_num - 1) / current_video_fps
    minutes = secs / 60
    secs = secs % 60
    hours = minutes / 60
    minutes = minutes % 60
    milliseconds = (secs - int(secs)) * 1000
    timeinfo = f"{int(hours):0>2}:{int(minutes):0>2}:{int(secs):0>2}.{int(milliseconds):0>3}"
    if total > 0:
        return gr.Image(value=util.convert_to_gradio(current_frame), interactive=True), gr.Slider(info=timeinfo, minimum=1, maximum=total, interactive=True)
    return gr.Image(value=util.convert_to_gradio(current_frame), interactive=True), gr.Slider(info=timeinfo, interactive=True)


def on_face_selected(evt: gr.SelectData) -> None:
    global selected_face_index

    if evt is not None:
        selected_face_index = evt.index


def on_frame_num_fst_changed(frame_num: int):
    if _video_path is None:
        return gr.update()
    video_image, _ = display_video_frame(_video_path, frame_num, 0)
    return video_image


def on_cutfromframe_clicked(frame_num: int):
    if _video_path is None:
        return _outputs()
    found = extract_face_images(_video_path, (True, frame_num), 0.5)
    for face, image in found:
        _add(face, image)
    if found:
        _changed()
    return _outputs()


def on_remove_clicked():
    global selected_face_index

    if not 0 <= selected_face_index < len(entries):
        gr.Warning('Select a face in the gallery first')
        return _outputs()
    entries.pop(selected_face_index)
    _changed()
    return _outputs()


def on_clear_clicked():
    global _video_path, _saved_name, selected_face_index

    entries.clear()
    _seen_inputs.clear()
    _video_path = None
    _saved_name = None
    selected_face_index = -1
    _changed()
    return [_gallery(), None, None, gr.Image(value=None), gr.Slider(value=1, maximum=1, interactive=False),
            gr.Button(interactive=False), *_report()]


def on_update_clicked() -> Optional[str]:
    global _saved_name

    if len(entries) < 1:
        gr.Warning(f"No faces to create faceset from!")
        return None
    folder = tempfile.mkdtemp(prefix='faceset_')
    try:
        imgnames = []
        for index, e in enumerate(entries):
            filename = os.path.join(folder, f'{index}.png')
            cv2.imwrite(filename, e['image'])
            imgnames.append(filename)
        # a new name each time: an earlier faceset in the folder is not overwritten
        base = f"faceset_{time.strftime('%Y%m%d_%H%M%S')}"
        finalzip = os.path.join(roop.globals.output_path, base + '.fsz')
        n = 2
        while os.path.exists(finalzip):
            finalzip = os.path.join(roop.globals.output_path, f'{base}_{n}.fsz')
            n += 1
        util.zip(imgnames, finalzip)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    _saved_name = os.path.basename(finalzip)
    gr.Info(f'Saved {len(entries)} faces to {finalzip}')
    return finalzip


def on_check_clicked(progress=gr.Progress()):
    global _rows

    if not entries:
        gr.Warning('No faces to check')
        return _outputs()
    for k, e in enumerate(entries):
        if e['metrics'] is None:
            progress(k / len(entries), desc="Checking faces")
            e['metrics'] = faceset_check.photo_metrics(e['face'], e['image'])
    _rows = faceset_check.check([e['metrics'] for e in entries])
    return _outputs()


def on_remove_flagged_clicked(progress=gr.Progress()):
    global _rows

    if _rows is None:
        on_check_clicked(progress)
    if _rows is None:
        return _outputs()
    keep = [e for e, r in zip(entries, _rows) if not r['removable']]
    removed = len(entries) - len(keep)
    if removed == 0:
        gr.Info('Nothing flagged for removal')
        return _outputs()
    if not keep:
        gr.Warning('Every face is flagged: nothing removed. Check the photos by eye and remove them with "Remove selected".')
        return _outputs()
    entries[:] = keep
    _changed()
    # check the rest again: the average of the others changed
    _rows = faceset_check.check([e['metrics'] for e in entries]) if entries else None
    gr.Info(f'Removed {removed} flagged face(s)')
    return _outputs()
