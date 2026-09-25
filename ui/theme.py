"""
Gradio theme + CSS + load-JS for the redesigned roop-unleashed UI.
Tuned against Gradio 5.9.1.

Exports:
  runleashed_theme  -> pass as theme=
  runleashed_css    -> pass as css=
  runleashed_js     -> pass as js=   (runs on app load; this is where the sticky
                                       center column lives — head=<script> does NOT
                                       reliably execute in Gradio, js= does)
"""

import gradio as gr

runleashed_theme = gr.themes.Default(
    primary_hue="orange",
    secondary_hue="orange",
    neutral_hue="gray",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    button_secondary_background_fill="white",
    button_secondary_background_fill_hover="#f9fafb",
    button_secondary_border_color="*neutral_200",
    button_secondary_text_color="*neutral_700",
    block_title_text_weight="600",
    block_label_text_weight="600",
)

runleashed_css = """
/* ---------- width: fill the page ---------- */
.gradio-container { max-width: 1840px !important; width: 96% !important; margin: 0 auto !important; }

/* ---------- header: title left, versions hard right ---------- */
#app_header { padding: 4px 4px 0; border: none !important; background: transparent !important;
  justify-content: space-between !important; align-items: baseline !important; flex-wrap: nowrap !important; }
#app_header h1 { margin: 0; font-size: 19px; font-weight: 700; letter-spacing: -.01em; }
#versions { margin-left: auto !important; text-align: right !important; }
#versions, #versions * { font-family: var(--font-mono); font-size: 12.5px; color: var(--body-text-color-subdued); }

/* ---------- less chrome above the tabs (1080p screens) ---------- */
.gradio-container { padding-top: 6px !important; padding-bottom: 6px !important; }
#app_header { margin-bottom: -10px !important; }
.tabs { gap: 6px !important; }
.tabitem { padding-top: 4px !important; padding-bottom: 4px !important; }

/* ---------- tabs: bold, orange when active ---------- */
.tab-nav button, button[role="tab"], .tabs > .tab-nav button { font-weight: 700 !important; }
button.selected { color: var(--primary-600) !important; }

/* ---------- accordion / section titles: bold ---------- */
.label-wrap > span, button.label-wrap span, .gradio-accordion .label-wrap span {
  font-weight: 700 !important; font-size: 15px !important; color: var(--body-text-color) !important; }

/* ---------- hide the component-type glyph next to block labels ---------- */
.block > label > span > svg.svelte-43sxxs, span[data-testid="block-label"] svg, .block-label svg { display: none !important; }

/* ---------- sleeker secondary buttons ---------- */
button.secondary { background:#fff !important; border:1px solid var(--border-color-primary) !important;
  color: var(--body-text-color) !important; box-shadow:none !important; font-weight:500 !important; font-size:13px !important; }
button.secondary:hover { background:#f9fafb !important; border-color:#d1d5db !important; }

/* ---------- shrink the Source/Target dropzones + show only icon + 'Drop File Here' ----------
   The dropzone text ('Drop File Here', '- or -', 'Click to Upload') is partly bare
   text, so we can't hide just one piece by selector. Instead: zero the wrap font
   (hides ALL its text, svg unaffected) and re-add our own single line via ::after. */
#src_drop, #tgt_files { min-height: 0 !important; }
/* Face Management: the status on each photo must stay readable */
#facemgr_gallery .caption-label { opacity: 1 !important; font-size: 12px !important; max-width: 94% !important; white-space: nowrap; text-overflow: ellipsis; }
#facemgr_gallery .thumbnail-lg:hover .caption-label { opacity: 1 !important; }
#facemgr_gallery .grid-wrap { min-height: 360px !important; max-height: 68vh !important; overflow-y: auto !important; }
#src_drop > button[tabindex], #tgt_files > button[tabindex] { height: 54px !important; min-height: 0 !important; }
#src_drop > button .wrap, #tgt_files > button .wrap { min-height: 0 !important; padding: 6px !important; font-size: 0 !important; }
#src_drop > button .wrap svg, #tgt_files > button .wrap svg { width: 18px !important; height: 18px !important; }
#src_drop > button .wrap::after, #tgt_files > button .wrap::after {
  display: block; margin-top: 2px; font-size: 12.5px; font-weight: 500; color: var(--body-text-color-subdued); }
#src_drop > button .wrap::after { content: "Drop photos or a faceset (.fsz), or click"; }
#tgt_files > button .wrap::after { content: "Drop images or videos, or click"; }
#src_drop .file-preview { min-height: 0 !important; }

/* ---------- Face Swap: lists, hints, run bar ---------- */
#src_gal .grid-container { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; }
#people_gal .grid-container { grid-template-columns: repeat(5, minmax(0, 1fr)) !important; }
#src_gal .thumbnail-item, #people_gal .thumbnail-item { min-height: 0 !important; position: relative !important; }
#src_gal .thumbnail-item.selected, #people_gal .thumbnail-item.selected {
  outline: 3px solid var(--color-accent) !important; outline-offset: -3px; }
.fs-line, .fs-line * { font-size: 12.5px !important; line-height: 1.35 !important; }
.fs-line p { margin: 0 !important; }
#run_bar { align-items: center !important; }
#ready_line, #ready_line * { font-size: 13px !important; }
#ready_line p { margin: 0 0 2px !important; }
#status_line, #status_line * { font-size: 13px !important; }
#range_line, #range_line * { color: var(--body-text-color-subdued) !important; }

/* ---------- Face Swap: compact spacing (1920x1080 target) ---------- */
#fs_left, #center_stage, #fs_settings { gap: 8px !important; }
#swap_row { gap: 12px !important; }
#fs_settings .gradio-accordion > div, #fs_settings .form { gap: 8px !important; }
.fs-buttons { gap: 6px !important; }
.fs-path { gap: 6px !important; }
#people_col { gap: 6px !important; }
#view_bar, #frame_bar { align-items: center !important; gap: 6px !important; }
#view_radio .wrap { gap: 4px !important; }
#view_radio label { padding: 4px 10px !important; }
#range_bar { align-items: center !important; gap: 6px !important; flex-wrap: nowrap !important; }
#range_bar #range_line { flex: 1 1 auto !important; min-width: 0 !important; }
.fs-checks { gap: 6px !important; }
/* an empty gallery draws a 236px placeholder: keep the lists their own size */
#src_gal .empty { min-height: 0 !important; height: 146px !important; }
#people_gal .empty { min-height: 0 !important; height: 90px !important; }
/* the × on each source / picked person (added by runleashed_js) */
.fs-hidden { display: none !important; }
.fs-x { position: absolute; top: 3px; right: 3px; z-index: 3; width: 20px; height: 20px; border-radius: 50%;
  background: rgba(0, 0, 0, .6); color: #fff; font-size: 15px; line-height: 19px; text-align: center;
  cursor: pointer; opacity: .8; user-select: none; }
.fs-x:hover { opacity: 1; background: #dc2626; }
/* "One source per face": the order, small in a corner (no captions over the faces) */
#src_gal.fs-numbered .grid-container { counter-reset: fs-src; }
#src_gal.fs-numbered .thumbnail-item { counter-increment: fs-src; }
#src_gal.fs-numbered .thumbnail-item::before { content: counter(fs-src); position: absolute; top: 3px; left: 3px; z-index: 3;
  min-width: 18px; height: 18px; padding: 0 4px; border-radius: 9px; background: var(--color-accent); color: #fff;
  font-size: 11px; font-weight: 700; line-height: 18px; text-align: center; }
/* target list: the file shown in the preview is marked (runleashed_js) */
#tgt_files tr.file { cursor: pointer; }
#tgt_files tr.file.fs-shown { background: var(--color-accent-soft) !important; box-shadow: inset 3px 0 0 var(--color-accent); }
#tgt_files tr.file.fs-shown .stem { font-weight: 600; }
.fs-checks label { white-space: nowrap !important; }
/* Gradio dims a Markdown to 20% while any event writing it runs; the readiness
   line is refreshed by every preview, so it was faded most of the time */
#ready_line .pending, #status_line .pending, #fs_left .pending { opacity: 1 !important; }
#ready_line code, #status_line code { white-space: normal; word-break: break-all; }
#mask_editor button[aria-label="Clear canvas"] { display: none !important; }
#results .label-clear-button { display: none !important; }

/* ---------- Eyes / Mouth / Brows forced onto a single row ----------
   Gradio groups the 3 adjacent checkboxes into a .form wrapper that wraps at 2.
   Force that .form (and its children) to a single nowrap flex row. */
#expr_checks .form, #expr_checks > div {
  display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important; gap: 4px !important; }
#expr_checks .form > *, #expr_checks > div > * { flex: 1 1 0 !important; min-width: 0 !important; }
#expr_checks label { white-space: nowrap !important; }

/* ---------- centre column stays in view while the settings scroll ----------
   Pure CSS sticky (the container's overflow:hidden blocked it before). Only
   where the three columns fit side by side; the column scrolls on its own
   when it is taller than the window. */
.gradio-container { overflow: unset !important; }
#swap_row { align-items: flex-start !important; }
@media (min-width: 1200px) {
  #swap_row { flex-wrap: nowrap !important; }
  #center_stage { position: sticky; top: 8px; align-self: flex-start; max-height: calc(100vh - 16px);
                  overflow-y: auto; flex-wrap: nowrap !important; }
  #center_stage > * { flex-shrink: 0 !important; }
}

/* ---------- tidy spacing ---------- */
.block { border-radius: 8px; }

/* ---------- clean footer: hide Gradio default (Use via API · Built with Gradio ·
   Settings + icons), show a single tidy 'Use via API' line ---------- */
footer { display: none !important; }
.rl-footer {
    text-align: center; padding: 20px; font-size: 11px;
    color: var(--body-text-color-subdued); font-family: var(--font-mono);
}
"""

# Runs on app load (gr.Blocks(js=...)). The centre column used to follow the
# scroll with a requestAnimationFrame loop; CSS sticky does it now.
runleashed_js = """
() => {
    // Face Swap: the arrow keys step the preview frame (like its ◀ ▶ buttons)
    // unless the focus is in a text field or a slider being dragged.
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
        const el = document.activeElement;
        // taken over as well: the frame slider (its own arrow stepping does not
        // refresh the preview) and the View radios (a click leaves the focus
        // there, and arrows would switch the view instead of the frame)
        const ours = el && el.tagName === 'INPUT' && el.closest &&
            ((el.type === 'range' && el.closest('#frame_slider')) || (el.type === 'radio' && el.closest('#view_radio')));
        if (!ours && el && (el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName))) return;
        const btn = document.getElementById(e.key === 'ArrowLeft' ? 'frame_prev' : 'frame_next');
        if (!btn || !btn.offsetParent) return;          // other tab, or not a video
        e.preventDefault();
        btn.click();
    });

    // × on each source and picked-person thumbnail. A click on it sends
    // "index:time" through a hidden textbox (its .input event runs the
    // removal on the server) and never reaches the thumbnail (no select).
    const X_TARGETS = [['src_gal', 'src_x'], ['people_gal', 'people_x']];
    document.addEventListener('click', (e) => {
        const x = e.target && e.target.closest ? e.target.closest('.fs-x') : null;
        if (!x) return;
        e.preventDefault();
        e.stopPropagation();
        const gallery = document.getElementById(x.dataset.gallery);
        const item = x.closest('.thumbnail-item');
        if (!gallery || !item) return;
        const index = [...gallery.querySelectorAll('.thumbnail-item')].indexOf(item);
        const box = document.querySelector('#' + x.dataset.box + ' textarea, #' + x.dataset.box + ' input');
        if (index < 0 || !box) return;
        box.value = index + ':' + Date.now();
        box.dispatchEvent(new Event('input', {bubbles: true}));
    }, true);

    const tidy = () => {
        for (const [gid, bid] of X_TARGETS) {
            const gallery = document.getElementById(gid);
            if (!gallery) continue;
            gallery.querySelectorAll('.thumbnail-item').forEach((item) => {
                if (item.querySelector(':scope > .fs-x')) return;
                const x = document.createElement('span');
                x.className = 'fs-x';
                x.textContent = '×';
                x.title = 'Remove';
                x.dataset.gallery = gid;
                x.dataset.box = bid;
                item.appendChild(x);
            });
        }
        // order numbers only when the order matters
        const src = document.getElementById('src_gal');
        const mode = document.querySelector('#mode_dd input');
        if (src && mode) src.classList.toggle('fs-numbered', (mode.value || '').startsWith('One source'));
        // mark the target file the preview shows (its label starts with the name)
        const label = document.querySelector('#preview_img [data-testid="block-label"]');
        const shown = label ? label.innerText.split(' · ')[0].split(' — ')[0].trim() : '';
        document.querySelectorAll('#tgt_files tr.file').forEach((row) => {
            const cell = row.querySelector('td.filename');
            row.classList.toggle('fs-shown', !!cell && cell.getAttribute('aria-label') === shown);
        });
    };
    let queued = false;
    new MutationObserver(() => {
        if (queued) return;
        queued = true;
        setTimeout(() => { queued = false; tidy(); }, 60);
    }).observe(document.body, {childList: true, subtree: true, characterData: true});
    setInterval(tidy, 1000);        // the dropdown's value is no DOM mutation
}
"""
