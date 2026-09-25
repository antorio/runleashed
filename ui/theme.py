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
#app_header { padding: 12px 4px 2px; border: none !important; background: transparent !important;
  justify-content: space-between !important; align-items: baseline !important; flex-wrap: nowrap !important; }
#app_header h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -.01em; }
#versions { margin-left: auto !important; text-align: right !important; }
#versions, #versions * { font-family: var(--font-mono); font-size: 12.5px; color: var(--body-text-color-subdued); }

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

/* ---------- Video FPS: label + small input on one line, input close to label ---------- */
#fps_field label { display:flex !important; align-items:center !important; gap:14px !important; flex-wrap:nowrap !important; justify-content:flex-start !important; }
#fps_field label > span { white-space:nowrap !important; margin:0 !important; flex:0 0 auto !important; }
#fps_field input[type="number"] { max-width:110px !important; flex:0 0 auto !important; }

/* ---------- shrink the Source/Target dropzones + show only icon + 'Drop File Here' ----------
   The dropzone text ('Drop File Here', '- or -', 'Click to Upload') is partly bare
   text, so we can't hide just one piece by selector. Instead: zero the wrap font
   (hides ALL its text, svg unaffected) and re-add our own single line via ::after. */
#src_drop, #tgt_drop { min-height: 0 !important; }
/* Face Management: the status on each photo must stay readable */
#facemgr_gallery .caption-label { opacity: 1 !important; font-size: 12px !important; max-width: 94% !important; white-space: nowrap; text-overflow: ellipsis; }
#facemgr_gallery .thumbnail-lg:hover .caption-label { opacity: 1 !important; }
#facemgr_gallery .grid-wrap { min-height: 360px !important; max-height: 68vh !important; overflow-y: auto !important; }
#src_drop button[tabindex], #tgt_drop button[tabindex] { height: 72px !important; min-height: 0 !important; }
#src_drop .wrap, #tgt_drop .wrap { min-height: 0 !important; padding: 6px !important; font-size: 0 !important; }
#src_drop .wrap svg, #tgt_drop .wrap svg { width: 22px !important; height: 22px !important; }
#src_drop .wrap::after, #tgt_drop .wrap::after {
  content: "Drop files here or click"; display: block; margin-top: 4px;
  font-size: 13px; font-weight: 500; color: var(--body-text-color-subdued); }
#src_drop .file-preview, #tgt_drop .file-preview { min-height: 0 !important; }

/* ---------- Face Swap: lists, hints, run bar ---------- */
#src_gal .grid-container, #tgt_gal .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
#people_gal .grid-container { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; }
#src_gal .thumbnail-item, #tgt_gal .thumbnail-item, #people_gal .thumbnail-item { min-height: 0 !important; }
#src_gal .thumbnail-item.selected, #tgt_gal .thumbnail-item.selected, #people_gal .thumbnail-item.selected {
  outline: 3px solid var(--color-accent) !important; outline-offset: -3px; }
#src_gal .caption-label, #tgt_gal .caption-label { opacity: 1 !important; font-size: 11px !important; max-width: 94% !important;
  white-space: nowrap; text-overflow: ellipsis; }
.fs-step h3 { margin: 6px 0 0 !important; }
.fs-hint, .fs-hint * { font-size: 12.5px !important; color: var(--body-text-color-subdued) !important; }
#mode_radio .wrap { flex-direction: column !important; align-items: stretch !important; gap: 4px !important; }
#run_bar { align-items: center !important; }
#ready_line, #ready_line * { font-size: 13px !important; }
#status_line, #status_line * { font-size: 13px !important; }
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

/* ---------- galleries: fixed height, auto-scroll when boxes overflow ---------- */
.facegrid .grid-wrap, .facegrid .grid-container { overflow-y: auto !important; }
.facegrid { min-height: 0 !important; }

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
() => {}
"""
