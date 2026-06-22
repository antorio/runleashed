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
#src_files, #dst_files { min-height: 0 !important; }
#src_files .wrap, #dst_files .wrap,
#src_files [data-testid="upload"] .wrap, #dst_files [data-testid="upload"] .wrap {
  min-height: 78px !important; padding: 10px !important; font-size: 0 !important; }
#src_files .wrap svg, #dst_files .wrap svg { width: 24px !important; height: 24px !important; }
#src_files .wrap::after, #dst_files .wrap::after {
  content: "Drop File Here"; display: block; margin-top: 6px;
  font-size: 14px; font-weight: 500; color: var(--body-text-color-subdued); }
#src_files .file-preview, #dst_files .file-preview { min-height: 0 !important; }

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

/* ---------- center column: stacking context for the JS sticky ---------- */
#swap_row { align-items: flex-start !important; }
#center_stage { position: relative !important; z-index: 5; will-change: transform; }

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

# Runs on app load (gr.Blocks(js=...)). Pure-JS "fake sticky": a permanent rAF
# loop translates the center column to follow scroll, clamped within its row.
# js= is used (not head=<script>) because head scripts are injected as inert text
# in Gradio and never execute, whereas js= is guaranteed to run on load.
runleashed_js = """
async () => {
  const TOP = 8;
  let last = null;
  function tick() {
    const col = document.querySelector('#center_stage');
    const row = document.querySelector('#swap_row');
    if (col && row && col.offsetWidth !== 0) {
      const rowTop = row.getBoundingClientRect().top;
      let shift = TOP - rowTop;
      if (shift < 0) shift = 0;
      let maxShift = row.offsetHeight - col.offsetHeight;
      if (maxShift < 0) maxShift = 0;
      if (shift > maxShift) shift = maxShift;
      if (shift !== last) { col.style.transform = 'translateY(' + shift + 'px)'; last = shift; }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
"""
