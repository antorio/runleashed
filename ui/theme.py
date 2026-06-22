"""
Gradio theme + CSS for the redesigned roop-unleashed UI.
Orange primary, Source Sans Pro / IBM Plex Mono, light surfaces,
sleeker buttons, sticky center column, hidden component-type glyphs.

Tuned against Gradio 5.9.1. If a selector ever stops matching after a
Gradio upgrade, inspect the element and adjust the class here.
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
:root { --layout-gap: 14px; }
.gradio-container { max-width: 1500px !important; margin: 0 auto !important; }

/* ---------- header ---------- */
#app_header { padding: 12px 4px 2px; border: none !important; background: transparent !important; }
#app_header h1 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -.01em; }
#versions, #versions * { font-family: var(--font-mono); font-size: 12.5px; color: var(--body-text-color-subdued); }

/* ---------- tabs: plain text, orange underline ---------- */
button.selected { color: var(--primary-600) !important; }

/* ---------- hide the little component-type glyph next to block labels ---------- */
.block > label > span > svg.svelte-43sxxs,
span[data-testid="block-label"] svg,
.block-label svg,
label.svelte-1b6s6s svg { display: none !important; }

/* ---------- sleeker / lighter secondary buttons ---------- */
button.secondary {
  background: #fff !important; border: 1px solid var(--border-color-primary) !important;
  color: var(--body-text-color) !important; box-shadow: none !important;
  font-weight: 500 !important; font-size: 13px !important;
}
button.secondary:hover { background: #f9fafb !important; border-color: #d1d5db !important; }

/* ---------- galleries: fixed height, auto-scroll when boxes overflow ---------- */
.facegrid .grid-wrap, .facegrid .grid-container { overflow-y: auto !important; }
.facegrid { min-height: 0 !important; }

/* ---------- trim empty upload/file dropzones ---------- */
#filelist .file-preview, #filelist { min-height: 0 !important; }

/* ---------- sticky center preview column ---------- */
#center_stage { position: sticky !important; top: 8px !important; align-self: flex-start !important; }

/* ---------- Video FPS: info text becomes a hover tooltip ---------- */
#fps_field { position: relative; }
#fps_field .info, #fps_field p, #fps_field span.svelte-1gfkn6j {
  position: absolute; z-index: 20; top: 100%; left: 0;
  visibility: hidden; opacity: 0; transition: opacity .12s;
  background: #1f2937; color: #fff; padding: 4px 8px; border-radius: 6px;
  font-size: 12px; max-width: 230px; margin-top: 2px; pointer-events: none; white-space: normal;
}
#fps_field:hover .info, #fps_field:hover p { visibility: visible; opacity: 1; }

/* ---------- tidy spacing ---------- */
.block { border-radius: 8px; }
"""
