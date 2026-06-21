"""
Gradio theme + CSS to match the redesigned (Gradio-native) look:
orange primary, Source Sans Pro / IBM Plex Mono, light surfaces,
corner block-labels, sticky center preview column, custom header.

Usage: import runleashed_theme + runleashed_css in ui/main.py (already wired
in the main.py shipped alongside this file).
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
    block_label_text_size="*text_sm",
    block_title_text_weight="600",
)

runleashed_css = """
    span {color: var(--block-info-text-color)}
    #fixedheight { max-height: 238.4px; overflow-y: auto !important; }
    .image-container.svelte-1l6wqyv { height: 100%; }

    /* ---- container width ---- */
    .gradio-container { max-width: 1500px !important; margin: 0 auto !important; }

    /* ---- custom header ---- */
    #app_header { padding: 14px 4px 2px; border: none !important; }
    #app_header h1 { margin: 0; font-size: 28px; font-weight: 700;
        letter-spacing: -.01em; color: var(--body-text-color); }
    #app_header h1 a { color: inherit; text-decoration: none; }
    #versions { display: flex; align-items: center; }
    #versions, #versions * { font-family: var(--font-mono);
        font-size: 12.5px; color: var(--body-text-color-subdued); }

    /* ---- corner block labels (mimics the mockup's bottom-right corner tabs) ---- */
    .block > .label-wrap,
    span.svelte-1gfkn6j,
    .gradio-container .block > label > span:first-child {
        /* default Gradio label is already top-left; nudge styling to match */
        font-size: 13px !important; color: var(--body-text-color) !important;
    }

    /* ---- sticky center preview column ---- */
    @media (min-width: 1024px) {
        #center_stage {
            position: sticky; top: 12px;
            align-self: flex-start;
            max-height: calc(100vh - 24px);
            overflow-y: auto;
        }
    }

    /* ---- FPS info as hover tooltip (set elem_id="fps_field" on the Number) ---- */
    #fps_field { position: relative; }
    #fps_field .info {
        position: absolute; z-index: 10; visibility: hidden; opacity: 0;
        transition: opacity .12s; background: #1f2937; color: #fff;
        padding: 4px 8px; border-radius: 6px; font-size: 12px; max-width: 220px;
        margin-top: 4px; pointer-events: none;
    }
    #fps_field:hover .info { visibility: visible; opacity: 1; }
"""
