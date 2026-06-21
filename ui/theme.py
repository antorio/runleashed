"""
Gradio theme + CSS to match the redesigned (Gradio-native) look:
orange primary, Source Sans Pro / IBM Plex Mono, light surfaces.

Usage in ui/main.py:

    from ui.theme import runleashed_theme, runleashed_css
    ...
    with gr.Blocks(title=..., theme=runleashed_theme, css=runleashed_css,
                   delete_cache=(60, 86400)) as ui:
        ...
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
    # primary action button = solid orange / white text
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    block_label_text_size="*text_sm",
    block_title_text_weight="600",
)

runleashed_css = """
    span {color: var(--block-info-text-color)}
    #fixedheight {
        max-height: 238.4px;
        overflow-y: auto !important;
    }
    .image-container.svelte-1l6wqyv {height: 100%}

    /* keep the three Face-Swap columns from collapsing too early */
    .gradio-container {max-width: 1500px !important; margin: 0 auto !important;}

    /* OPTIONAL: turn a Number's info text into a hover-only tooltip.
       Give the component elem_id="fps_field" to enable, e.g.
           forced_fps = gr.Number(..., elem_id="fps_field")  */
    #fps_field .info {
        position: absolute; z-index: 10; visibility: hidden; opacity: 0;
        transition: opacity .12s; background: #1f2937; color: #fff;
        padding: 4px 8px; border-radius: 6px; font-size: 12px; max-width: 220px;
        margin-top: 4px;
    }
    #fps_field:hover .info { visibility: visible; opacity: 1; }
"""
