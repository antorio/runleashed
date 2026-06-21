"""
RunLeashed visual identity for the Gradio UI.

The look is a reskin only: every existing Gradio component keeps its variable
name and event wiring, so all functionality is preserved. This module supplies
(1) a safe Gradio Theme object (constructor-only args, so it can never crash the
app at launch) and (2) a CSS layer that enforces the exact palette / card layout
from the Claude Design mockup. Bad CSS is harmless (it just doesn't apply), which
is why all the precise colours live here rather than in fragile theme vars.

Palette (from the mockup):
  bg          #f6f6f5    card        #ffffff / border #e6e5e2
  accent      #ef5b1f    accent soft #fff6f1 / #fff1ea / border #f3d8cb
  text        #1c1b1a #3a3833 #78756f / muted #a5a29b #bfbcb4
  input       bg #fcfcfb / border #e2dfd9
  ok badge    #1f8a5b on #eaf6ef      warn badge #c2820a on #fbf3e2
  fonts       Instrument Sans (body) · JetBrains Mono (data)
"""
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes


def runleashed_theme():
    """Safe base theme: only constructor args (hues, fonts, sizes). Exact
    brand colours are applied by RUNLEASHED_CSS so a bad value can't break launch."""
    return gr.themes.Base(
        primary_hue=colors.orange,
        secondary_hue=colors.orange,
        neutral_hue=colors.stone,
        font=[fonts.GoogleFont("Instrument Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[fonts.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        radius_size=sizes.radius_md,
        spacing_size=sizes.spacing_md,
        text_size=sizes.text_md,
    )


RUNLEASHED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- palette as variables ---- */
:root, .gradio-container {
  --rl-bg:#f6f6f5; --rl-card:#fff; --rl-border:#e6e5e2; --rl-line:#efeee9;
  --rl-accent:#ef5b1f; --rl-accent-hov:#e04e15;
  --rl-accent-soft:#fff6f1; --rl-accent-soft2:#fff1ea; --rl-accent-border:#f3d8cb;
  --rl-t1:#1c1b1a; --rl-t2:#3a3833; --rl-t3:#78756f; --rl-mut:#a5a29b; --rl-mut2:#bfbcb4;
  --rl-in-bg:#fcfcfb; --rl-in-border:#e2dfd9;
  --rl-mono:'JetBrains Mono',ui-monospace,monospace;
}

/* ---- canvas ---- */
gradio-app, .gradio-container {
  background:var(--rl-bg) !important;
  font-family:'Instrument Sans',-apple-system,system-ui,sans-serif !important;
  color:var(--rl-t1);
}
.gradio-container { max-width:1640px !important; margin:0 auto !important; }
footer { display:none !important; }

/* ---- top header (the markdown + version row) ---- */
#rl-header {
  background:var(--rl-card); border:1px solid var(--rl-border); border-radius:14px;
  padding:12px 20px !important; margin-bottom:14px; align-items:center; gap:18px;
}
#rl-header h1, #rl-header h2, #rl-header h3 { margin:0 !important; font-size:15px !important; font-weight:700 !important; letter-spacing:-.01em; }
#rl-header a { color:var(--rl-t1) !important; text-decoration:none !important; }
#versions, #versions * { font-family:var(--rl-mono) !important; font-size:11px !important; color:var(--rl-mut2) !important; }

/* ---- tabs: quiet nav with orange active underline ---- */
.tab-nav, div.tab-nav { border-bottom:1px solid var(--rl-border) !important; gap:2px; }
.tab-nav button {
  background:none !important; border:none !important; color:var(--rl-t3) !important;
  font-weight:500 !important; font-size:13.5px !important; padding:9px 15px !important;
  border-bottom:2px solid transparent !important; border-radius:0 !important;
}
.tab-nav button.selected {
  color:var(--rl-accent) !important; font-weight:600 !important;
  border-bottom:2px solid var(--rl-accent) !important;
}

/* ---- generic blocks → flat, the .rl-card wrappers carry the card look ---- */
.gradio-container .block { box-shadow:none !important; }

/* card wrapper (groups/columns tagged elem_classes="rl-card") */
.rl-card {
  background:var(--rl-card) !important; border:1px solid var(--rl-border) !important;
  border-radius:14px !important; padding:16px !important;
  box-shadow:none !important;
}
.rl-card .block { background:transparent !important; border:none !important; padding:0 !important; }
/* section heading used inside cards */
.rl-h { font-size:13px !important; font-weight:600 !important; color:var(--rl-t2) !important; }
.rl-h p { margin:0 !important; font-size:13px !important; font-weight:600 !important; }
.rl-sub p { margin:0 !important; font-size:11px !important; color:var(--rl-mut) !important; }

/* ---- labels / info ---- */
.gradio-container label span, .gradio-container .block-title, span[data-testid="block-info"] { color:var(--rl-t2) !important; }
.gradio-container .block-info, .gradio-container small { color:var(--rl-mut) !important; font-size:11px !important; }

/* ---- inputs / dropdowns / textboxes ---- */
.gradio-container input[type="text"], .gradio-container input[type="number"],
.gradio-container textarea, .gradio-container .wrap-inner, .gradio-container select {
  background:var(--rl-in-bg) !important; border:1px solid var(--rl-in-border) !important;
  border-radius:9px !important; color:var(--rl-t2) !important;
}
.gradio-container input[type="text"], .gradio-container input[type="number"] { font-family:var(--rl-mono) !important; font-size:12px !important; }
.gradio-container .wrap.svelte-1ybaih5 { border-radius:9px !important; }

/* ---- buttons ---- */
.gradio-container button.primary, .gradio-container .primary button,
.gradio-container button[variant="primary"] {
  background:var(--rl-accent) !important; border:1px solid var(--rl-accent) !important;
  color:#fff !important; font-weight:700 !important; border-radius:10px !important;
  box-shadow:0 2px 8px rgba(239,91,31,.26) !important;
}
.gradio-container button.primary:hover { background:var(--rl-accent-hov) !important; }
.gradio-container button.secondary, .gradio-container button[variant="secondary"] {
  background:var(--rl-card) !important; border:1px solid var(--rl-border) !important;
  color:var(--rl-t3) !important; font-weight:600 !important; border-radius:9px !important;
}
.gradio-container button.sm, .gradio-container button[size="sm"] { font-size:12px !important; }
/* accent-soft small buttons (we tag these elem_classes="rl-accent-btn") */
.rl-accent-btn button, button.rl-accent-btn {
  background:var(--rl-accent-soft) !important; border:1px solid var(--rl-accent-border) !important;
  color:var(--rl-accent) !important; font-weight:600 !important; box-shadow:none !important;
}

/* ---- sliders ---- */
.gradio-container input[type="range"] { accent-color:var(--rl-accent) !important; }
.gradio-container .slider_input_container .tab-like-container, .gradio-container .head .min, .gradio-container .head .max { color:var(--rl-mut2) !important; font-family:var(--rl-mono) !important; }
/* numeric value chip next to sliders */
.gradio-container .slider_input { color:var(--rl-accent) !important; font-family:var(--rl-mono) !important; }

/* ---- checkboxes ---- */
.gradio-container input[type="checkbox"] { accent-color:var(--rl-accent) !important; }

/* ---- accordions ---- */
.gradio-container .label-wrap, .gradio-container .accordion {
  border-radius:12px !important;
}
.gradio-container .label-wrap > span { font-weight:600 !important; color:var(--rl-t2) !important; }

/* ---- galleries (face thumbnails) ---- */
.rl-faces .grid-wrap, .rl-faces .grid-container { gap:8px !important; }
.rl-faces .thumbnail-item { border-radius:10px !important; border:1px solid var(--rl-in-border) !important; }
.rl-faces .thumbnail-item.selected { border:2px solid var(--rl-accent) !important; box-shadow:0 0 0 3px rgba(239,91,31,.18) !important; }

/* ---- preview image / video stage ---- */
#fixedheight { max-height:238.4px; overflow-y:auto !important; }
.image-container.svelte-1l6wqyv { height:100%; }
.rl-stage .block { border-radius:12px !important; }

/* ---- start / stop big buttons ---- */
.rl-start button { font-size:15px !important; padding:14px !important; }

/* ---- settings page section cards already use .rl-card ---- */
.rl-section-title p { margin:0 !important; font-size:13px !important; font-weight:600 !important; color:var(--rl-t1) !important; }

/* ---- 'live · no restart' style badge via markdown ---- */
.rl-badge-ok p { display:inline-block; margin:0 !important; font-family:var(--rl-mono); font-size:10px;
  color:#1f8a5b; background:#eaf6ef; padding:2px 7px; border-radius:6px; }
.rl-badge-warn p { display:inline-block; margin:0 !important; font-family:var(--rl-mono); font-size:10px;
  color:#c2820a; background:#fbf3e2; padding:2px 7px; border-radius:6px; }

/* keep info spans readable on the dark-on-light theme */
span { color: var(--block-info-text-color); }
"""
