"""
Gradio theme + CSS for the redesigned roop-unleashed UI.
Orange primary, Source Sans Pro / IBM Plex Mono, light surfaces,
sleeker buttons, sticky center column, bold section titles.

Tuned against Gradio 5.9.1. If a selector stops matching after a Gradio
upgrade, inspect the element and adjust the class here.
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
/* ---------- width: fill the page (was capped ~75%) ---------- */
.gradio-container { max-width: 1840px !important; width: 96% !important; margin: 0 auto !important; }

/* ---------- header ---------- */
#app_header { padding: 12px 4px 2px; border: none !important; background: transparent !important; }
#app_header h1 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -.01em; }
#versions, #versions * { font-family: var(--font-mono); font-size: 12.5px; color: var(--body-text-color-subdued); }

/* ---------- tabs: plain text, orange when active ---------- */
button.selected { color: var(--primary-600) !important; }

/* ---------- section (accordion) titles: BOLD, to match inner labels ---------- */
.label-wrap > span,
button.label-wrap span,
.gradio-accordion .label-wrap span { font-weight: 700 !important; font-size: 15px !important; color: var(--body-text-color) !important; }

/* ---------- hide the small component-type glyph next to block labels ---------- */
.block > label > span > svg.svelte-43sxxs,
span[data-testid="block-label"] svg,
.block-label svg { display: none !important; }

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

/* ---------- STICKY center preview column ----------
   Minimal + non-invasive: top-align the row so the column isn't stretched to
   full height, then stick it. No forced overflow on ancestors (that broke the
   layout / caused overlapping panels). If it still scrolls, see STICKY note
   in INTEGRATION.md. */
#swap_row { align-items: flex-start !important; }
#center_stage { position: sticky !important; top: 8px !important; align-self: flex-start !important; }

/* ---------- tidy spacing ---------- */
.block { border-radius: 8px; }
"""

# Injected into <head>. Native position:sticky silently fails when an ancestor
# establishes a containing block via transform / filter / perspective / contain
# (Gradio uses transforms for tab transitions). This neutralises those on the
# ancestors of #center_stage so the CSS sticky actually engages. It also retries
# because Gradio mounts tab content dynamically after first paint.
runleashed_head = """
<script>
(function () {
  function clearBlockers() {
    var el = document.querySelector('#center_stage');
    if (!el) return false;
    var n = el.parentElement;
    while (n && n !== document.documentElement) {
      var s = getComputedStyle(n);
      if (s.transform !== 'none' || s.filter !== 'none' ||
          s.perspective !== 'none' || /(paint|layout|content|strict)/.test(s.contain)) {
        n.style.setProperty('transform', 'none', 'important');
        n.style.setProperty('filter', 'none', 'important');
        n.style.setProperty('perspective', 'none', 'important');
        n.style.setProperty('contain', 'none', 'important');
      }
      n = n.parentElement;
    }
    return true;
  }
  var tries = 0;
  var iv = setInterval(function () {
    if (clearBlockers() || ++tries > 60) clearInterval(iv);
  }, 250);
  window.addEventListener('load', clearBlockers);
  document.addEventListener('click', function () { setTimeout(clearBlockers, 50); });
})();
</script>
"""

