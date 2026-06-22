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

/* ---------- center preview column ----------
   The "stick to scroll" behaviour is done in JS (runleashed_head) via translateY,
   because CSS position:sticky is unreliable inside Gradio's tab transforms.
   Here we just top-align the row and give the column a stacking context. */
#swap_row { align-items: flex-start !important; }
#center_stage { position: relative !important; z-index: 5; will-change: transform; }

/* ---------- tidy spacing ---------- */
.block { border-radius: 8px; }
"""

# Injected into <head>. We do NOT use CSS position:sticky for the center column
# — it silently fails in Gradio (ancestor transforms create a containing block).
# Instead this is a pure-JS "fake sticky": translateY the column to follow scroll,
# clamped within its row. No dependency on overflow / transform / sticky support.
runleashed_head = """
<script>
(function () {
  var TOP = 8;            // gap from viewport top (px)
  function tick() {
    var col = document.querySelector('#center_stage');
    var row = document.querySelector('#swap_row');
    if (!col || !row || col.offsetWidth === 0) return;     // hidden tab -> skip
    col.style.transform = 'none';                          // reset to measure
    var rowTop = row.getBoundingClientRect().top;
    var shift = TOP - rowTop;                              // move down to reach the top gap
    if (shift < 0) shift = 0;
    var maxShift = row.offsetHeight - col.offsetHeight;    // don't pass the row bottom
    if (maxShift < 0) maxShift = 0;
    if (shift > maxShift) shift = maxShift;
    col.style.transform = 'translateY(' + shift + 'px)';
  }
  var scheduled = false;
  function onScroll() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(function () { scheduled = false; tick(); });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  // Gradio mounts tab content after first paint + on tab switch -> keep re-binding.
  var iv = setInterval(tick, 400);
  setTimeout(function () { clearInterval(iv); }, 30000);   // stop polling after warm-up
  document.addEventListener('click', function () { setTimeout(tick, 60); });
  window.addEventListener('load', tick);
})();
</script>
"""

