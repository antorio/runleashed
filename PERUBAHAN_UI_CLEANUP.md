# Cabut colour-match + rapikan UI

## 1. "Match LivePortrait colour to the swap" DICABUT TOTAL
User: tak terlihat pengaruhnya, dan warna "mentah" di tampak samping ternyata
terjadi juga TANPA ER -- jadi itu masalah swap di profil, bukan khusus ER.
Dihapus sampai bersih, tidak menyisakan komputasi apa pun:
  - call site di Expression_LivePortrait.py
  - fungsi `match_color()` di live_portrait_util.py
  - globals expression_color_match + expression_color_match_strength
  - checkbox UI + wiring-nya
Diverifikasi: nol referensi `match_color`/`expression_color_match` di seluruh
tree; audit ulang -> fungsi mati 0, globals mati 0.
CATATAN: `_match_color_masked` di ProcessMgr BUKAN milik fitur ini -- itu milik
toggle "Color transfer (LAB)" yang sudah ada sejak awal dan dijaga
`if roop.globals.use_color_transfer`, jadi nol beban saat mati.

## 2. Detection confidence + size: DIPERTAHANKAN (user masih meneliti)

## 3-7. UI dirapikan
Alignment & stabilization:
  - Detection confidence : info dihapus
  - Detection size       : info dihapus
  - Landmark alignment   : -> "Off = align from the detector's 5 keypoints"
  - 2dfan4               : -> "Only used while landmark alignment is on"
  - Sanity gate          : -> "Falls back to detector keypoints when the 68pt landmarks look broken"
  - Gate threshold       : -> "fraction of face size. Lower = fires more often"
  - Landmark smoothing   : -> "Reduces per-frame jitter"
  - Smoothing strength   : info dihapus
  - "Still-face dead-zone" -> RENAME "Still-face threshold freezes",
                              info "Higher = calmer, 0 = off"
  - Occlusion mask after enhancer: DEFAULT -> ON (globals mask_after_enhancer=True)
Expression Restorer:
  - Expression smoothing : -> "0 = off"
  - Expression power     : -> "1.0 = target amount"
  - Pose lock (adaptive) : info dihapus
  - komentar blok usang di dalam grup dihapus

## Validasi
py_compile seluruh tree; build Gradio 5.9.1 (3 tab); 7 pemeriksaan otomatis untuk
tiap permintaan; audit kode mati (fungsi 0, globals 0); live_portrait_util
diperiksa tidak terpotong setelah pemotongan fungsi.

## Expression Restorer: power default 1.0, strength 0-300
1. `expression_power` default 2.0 -> **1.0**, teks info dihapus.
   2.0 berarti delta ekspresi dikali dua -- ikut menggandakan noise per-frame
   motion extractor, sumber getaran di dalam wajah.
2. "Expression smoothing (video wobble)" -> "Expression smoothing", info dihapus.
3. Slider Strength 0-100 -> **0-300**.

### Bug yang ikut ditemukan & diperbaiki
Mapping lama: `np.interp(value, [0,100], [0.0,1.0])`. `np.interp` MENG-CLAMP di
ujung rentang, jadi melebarkan slider ke 300 saja TIDAK akan berefek -- semua
nilai di atas 100 tetap dipetakan ke 1.0 dan slider diam-diam mati di separuh
atasnya. Diganti pembagian langsung `value/100.0`.
Terverifikasi: 0->0.00, 50->0.50, 100->1.00, 150->1.50, 200->2.00, 300->3.00
(sebelumnya 150/200/300 semuanya 1.00).

### Catatan pemakaian
factor dan power adalah pengali yang sama: amt = factor * power. Dengan power
kini tetap 1.0, Strength menjadi SATU-SATUNYA knob penguatan:
  Strength 100 = ekspresi target apa adanya (tanpa penguatan noise)
  Strength >100 = ekstrapolasi melewati target -> ekspresi lebih tegas, TAPI
  noise per-frame ikut dikali proporsional (inilah sumber jitter di dalam wajah).
