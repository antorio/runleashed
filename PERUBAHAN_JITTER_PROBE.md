# JITTER — alat diagnostik + arah baru

## Kenapa perbaikan sebelumnya tidak terasa
Perbaikan lalu (bbox-smoothing, RANSAC thr, alpha_min) menyasar noise KONTINU
sub-piksel. Tapi laporan user: jitter "berkedip antar frame di TEMPAT YANG SAMA"
= lompatan DISKRIT deterministik, terikat konten frame tertentu. Smoothing tidak
menolong untuk ini: lompatan besar dibaca stabilizer sebagai gerakan nyata
(alpha->1) sehingga smoothing justru lepas.

## Yang sudah dicek dan BUKAN dominan (terukur)
- Sanity gate toggle (68->5 vs kps): lompatan hanya ~0.5px, jarang trip. Bukan.
- Paste matrix M: diturunkan dari landmark yang SUDAH di-smooth. Konsisten. Benar.
- Smoothing tidak aktif? Sudah dicek: smoothing_on menyala benar untuk video.

## Alat baru: tools_jitter_probe.py (dijalankan user di video ASLI)
    python tools_jitter_probe.py clip.mp4 --frames 200 --hi-landmarker
Mengukur residual jitter (setelah membuang gerakan nyata via moving-average) per
tahap: bbox_raw, kps_raw, lm68_raw, M_raw (paste dari landmark mentah), M_smoothed
(setelah stabilizer). Menunjukkan PERSIS tahap mana yang goyang, alih-alih
menebak. Metrik divalidasi: gerak halus -> 0.12, jitter 1px -> 0.88.

Cara baca: tahap RAW yang tinggi DAN tetap tinggi di M_smoothed = sumber kedip.
Jika M_raw tinggi tapi M_smoothed rendah -> smoothing bekerja, sisa jitter ada di
tahap lain (enhancer / mask / ER / paste blend).

## Langkah berikutnya (butuh output probe dari user)
Angka dari probe akan mengarahkan: apakah perlu (a) smoothing lebih kuat di
motion_frac, (b) menstabilkan sumber landmark, atau (c) jitter sebenarnya dari
enhancer/mask yang belum kita sentuh sama sekali.
