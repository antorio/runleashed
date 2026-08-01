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

## Hotfix probe (setelah run pertama user)
1. BUG saya: residual_jitter di file berbeda dari versi yang saya uji inline
   (salah indexing utk array 2D spt bbox) -> crash IndexError. Diperbaiki; kini
   fungsi YANG DIKIRIM diuji langsung pada 3 bentuk data (bbox/kps/lm68):
   smooth ~0.12, jitter 1px ~0.89. Pelajaran: tested code == shipped code.
2. Run pertama user TIDAK memakai 2dfan4 (log: "2dfan4.onnx not found -> using
   buffalo_l") karena probe tidak menjalankan pre_check -> hasil tidak mewakili
   config render. Probe kini auto-fetch 2dfan4.onnx dari repo HF user bila
   hilang, dan memakai CUDAExecutionProvider bila tersedia (run CPU tetap valid,
   hanya lambat).
3. Catatan: buffalo_l.zip masih diunduh dari GitHub insightface (di luar 15 model
   HF kita) -- kandidat migrasi HF berikutnya bila unduhan itu pernah stuck.
Jalankan ulang:
   NO_ALBUMENTATIONS_UPDATE=1 python tools_jitter_probe.py clip.mp4 --frames 200 --hi-landmarker
