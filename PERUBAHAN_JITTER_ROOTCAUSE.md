# PERBAIKAN JITTER — 3 akar, diukur satu per satu

Konteks: user SUDAH memakai 2dfan4 (use_hi_landmarker ON), smoothing strength
0.95 (slider di grup Alignment & stabilization), deadzone 0.012, ER smoothing 0.
Jitter tetap ada, terutama di pose tidak lurus depan.

## Akar 1 (DOMINAN, khusus 2dfan4) — bbox goyang MEMPERKUAT jitter landmark
`_Fan2d4.detect` membangun crop 256x256 LANGSUNG dari bbox mentah detektor:
    scale = 195 / max(bbox_w, bbox_h)
    tx,ty = (256 - (bbox_min+bbox_max) * scale) * 0.5
bbox SCRFD bergoyang tiap frame -> crop yang masuk 2dfan4 ikut bergoyang. CNN
landmark TIDAK ekuivarian sempurna terhadap translasi/skala input: framing yang
sedikit berbeda menghasilkan prediksi yang bergeser non-linear. Jadi goyangan
bbox DIPERKUAT jadi goyangan landmark SEBELUM smoothing sempat bekerja. Dan bbox
paling tidak stabil justru di pose miring -> persis pola yang dilaporkan.

FIX: urutan diubah menjadi  bbox dihaluskan -> landmarker jalan -> landmark
dihaluskan. `LandmarkStabilizer.stabilize(faces, refine_fn=...)` kini memanggil
refine_fn di TENGAH: bbox di-smooth dulu (bobot adaptif yang sama), 2dfan4 jalan
di crop yang stabil, baru landmark di-smooth seperti sebelumnya.
Terukur: jitter bbox yang diumpankan ke landmarker 2.771px -> 0.025px (-99%),
sementara bbox tetap mengikuti gerak (lag 3.83px pada pan 6px/frame).

## Akar 2 — "RANSAC robust" tidak menolak outlier sama sekali
`ransacReprojThreshold=100` pada crop 128-512 membuat SEMUA titik selalu inlier
-> RANSAC merosot jadi least-squares, menolak NOL outlier. Satu landmark meleset
(sering terjadi di pose miring: oklusi/foreshortening) menyeret seluruh alignment.
Terukur (crop 256, satu titik meleset 40px): 6.93px -> 1.94px (3.6x lebih baik).
FIX: thr = max(4, image_size * 0.05) + maxIters/confidence/refineIters eksplisit.
Diverifikasi di crop 128/256/384/512 (perbaikan konsisten ~3.5x), 2000 fit acak
semua finite & skala waras, input degenerate aman, minimal 4 inlier (tak pernah
kolaps ke fit 2-titik). Sweep 0.03-0.39 -> 0.05 titik optimal.

## Akar 3 — slider Smoothing strength MATI di atas ~0.92
`alpha_min = clip(1-strength, 0.08, 1.0)`: strength 0.92 dan 0.98 sama-sama
menghasilkan 0.08, jadi menaikkan slider ke 0.95 (yang user lakukan) TIDAK
berefek apa pun. FIX: batas bawah 0.08 -> 0.02 (0.95 -> 0.05, 0.98 -> 0.02).

## Sudah diperiksa, TERNYATA BUKAN penyebab
- RANSAC non-deterministik? BUKAN (200 panggilan identik, spread 0.0).
- Beda metode fit RANSAC vs similarity-LS? BUKAN (selisih M 2e-5 px).
- Sanity gate berganti jalur 68to5 <-> kps? Kecil (~0.45px), bukan dominan.
- albumentations? Tidak menyentuh jalur alignment sama sekali.

## Knob yang SENGAJA tidak diubah diam-diam
`motion_frac` (0.05) adalah pengikat sebenarnya untuk sisa jitter saat kepala
diam (bukan alpha_min). 0.08 memberi jitter -21% tapi lag naik 5.10 -> 6.43px.
Kalau setelah uji masih kurang tenang dan rela sedikit lag, itu angka berikutnya.

## Validasi
Urutan refine (bbox halus terlihat oleh landmarker), refine_fn dipanggil tepat
sekali (termasuk saat 0 wajah), smoothing landmark tanpa regresi (0.209px),
kompatibel tanpa refine_fn, multi-face + pan cepat, integrasi process_face x30
(M rotation std 0.018deg, scale std 0.0012), py_compile, build Gradio 5.9.1.
