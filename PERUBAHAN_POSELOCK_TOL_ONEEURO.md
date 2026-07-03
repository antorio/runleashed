# PERUBAHAN — POSE LOCK TOLERANSI + ONE-EURO M + DIET MENU (batch 2)

## 1. Pose lock ADAPTIF dengan toleransi (mengganti 3 checkbox binary)
`lock_pose` kini menahan hanya KELEBIHAN komponen global delta ekspresi:
- Translation: selalu full lock (pergeseran centroid netto bukan ekspresi asli).
- Scale: deviasi sebaran ±`expression_pose_lock_scale_tol` (default 0.04) lolos
  sebagai ekspresi asli (rahang buka); kelebihannya di-clamp ke tepi band.
- Rotation: sudut Kabsch ≤ `expression_pose_lock_rot_tol` (default 2.0°) lolos;
  hanya kelebihan sudutnya yang dirotasi balik (sumbu sama, Rodrigues).
tol=0 mereproduksi full-lock lama EKSAK (diverifikasi max|diff|=3e-8); tol besar
= efektif tanpa scale/rot lock. Globals lama `expression_pose_lock_scale/rotation`
DIHAPUS. `[expr-delta]` kini mencetak `sdev=..(tol ±..)` dan `kabsch_rot=..(tol ..)`
supaya toleransi bisa di-tuning langsung dari log (naikkan tol bila nilai normal
sering terpotong; turunkan bila drift lolos).
Validasi 5 kasus: ekuivalensi tol=0, pass-through dalam band, clamp presisi ke
tepi band (7°→2.00°; dev 1.15→1.040), tol besar == translation-only, NaN-safe.

CATATAN JUJUR: implementasi lock_pose toleransi ditemukan SUDAH ada di working
copy sesi ini tanpa jejak edit dari asisten (anomali sesi/tool). Kode diaudit
baris-per-baris + divalidasi numerik penuh sebelum diadopsi; pemanggilnya yang
masih memakai signature lama (akan TypeError) diperbaiki oleh asisten.

## 2. One-Euro smoothing untuk MATRIKS ALIGNMENT M (video)
`estimate_norm` mengubah sisa noise landmark menjadi flicker rotasi/skala/posisi
warp (shimmer wajah saat kepala diam). Baru: `MatrixStabilizer` di
`face_stabilizer.py` — dekomposisi M → (θ, log s, tx/crop, ty/crop), tiap kanal
difilter One-Euro per-track wajah (match bbox centre), M dirakit ulang, crop
di-warp ulang dengan M halus. `target_face.matrix` = M halus → seluruh downstream
(hull matte, LP full-pipeline, paste-back) konsisten otomatis.
- TANPA opsi baru: nebeng toggle + slider "Landmark + warp smoothing" yang ada;
  strength → min_cutoff log-scale (0→4Hz, 0.7→0.42Hz, 1→0.16Hz), beta=10.
- Aktif hanya video (atau force_landmark_smoothing); jalur rotate-retry dilewati
  (ruang koordinat beda); frame pertama passthrough eksak; unwrap θ anti-glitch ±π.
- Validasi: peredaman jitter statis 3.3–4.4×, lag pan 3px/f = 2.2px, whip-pan
  10px/f = 3.2px, wrap ±π mulus, multi-face track independen.

## 3. DIET MENU — 8 kontrol dihapus, 0 ditambah (semua global tetap ada di
`roop/globals.py` untuk tuning file-level)
- "Use landmark alignment (68pt+RANSAC)" → dihapus (settled ON).
- "Landmark sanity gate" checkbox → dilebur ke slider threshold-nya (gate selalu
  siaga; threshold tinggi = efektif off).
- "Landmark dead-zone" slider → dihapus (default 0.006 settled).
- "Blend border" slider → dihapus (0.2 settled).
- "Pose lock translation/scale/rotation" 3 checkbox → 1 checkbox "Pose lock
  (adaptive)".
- "Pose gate soft/hard" 2 slider → dihapus (45/65° settled; checkbox tetap).
Label "Landmark smoothing (video)" → "Landmark + warp smoothing (video)".

## Validasi build
Math lock_pose (5 kasus) + MatrixStabilizer (6 kasus) lolos; `py_compile` seluruh
tree; build-test Gradio 5.9.1 (3 tab, wiring bersih, tak ada referensi komponen
terhapus); smoke-test integrasi process_face 12 frame dengan M-smoothing aktif +
jalur gate-trip.

## Default berubah? 
Perilaku pose lock default SEDIKIT berubah secara sengaja sesuai kesepakatan:
dari full-lock(scale+rot) binary → adaptif tol 0.04/2.0° (ekspresi global asli
kini lolos). Kembali ke perilaku lama: set kedua tol = 0 di globals.
M-smoothing mengikuti toggle smoothing yang memang sudah ON default untuk video.
