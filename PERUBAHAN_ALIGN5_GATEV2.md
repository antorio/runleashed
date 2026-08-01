# PERBAIKAN JITTER — berdasarkan data probe (akhirnya terarah)

Probe user (2dfan4 ON, strength 0.95, deadzone 0.012):
  bbox_raw 3.52 | kps_raw 3.89 | lm68_raw 2.38 | M_raw 16.48 | M_smoothed 13.04
  gate tripped 31% | smoothing hanya buang 21%

## Yang dibaca dari data
1. AMPLIFIKASI 7x: landmark cuma 2.38px, tapi M paste 16.48px. estimate() dari 5
   titik memperkuat noise rotasi/skala x lengan tuas ~180px ke tepi crop =
   wobble besar di rahang/garis rambut. INILAH micro-wobble.
2. Smoothing existing cuma buang 21% karena alpha adaptif LEPAS saat kepala
   gerak (by design, anti-lag). Maka tuning strength/deadzone kemarin nyaris tak
   terasa -- hanya menolong segmen diam.
3. GATE TRIP 31%: dengan 2dfan4, 68->5 vs kps SCRFD punya OFFSET SISTEMATIS ~0.09
   (konvensi titik beda, bukan rusak) yang kebetulan di ATAS threshold lama 0.08.
   Gate lama berganti basis alignment (68->5 <-> kps) di sepertiga frame ->
   lompatan diskrit di lokasi yg ditentukan konten = "berkedip di tempat sama".

## Ide yang DIUJI lalu DIBUANG (karena data menolak)
- Pembobotan titik mata di fit: menambah wobble (baseline mata pendek -> noise
  rotasi lebih besar) + bias. Dibuang.
- Smoothing rotasi/skala di ruang-M dengan anchor passthrough: cuma 4-7% (translasi
  mentah ikut lolos). Dibuang.
Diukur dulu sebelum ditulis ke app -> tidak masuk ke kode.

## Perbaikan yang MASUK (terukur berhasil)
### A. Smoothing adaptif pada 5 TITIK ALIGNMENT (align5)
Menghaluskan tepat di titik tempat amplifikasi terjadi -- 5 titik align, SEBELUM
estimate(). alpha didorong oleh gerak PUSAT wajah: kepala diam -> pusat diam ->
smoothing kuat (wobble hilang); pan -> pusat gerak -> alpha->1 (tanpa lag).
Terukur in-context: corner wobble kepala-diam 4.18 -> 1.27px (-70%).
Globals: align5_smoothing=True, align5_alpha_min=0.12, align5_motion_frac=0.04.

### B. Sanity gate v2 (baseline adaptif + histeresis)
Belajar disagreement NORMAL sebagai baseline; hanya trip pada ANOMALI (spike
jauh di atas baseline), dengan histeresis (butuh 5 frame tenang utk keluar) biar
tak chatter. Simulasi footage user (offset 0.09 + spike): trip 80% -> 3%, flip
80 -> 6. Spike asli tetap tertangkap.

## Kill-switch
align5_smoothing=False mematikan A (kembali ke perilaku lama). Gate v2
menggantikan gate lama; ambang & histeresis via globals.

## Validasi
Semua angka di atas dari simulasi numerik + integrasi. py_compile + build Gradio
5.9.1. Reset state per-run (tak mewarisi klip sebelumnya). Guard getattr untuk
instance __new__.

## Uji L4 (jalankan probe LAGI untuk konfirmasi objektif)
  NO_ALBUMENTATIONS_UPDATE=1 python tools_jitter_probe.py clip.mp4 --frames 200 --hi-landmarker
M_smoothed harusnya turun jauh dari 13.04, gate trip dari 31% ke <5%. Lalu render
dan lihat: micro-wobble di rahang/garis rambut harusnya jauh berkurang, dan kedip
"di tempat yang sama" hilang. Kalau masih ada lag saat pan, naikkan
align5_motion_frac; kalau masih wobble saat diam, turunkan align5_alpha_min.
