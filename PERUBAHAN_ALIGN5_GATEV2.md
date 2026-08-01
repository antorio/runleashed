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

## HOTFIX (setelah probe run kedua user)
Probe run kedua menunjukkan angka IDENTIK (62/200 trip, raw sama persis) --
karena PROBE TIDAK MELEWATI KODE BARU: align5+gate v2 hidup di process_face,
sedangkan probe menghitung gate & M dengan logika lamanya sendiri. Kesalahan
saya (tested != shipped).
FIX STRUKTURAL: logika diekstrak ke kelas bersama `Align5Conditioner`
(roop/align_conditioner.py) yang dipakai OLEH KEDUANYA -- process_face DAN
probe. Probe kini melaporkan baris baru:
    M_pipeline   <- jalur render sebenarnya (align5 + gate v2)
    gate v2: fallback X%, state flips N, learned baseline B
dan baris lama "gate tripped" diganti label "[info only]" (statistik mentah di
atas threshold lama, bukan perilaku app).
Validasi pada kelas yang dikirim (bukan salinan): gate fallback 3% (vs 80% lama),
6 flips, spike tertangkap; align5 wobble steady -67%, pan lag 1.6px; integrasi
process_face x12 OK; smoothing dimatikan otomatis untuk batch gambar
(video_mode=False).

## HOTFIX 2 — threshold RANSAC saya sendiri menimbulkan kedip baru
Dari dua run probe user yang hasilnya sedikit BERBEDA, saya menguji stabilitas
himpunan inlier RANSAC. Hasil: dengan thr = 0.05*crop (12.8px pada 256), satu
titik yang duduk dekat ambang KELUAR-MASUK himpunan inlier pada ~34% frame ->
fit berganti antara 4 dan 5 korespondensi -> lompatan diskrit di M tiap kali.
Terukur: lompatan sudut antar-frame 5.92px (thr>=0.08) vs 9.16px (thr 0.05) --
55% LEBIH BURUK. Saya menukar penanganan outlier yang jarang dengan kedip di
sepertiga frame. Salah.
FIX: thr = 0.08*image_size -> 0% inlier flipping, lompatan kembali 5.92px, dan
tetap menolak outlier jauh lebih baik daripada thr=100 lama.

## Status gate v2: BERHASIL (dari probe user)
  gate v2: fallback 0/200 (0.0%), state flips 0, learned baseline 0.07
Bandingkan disagreement mentah yang di atas threshold lama: 62/200 (31%).
Artinya offset sistematis 2dfan4-vs-kps kini dikenali sebagai NORMAL (baseline
0.07 dipelajari), bukan dianggap rusak -> tidak ada lagi pergantian basis
alignment per frame. Sumber "kedip di tempat yang sama" itu sudah hilang.

## Catatan variabilitas probe (pertanyaan user)
bbox_raw & kps_raw IDENTIK antar run (3.520 / 3.889) -> deteksi SCRFD
deterministik. lm68_raw sedikit berbeda (2.377 vs 2.381) -> sesi 2dfan4 dengan
cudnn_conv_algo_search=EXHAUSTIVE memilih algoritma konvolusi yang bisa berbeda
antar proses, memberi selisih floating-point kecil. Selisih kecil itu lalu
diperkuat ~7x oleh estimate() menjadi ~1px perbedaan di M_raw. M_pipeline justru
paling stabil antar run (10.356 vs 10.397) -- itu angka yang layak dipakai
sebagai pembanding.

## HOTFIX 3 — regresi ditemukan lewat run 1000 frame user
Probe 1000 frame: M_pipeline 119.5 vs M_raw 50.0 -> pipeline saya 139% LEBIH
BURUK. Baris gate menjelaskan: fallback 67/1000, flips 9. Baseline terpelajar
0.07 berarti basis 68->5 dan basis kps berjarak SISTEMATIS ~14px pada wajah
200px -- jadi tiap pergantian basis = lompatan besar. Di klip 200 frame fallback
0% sehingga tak terlihat; di 1000 frame muncul.
Akar: saya merancang CROSSFADE di simulasi awal tapi TIDAK mengimplementasikannya
-- yang masuk ke kode hanya switch boolean keras.
FIX:
 1. CROSSFADE: bobot berjalan 0.25/frame antara kedua basis, jadi fallback
    protektif tidak pernah tampak sebagai kedip. condition() kini SELALU
    mengembalikan satu himpunan titik siap-fit (tak ada lagi cabang keras).
 2. RESET DISKONTINUITAS: bila pusat wajah melompat > 0.5x ukuran wajah (scene
    cut / wajah target berganti orang), history dibuang alih-alih di-blend --
    blending dua wajah berbeda menghasilkan posisi yang bukan milik keduanya.
Terukur: residual raw 3.95 -> pipeline 2.50; LOMPATAN ANTAR-FRAME 6.15 -> 1.09px
(-82%); pipeline tidak pernah lebih buruk dari raw; scene cut ter-reset (deviasi
<1px setelah cut).

## Metrik baru di probe: FLICKER
residual_jitter memakai moving-average 5 frame -> pada klip dengan scene cut /
pan cepat, sisa gerak NYATA ikut terhitung (itu sebabnya angka 1000-frame
melonjak ke 40-50px: bbox_raw 41.9 menunjukkan klip itu penuh gerak/cut, bukan
jitter). Ditambahkan metrik FLICKER = median lompatan antar-frame, yang kebal
terhadap cut & pan dan paling cocok dengan apa yang MATA lihat. Bandingkan
baris M_raw vs M_pipeline pada blok FLICKER.
