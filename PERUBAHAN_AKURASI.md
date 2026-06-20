
---

# TAHAP 6 — Perbaikan bug (preview lelet + expression identical) & Cleanup

## Bug #2: Expression restorer tidak berefek (FIXED)
Akar masalah: **urutan input generator terbalik**. FaceFusion memanggil
`forward_generate_frame(feature_volume, target_motion_points, temp_motion_points)`
-> input[1]=target/driving (ekspresi restored), input[2]=temp/source. Kode lama
menaruh kp_source di input[1] dan kp_driving di input[2], sehingga generator
mewarp ke arah ekspresi source -> hasil nyaris tak berubah ("identical").
- FIX di `Expression_LivePortrait.py`: input[1]=kp_driving, input[2]=kp_source.
- Jalur error dibuat vokal (traceback sekali) supaya ketahuan kalau model belum
  ter-download / ada masalah I/O, bukan diam-diam jadi no-op.
- `build_applied_expression` direstruktur ke pola FaceFusion (blend semua ->
  reset area non-terpilih ke source). Divalidasi: factor=0 no-op; factor=1
  semua-on = restore penuh; mouth-off = mulut tetap source, area lain restored.

## Bug #1: Preview lelet (dioptimalkan)
- Hull mask: `cv2.dilate` kernel ~103x103 per wajah/frame -> diganti ekspansi
  poligon dari centroid (hasil setara, ~10x lebih cepat).
- `trans_points2d`: loop Python -> divektorisasi.
- `pyvirtualcam` tidak lagi dependency startup (import StreamWriter dibuat lazy).
- Catatan: penyebab terberat kemungkinan saat expression restorer ON dengan bug
  urutan generator (4 inferensi ONNX/wajah + hasil identical). Di jalur dasar,
  tambahan komputasi kecil (~2ms) dan sudah dipangkas.

## Cleanup (sesuai permintaan)
- Download dihapus: DMDNet, GPEN, deoldify_artistic, deoldify_stable,
  real_esrgan_x2/x4, lsdir_x4. (isnet/CLIP/CodeFormer/GFPGAN/Restoreformer tetap.)
- File dihapus: Enhance_DMDNet.py, Enhance_GPEN.py, Frame_Colorizer.py,
  Frame_Upscale.py, ui/tabs/livecam_tab.py, roop/virtualcam.py.
- Registrasi plugin + branch get_processing_plugins untuk dmdnet/gpen/colorizer/
  upscale dihapus.
- UI: dropdown enhancer -> ["None","Codeformer","GFPGAN","Restoreformer++"];
  extras: entri deoldify & ESRGAN/LSDIR dihilangkan; Live Cam tab dihapus dari
  main.py; section "Select Output Method" dihapus (output selalu File); tombol
  "Open Output Folder" dihapus dari ketiga tab.
- pyvirtualcam dihapus dari requirements.txt.

## Yang masih perlu diverifikasi di run pertama Anda (tak bisa diuji di sandbox)
- Urutan output `motion_extractor` ONNX diasumsikan [pitch,yaw,roll,scale,
  translation,exp,motion_points] (sesuai FaceFusion). Jika ekspresi aneh, ini
  titik pertama yang dicek (traceback sekarang vokal akan membantu).
- Indeks keypoint area (Eyes/Mouth/Brows) empiris; semua-on = restore penuh.

---

# TAHAP 7 - Perbaikan LELET preview (2-3 menit/frame) + diagnostik

## Akar masalah (dikonfirmasi dari console L4 user)
onnxruntime CUDA memakai default `cudnn_conv_algo_search=EXHAUSTIVE`, yang
mem-benchmark SEMUA algoritma konvolusi cuDNN untuk setiap bentuk input baru
(di-cache per-bentuk). Dengan detektor `det_10g.onnx` ber-input dinamis
`[1,3,?,?]` dan generator LivePortrait yang besar, pencarian ini berulang dan
menyebabkan stall menit-an per frame. Model TIDAK reload per-frame (terverifikasi).

## Perbaikan
- `roop/utilities.py`: `tuned_execution_providers()` menyuntik
  `cudnn_conv_algo_search='HEURISTIC'` + `arena_extend_strategy='kSameAsRequested'`
  ke entry CUDA provider (aman untuk CPU-only).
- Dipakai di: `get_face_analyser` (buffalo_l), `FaceSwapInsightFace.Initialize`
  (inswapper), `Expression_LivePortrait._load` (3 model LP). insightface 0.7.3
  terbukti meneruskan provider_options -> buffalo_l ikut ter-tune.
- `roop/globals.py`: `cudnn_conv_algo_search='HEURISTIC'`, `profile_timings=True`.
- Logging `[load]` (saat session dibuat; harus 1x/sesi) dan `[timing]` (detect /
  swap+paste / per-processor) untuk verifikasi & diagnosa lanjutan.
- det_size tetap (640,640): SCRFD letterbox ke kanvas tetap -> blob detektor
  selalu (1,3,640,640), tak ada churn bentuk walau multi-angle merotasi frame.

## Catatan
- Set `profile_timings=False` di globals untuk render video panjang (kurangi spam).
- Blur expression: set "Subsample upscale to"=512px saat pakai expression restorer
  (LP terima crop hi-res, output tak dikecilkan ke 128).
