# TOGGLE ALIGNMENT + HAPUS KODE MATI + AUDIT A-Z (3 pass)

## 1. Toggle "Landmark alignment (68pt)" dikembalikan ke UI
Grup "Alignment & stabilization", paling atas. ON = align dari 68 titik (perilaku
sekarang). OFF = align dari 5 kps detektor (cara Rope: lebih kasar tapi lebih
tenang di pose sulit). Ter-wire ke roop.globals.use_landmark_alignment.
Catatan: saya sendiri yang menghapus toggle ini waktu "diet menu" batch 2 dengan
asumsi "sudah settled". Asumsi itu salah.

## 2. TIDAK ADA LAGI INFERENCE YANG DIBAYAR TAPI TIDAK DIPAKAI
Temuan audit: `landmark_3d_68` selalu ada di g_desired_face_analysis, sehingga
buffalo_l menjalankan 1k3d68.onnx untuk SETIAP wajah di SETIAP frame -- padahal
setelah hull mask dihapus, satu-satunya konsumen 68 titik adalah alignment.
FIX berlapis:
- `initialize`: "landmark_3d_68" hanya dimasukkan ke daftar modul analyser bila
  use_landmark_alignment ON. OFF -> model 68pt tidak dimuat/dijalankan sama sekali.
- `refine_faces_landmark68`: return lebih awal bila use_landmark_alignment OFF ->
  inference 2dfan4 juga tidak jalan (hasilnya toh tak dipakai).
Daftar modul tetap KONSTAN antar swap-mode (genderage selalu ada) supaya analyser
tak di-rebuild saat ganti mode dan ER tetap konsisten.

## 3. HULL MASK DIHAPUS TOTAL
`_build_hull_matte()` (~90 baris), call site di paste_upscale, dan 4 globals
(use_face_hull_mask, face_hull_forehead, face_hull_dilate, face_hull_min_area)
DIHAPUS. Fitur ini mati (default False) dan tidak punya kontrol UI sama sekali --
persis jenis kode yang membingungkan. Komentar usang yang menyebutnya juga
dibersihkan. Sisa referensi "hull" di seluruh tree: NOL.

## 4. Audit A-Z, tiga pass
PASS 1 -> 21 fungsi tak pernah dipanggil + 6 globals tak pernah dipakai. Dihapus.
  fungsi: suggest_max_memory, batch_process_with_options, auto_rotate_frame,
  unload_models, sort_filenames_ignore_path, normalize_output_path, move_temp,
  prepare_for_batch, face_offset_top, rotate_image_180, square_crop, trans_points,
  join_videos, create_video_from_gif, repair_video, get_perspective,
  on_set_output_folder, move_selected_input, move_selected_target,
  on_clear_input_faces, get_gradio_output_format
  globals: BATCH_IMAGE_CHAIN_PROCESSOR, FACE_ENHANCER, IMAGE_CHAIN_PROCESSOR,
  VIDEO_CHAIN_PROCESSOR, face_position, many_faces
PASS 2 -> 3 fungsi BARU jadi yatim akibat penghapusan pass 1 (trans_points2d &
  trans_points3d dipakai hull matte; get_temp_output_path dipakai move_temp).
  Dihapus. `roop/vr_util.py` jadi kosong (isinya cuma get_perspective) -> FILE
  DIHAPUS beserta importnya di ProcessMgr.
PASS 3 -> fungsi mati: 0, globals mati: 0. Bersih.

### Pembersihan import: DIBATALKAN dengan sengaja
Percobaan menghapus 21 import tak terpakai merusak 3 file enhancer (regex saya
menggabungkan baris). File dipulihkan dari referensi bersih dan pembersihan
import DIHENTIKAN: manfaatnya nol saat runtime, risikonya nyata. Import tak
terpakai yang tersisa dibiarkan. Yang sudah terlanjur dihapus diverifikasi satu
per satu benar-benar tak dipakai (11 pengecekan, semua 0 pemakaian tersisa).

## Validasi
py_compile seluruh tree; 6 pemeriksaan fungsional (hull hilang, toggle default,
daftar modul mengikuti toggle, skip 2dfan4, toggle ter-wire di UI, keeper utuh);
build Gradio 5.9.1; uji runtime process_face di 3 skenario: alignment ON dgn lm68,
alignment OFF tanpa lm68 (kondisi nyata saat toggle OFF), alignment ON tapi lm68
hilang (guard fallback ke kps). Semua lolos.

## Yang TIDAK diubah
Perilaku render saat toggle ON = persis seperti sebelumnya. Semua keeper tetap:
selected-mode source face, analyser konstan, refresh preview saat ganti source,
Restore eyes, model HF, perbaikan auto-stop.

## LANJUTAN — dua temuan yang audit pertama saya LEWATKAN

### Kenapa terlewat (jujur)
Audit pertama digerakkan skrip: cari fungsi/globals yang TIDAK PERNAH DIRUJUK.
Itu hanya SATU dimensi. Pemborosan di bawah ini kelas berbeda: kodenya dirujuk,
dipanggil, lalu hasilnya DIBUANG. Tidak ada grep yang menangkapnya -- hanya
penalaran aliran data. Saya menjalankan 3 pass alat lalu mengklaim "A sampai Z",
padahal yang diaudit hanya satu sumbu. Klaim itu berlebihan.

### Temuan 1: 1k3d68 terbuang saat 2dfan4 ON
2dfan4 MENIMPA landmark_3d_68; satu-satunya yang diambil dari buffalo adalah
kolom z, dan TIDAK ADA yang membaca z (landmark_68_to_5 pakai [:, :2], stabilizer
tak menyentuh z). Jadi dengan 2dfan4 ON, 1k3d68 dijalankan tiap wajah tiap frame
lalu dibuang seluruhnya.
FIX: "landmark_3d_68" hanya diminta bila alignment ON *dan* 2dfan4 OFF. Kode
penyalin z disederhanakan jadi nol + komentar jujur.
Hasil daftar modul:
  align ON  + 2dfan4 OFF -> [landmark_3d_68, landmark_2d_106, detection, recognition, genderage]
  align ON  + 2dfan4 ON  -> [landmark_2d_106, detection, recognition, genderage]
  align OFF              -> [landmark_2d_106, detection, recognition, genderage]

### Temuan 2: sanity gate cacat DESAIN -> DICABUT TOTAL
User benar: ambang berapa pun tidak menyembuhkan. Mekanisme yang memilih basis
alignment PER FRAME lewat ambang akan selalu bisa bolak-balik (frame 1 lewat,
frame 2 tidak, frame 3 lewat) -- dan kedua basis berjarak sistematis, jadi tiap
perpindahan adalah lompatan. Ambang hanya mengatur SEBERAPA SERING melompat.
Gate menukar distorsi yang jarang dengan lompatan yang sering.
Saya sempat menyadari gejalanya (karena itu saya buat crossfade) tapi memilih
menambal alih-alih menyimpulkan bahwa mekanismenya memang salah.
FIX: gate DICABUT seluruhnya -- kode di process_face, 2 globals
(landmark_sanity_gate, landmark_sanity_threshold), dan slider UI-nya. Kini SATU
basis alignment, konsisten, tanpa perpindahan per frame. Tidak ada swapper lain
(Rope, FaceFusion, deep-live-cam) yang punya gate semacam ini.

### Trade-off yang saya SEBUTKAN, bukan sembunyikan
'genderage' tetap selalu diminta walau hanya dipakai mode all_female/all_male.
Alasannya: daftar modul harus KONSTAN antar mode, kalau tidak analyser di-rebuild
dan landmark_2d_106 berubah halus -> ER terasa beda antar mode (bug yang sudah
kita perbaiki). genderage.onnx input 96x96 (jauh lebih ringan dari 1k3d68 192x192),
jadi ini pertukaran sadar: konsistensi ER dibayar dengan inference sangat kecil.
Kalau kamu lebih memilih menghematnya, bilang -- tapi ER akan beda lagi antar mode.

### Validasi
py_compile; audit ulang (fungsi mati 0, globals mati 0); daftar modul diverifikasi
untuk 3 kombinasi toggle; process_face diuji di 3 skenario (align ON+68, align OFF,
align ON tanpa 68); konsistensi basis diukur (tak ada perpindahan); build Gradio.
