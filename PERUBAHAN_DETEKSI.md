# Kontrol deteksi wajah (confidence + size)

## Masalah
Swap kedap-kedip: satu frame ke-swap, frame berikutnya tidak, padahal wajah cuma
bergerak sedikit. Penyebab: `prepare()` dipanggil TANPA `det_thresh`, jadi memakai
default insightface 0.5, dan tidak ada cara mengubahnya. Wajah yang skornya di
sekitar 0.5 lolos di satu frame dan gagal di frame berikutnya.

## Perubahan
- `det_thresh` (default 0.5) dan `det_size` (default 640) jadi globals.
- `get_face_analyser()` memanggil `prepare()` ULANG hanya bila salah satu berubah
  -> slider langsung berlaku tanpa restart, dan tidak me-reload model.
- UI (grup "Alignment & stabilization", tab Face swap):
  - slider "Detection confidence" 0.10-0.90
  - dropdown "Detection size" 320 / 640 / 1024
- Checkbox mati "Use default Det-Size" di tab Settings DIHAPUS (globalnya sudah
  tidak ada; hanya menulis ke config tanpa efek).

## Cara pakai
- Kedip swap muncul-hilang -> TURUNKAN Detection confidence (coba 0.35, lalu 0.25).
  Terlalu rendah bisa memunculkan deteksi palsu.
- Wajah kecil/jauh atau sumber low-res tak terdeteksi -> NAIKKAN Detection size
  ke 1024 (lebih lambat).

## Validasi
prepare() terbukti dipanggil ulang hanya saat parameter berubah
([(0.5,640),(0.30,640),(0.30,1024)] untuk 4 pemanggilan); globals benar;
default_det_size tak bersisa; build Gradio + wiring OK; py_compile seluruh tree.

## Catatan terkait (bukan perubahan)
- ER strength dan expression_power adalah KNOB YANG SAMA secara matematis:
  amt = factor * power, blended = temp + (targ-temp)*amt. Menaikkan strength ke
  150 identik dengan power 1.5 -> jitter yang sama. amt>1 = ekstrapolasi melewati
  ekspresi target, yang memperbesar noise motion-extractor secara proporsional.
- Warna "mentah" saat profil + ER on: keluaran generator LivePortrait MENGGANTI
  seluruh interior wajah (feather_blend hanya 20% tepi). LP dilatih dominan pada
  wajah frontal, jadi di profil tone keluarannya menyimpang. Belum diperbaiki --
  opsi: transfer statistik warna dari swapped_crop ke hasil LP.

## Colour match untuk Expression Restorer (toggle, default ON)
Masalah: keluaran generator LivePortrait MENGGANTI seluruh interior wajah
(feather_blend hanya 20% tepi). LP dilatih dominan pada wajah frontal, jadi di
pose samping tone keluarannya menyimpang -> kulit terlihat "mentah", sementara
di frontal cocok sehingga tak terasa.

Fix: `lpu.match_color(restored, swapped_crop)` -- transfer mean/std LAB dari
wajah hasil swap ke hasil LP, dijalankan SEBELUM feather_blend. LP tetap
menyumbang geometri ekspresi; warnanya ikut swap.
Detail penting: statistik diambil dari REGION TENGAH (60%), bukan seluruh crop,
karena hasil LP di-warp balik dengan BORDER_REFLECT sehingga pita luarnya berisi
piksel cermin yang akan mencemari mean/std. Kedua gambar sebidang, jadi jendela
yang sama sebanding.

Globals: expression_color_match=True, expression_color_match_strength=1.0
UI: checkbox "Match LivePortrait colour to the swap" (grup Expression Restorer).

Validasi numerik (tone shift realistis: lebih pucat, kontras turun, geser warna):
  selisih MEAN thd referensi 7.03 -> 0.49  (93% lebih dekat)
  selisih STD  thd referensi 3.43 -> 0.01  (100%)
  strength 0.5 memberi hasil di antara keduanya (3.64)
  sampah di tepi menggeser mean 0.00 -> region tengah efektif
  shape beda / None / sumber uniform (std=0) -> aman, tak ada NaN
