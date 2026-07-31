# PERUBAHAN — source-face refresh, error-connecting, whole-face jitter, ER-per-mode

## #1 — ganti source face "macet" (preview tidak refresh)
`input_faces.select` -> `on_select_input_face` hanya set global & return None, lalu
`.success()` (yang hanya jalan bila trigger menghasilkan perubahan output) sering
TIDAK jalan -> preview tetap pakai source lama sampai event lain me-refresh.
FIX: ganti ke `.then()` (jalan tanpa syarat setelah select) -> preview selalu
re-render dengan source baru.

## #5 — "error connecting" saat atur setting
Tiap .change/.select/.click memicu on_preview_frame_changed yang menjalankan SWAP
GPU penuh sinkron. Ubah setting cepat -> event menumpuk -> queue tersumbat ->
tunnel gradio.live timeout -> "error connecting".
FIX: semua 6 trigger preview kini `trigger_mode="once"` (event baru membatalkan
yang tertunda, tak menumpuk) + `concurrency_limit=1, concurrency_id="preview"`
(tak ada dua swap GPU jalan bersamaan).

## #3 — jitter di SELURUH wajah (alignment jitter)
Bukan albumentations (itu tak menyentuh alignment). Ini sisa jitter dari
LandmarkStabilizer: default lama meloloskan ~0.6px jitter saat kepala diam.
Data tuning (kepala diam, noise detektor 1.5px):
  deadzone 0.006 -> 0.595px ; 0.012 -> ~0.26px ; 0.015 -> 0.145px
  (menaikkan strength saja hampir tak berpengaruh; deadzone knob paling efektif)
  lag pan 3px/frame tetap ~4.4px (tak terasa).
FIX (A+B):
  A: default `landmark_smoothing_deadzone` 0.006 -> 0.012 (jitter wajah-diam -57%).
  B: slider "Still-face dead-zone" (0..0.03) dikembalikan ke UI -> bisa di-A/B
     sendiri per footage. Ini knob utama untuk whole-face jitter.

## #2 — ER "lebih ngefek" di all_female/male daripada selected
Akar: `initialize` menambahkan 'genderage' ke g_desired_face_analysis HANYA untuk
mode gender. Itu mengubah list -> get_face_analyser() me-REBUILD buffalo_l dengan
allowed_modules berbeda. Module set berbeda menghasilkan landmark_2d_106 yang
sedikit berbeda untuk wajah yang sama; karena landmark itu MENGGERAKKAN Expression
Restorer, ER strength yang sama terlihat berbeda (sering lebih kuat) di mode gender.
FIX: module set dibuat KONSTAN di semua mode (selalu sertakan 'genderage', murah)
+ default globals disamakan. Analyser tak lagi di-rebuild saat pindah mode, ER
konsisten, jadi value ER strength yang kamu pilih berlaku sama di semua mode deteksi.
Catatan: #1 dan #2 saling menguat -- keduanya membuat "selected mode" terasa
inkonsisten. Sekarang keduanya diperbaiki.

## Investigasi #2 (landmark selected vs all)
Diverifikasi: untuk wajah yang terpisah jelas, smoothing landmark IDENTIK antar
mode (0.0px). Perbedaan hanya muncul bila ada wajah kedua yang sangat dekat
(bisa "mencuri" track). Jadi "landmark beda antar mode" untuk kasus 1 wajah bukan
dari stabilizer, melainkan dari rebuild analyser (#2 di atas) + bug refresh (#1).

## Validasi
py_compile; build Gradio 5.9.1 (deadzone default 0.012, slider ada+wired, analysis
list konstan, 6 preview fn trigger_mode=once + concurrency=preview). Tuning
stabilizer diverifikasi numerik.

## Catatan uji L4
- Ganti source face -> preview harus langsung berubah (tanpa ganti detection).
- Atur setting cepat -> tidak lagi "error connecting".
- Whole-face jitter -> naikkan "Still-face dead-zone" (mulai 0.012, coba 0.015-0.02).
- ER strength -> sekarang harusnya konsisten antara selected dan all_female.
