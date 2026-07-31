# PERBAIKAN — akar sebenarnya: source face diabaikan di mode "selected" (+ regresi UI saya)

## A. REGRESI YANG SAYA BUAT (trigger_mode="once")
Di batch sebelumnya saya set `trigger_mode="once"` pada 6 trigger preview untuk
mencegah "error connecting". Dokumentasi Gradio: "once" = *would not allow any
submissions while an event is pending* -> KLIK DIBUANG saat preview sedang render.
Akibat: klik source face sering tidak berefek, harus klik Refresh lagi. Lebih
buruk lagi, `fake_preview.change` defaultnya "always_last", jadi saya justru
MENURUNKAN perilakunya.
FIX: `trigger_mode="always_last"` -> penumpukan tetap dicegah (maks 1 antre) TAPI
klik terakhir SELALU dihormati. Auto-refresh saat pilih source face kembali
seperti semula.

## B. AKAR SEBENARNYA — kenapa ganti source face tak berefek di mode "selected"
`process_frame`, mode "selected" (kode lama):
    use_index = (jumlah target face == 1)
    ...
    if use_index: process_face(selected_index, ...)   # 1 target -> pakai pilihan
    else:         process_face(i, ...)                # 2+ target -> PAKAI INDEX
                                                      # TARGET, pilihanmu DIBUANG
Jadi begitu kamu menandai 2+ target face, source picker MATI TOTAL di mode ini.
Mode lain ("first"/"all"/"all_female"/"all_male") semuanya memakai selected_index
-> di sana ganti source face berfungsi. Itu persis yang kamu amati.

Ini juga menjelaskan "landmarking terlihat sangat berbeda antar mode": landmark-nya
TIDAK berbeda (sudah saya verifikasi: semua mode memakai get_all_faces_multi +
refine_faces_landmark68 + stabilizer yang sama). Yang berbeda adalah SOURCE FACE
YANG DIPAKAI -- di "selected" wajah A dapat source#0 dan wajah B dapat source#1,
sedangkan di "all_female" keduanya dapat source pilihanmu. Wajah sumber berbeda =
hasil swap berbeda, yang terbaca seperti "metode landmarking berbeda".

FIX: mode "selected" kini SELALU memakai source face yang kamu pilih, sama seperti
semua mode lain. Perilaku 1:1 pairing (wajah ke-i <- source ke-i) tetap tersedia
sebagai mode tersendiri: "all_input" / "all_random".
Tambahan pada fix: satu wajah tak bisa ter-swap dua kali (dulu bisa, bila 2 target
cocok ke wajah yang sama), dan selected_index di luar rentang jatuh ke 0.

Terverifikasi numerik (simulasi logika lama vs baru, 2 target + 3 source):
  LAMA: pilih source 0/1/2 -> hasil SELALU [(0,A),(1,B)]  (picker mati)
  BARU: pilih source 0/1/2 -> [(0,A),(0,B)] / [(1,A),(1,B)] / [(2,A),(2,B)]
  Kasus 1 target: LAMA == BARU (tidak ada perubahan perilaku).

## C. Audit konsistensi seluruh pipeline (permintaan "periksa holistik")
Semua call-site process_face diaudit:
  "first"                -> selected_index
  "all"                  -> selected_index
  "selected"             -> selected_index  (SETELAH fix; dulu index target)
  "all_female"/"all_male"-> selected_index
  "all_input"/"all_random"-> i  (memang mode pairing, disengaja)
Deteksi & landmark: SEMUA mode non-"first" memakai jalur identik
(get_all_faces_multi -> refine_faces_landmark68 -> stabilizer). "first" memakai
get_first_face_multi yang di dalamnya memanggil get_all_faces_multi lalu memilih
wajah terbesar -> deteksi & landmark tetap sama. Jadi tidak ada "metode
landmarking berbeda" antar mode.

## Yang TIDAK diubah
Fix sebelumnya tetap: analisis modul konstan (ER konsisten antar mode), deadzone
0.012 + slider, concurrency_limit=1 pada preview (tetap mencegah dua swap GPU
bersamaan), .then() pada source-select.

## Validasi
py_compile; build Gradio 5.9.1; 6 preview fn trigger_mode=always_last +
concurrency preview; rantai on_select_input_face -> on_preview_frame_changed utuh;
simulasi logika selected-mode (4 skenario) lolos.

## Uji di L4
1. Mode "selected" + 2 target face: klik-klik source face -> hasil HARUS berubah.
2. Bandingkan "selected" vs "all female" dengan source yang sama -> hasil swap
   pada wajah yang sama harus setara (bukan lagi source berbeda diam-diam).
3. Klik source face saat preview masih render -> perubahan tetap muncul (tak perlu
   Refresh manual).
