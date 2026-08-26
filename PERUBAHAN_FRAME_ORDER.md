# FIX MENDASAR — temporal smoothing selama ini menerima frame TERACAK

## Temuan
`read_frames_thread` membagikan frame secara round-robin:
    frames_queue[num_frame % num_threads]
Dengan 8 thread: thread 0 dapat frame 0,8,16..., thread 1 dapat 1,9,17..., dan
KEDELAPAN thread berjalan BERSAMAAN sambil berbagi SATU LandmarkStabilizer.
Lebih parah: nomor frame TIDAK PERNAH dikirim ke worker sama sekali (hanya
gambarnya), jadi stabilizer bahkan tak punya informasi untuk mengurutkan.

Akibatnya:
- "frame sebelumnya" yang dipakai menghaluskan bisa frame mana pun dari 8 yang
  sedang diproses -> smoothing membaur dengan tetangga yang SALAH.
- alpha adaptif mengukur "gerakan" terhadap frame berjarak ~8 -> selalu
  menyimpulkan gerakan besar -> smoothing melepas diri.
- frame N dan N+1 dikerjakan thread BERBEDA dengan riwayat berbeda -> dua frame
  berurutan dengan wajah nyaris sama bisa menghasilkan landmark berbeda.
Terukur pada simulasi pembagian yang sama persis: 187 lompatan mundur dari 399
transisi. Hampir separuh frame tiba tidak berurutan.

Ini menjelaskan kenapa SEMUA penyetelan smoothing kita terasa sia-sia: datanya
memang sudah teracak sebelum sampai ke sana.

## Perbaikan
1. `read_frames_thread` kini mengirim `(nomor_frame, frame)`.
2. `process_videoframes` membongkarnya dan meneruskan ke
   `process_frame(frame, frame_index)` -> `swap_faces(..., frame_index)`.
3. `FrameSequencer` baru: bagian DETEKSI + refine landmark + smoothing dijalankan
   dalam URUTAN FRAME SESUNGGUHNYA. Bagian berat (swap/paste/enhancer) tetap
   PARALEL penuh -- saat frame N ada di bagian temporal, frame N-1 sudah
   menyawap. Jadi throughput hampir tak terpengaruh.
4. Pengurut hanya aktif bila smoothing menyala dan ada nomor frame; preview dan
   batch gambar melewatinya (index None -> tanpa pengurutan).

## Pengaman anti-hang (diuji)
- frame hilang/dilewati -> menunggu maksimal `timeout` (default 10s) lalu jalan
  terus tanpa urutan, BUKAN menggantung.
- exception di dalam bagian temporal -> counter tetap maju (try/finally), frame
  berikutnya tidak terkunci.
- abort (Stop) -> menunggu dalam potongan 0.1s supaya flag `processing` terbaca;
  penunggu keluar seketika (diukur 0.00s), bukan setelah timeout penuh.
- reset per-run bersamaan dengan reset stabilizer.

## Validasi
- 8 thread, pembagian round-robin persis seperti produksi, 400 frame ->
  bagian temporal dimasuki dalam urutan SEMPURNA 0..399.
- pembanding tanpa pengurut pada kondisi sama -> 187 lompatan mundur.
- frame hilang -> selesai 1.00s, frame lain tetap terproses.
- exception di frame 1 -> frame 0,2,3 tetap jalan.
- abort -> penunggu keluar 0.00s.
- END-TO-END dengan FFMPEG asli, 8 thread, 80 frame -> video tersimpan 80/80,
  urutan tulis benar (nilai frame monoton naik).
- py_compile seluruh tree.

## Catatan
Perbaikan ini TIDAK mengubah nilai setting apa pun. Ia membuat smoothing yang
sudah ada bekerja pada data yang benar. Efeknya baru terlihat saat render video
(preview tetap tanpa smoothing, seperti sebelumnya).
