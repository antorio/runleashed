# HOTFIX — video hasil hilang saat finish (regresi batch 4)

## Masalah (dilaporkan user)
Setelah batch 4: log 100% tapi "Finished" tak pernah muncul, harus Stop manual,
dan VIDEO HASIL TIDAK ADA di output folder. Lebih parah dari sebelumnya.

## Akar masalah (2 bug yang SAYA buat di teardown batch 4)
1. `run_batch_inmem` teardown memakai `writethread.join(timeout=30)` untuk SEMUA
   kasus. Pada video normal, write thread masih menulis frame terakhir saat
   timeout memutusnya -> kode lanjut ke `videowriter.close()` PADAHAL write belum
   selesai -> video ter-close di tengah, tak ter-mux, tak sampai restore_audio/
   move -> tidak ada di output. (Build asli: `join()` tanpa timeout = benar.)
2. Sentinel `None` di read thread di-`put(timeout=0.5)` lalu `except Full: pass`
   -> bila queue penuh sesaat, sentinel HILANG -> worker tak dapat None -> tak
   kirim (False,None) -> write thread tunggu producer selamanya -> join hang.

## Prinsip perbaikan (sesuai permintaan user: hasil ADA, error boleh)
JALUR SUKSES HARUS = PERILAKU BUILD ASLI PERSIS. Timeout hanya untuk jalur error.

- `run_batch_inmem`: bila tak ada worker error -> `readthread.join()` +
  `writethread.join()` TANPA timeout (tunggu tulis benar-benar selesai, video
  utuh tersimpan). Timeout-guarded teardown HANYA dipakai saat worker crash.
- Sentinel `None` kini `put(block=True)` TANPA timeout -> dijamin terkirim,
  tak pernah hilang.
- Error worker di-raise SETELAH videowriter.close() -> frame yang sempat
  diproses tetap ter-flush ke file (video parsial tetap ada, lalu error muncul).
- `start_swap`: filter file-by-mtime (yang bisa gagal karena clock Drive != lokal
  dan menyembunyikan hasil) DIBUANG; kembali ke listing semua file output seperti
  asli. Yang dipertahankan hanya: UI selalu balik idle + warning non-fatal.

## Validasi
- Simulasi pipeline thread: 6 skenario sukses (termasuk nf=7<th=8 dan nf habis
  dibagi thread) semua menulis 100% frame tanpa terpotong; stop-mid & crash tak
  hang, error ter-surface.
- END-TO-END dengan FFMPEG_VideoWriter asli: video 60 frame diproses lewat
  run_batch_inmem -> output MP4 ADA, 60/60 frame, ukuran wajar. Membuktikan jalur
  sukses menyimpan video utuh.
- py_compile + build Gradio 5.9.1 OK.

## Catatan untuk user
Video yang tadi hilang mungkin tertinggal sebagai `*__temp.mp4` di output folder
(writer terpotong sebelum rename final). Cek & rename manual bila ada. Build ini
tak mengulang masalahnya.
