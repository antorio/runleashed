# PERUBAHAN — DEADBAND ANTI-JITTER + AUTO-STOP FIX + MENU MERGE (batch 3)

## 1. FIX micro-jitter dari One-Euro M (deadband piksel)
Gejala: dengan build batch-2, ER ON di footage yang alignment-nya sudah tenang
justru muncul micro-jitter halus terus-menerus di sekitar wajah. Build asli
(tanpa One-Euro M) tidak mengalaminya. User mengonfirmasi: mematikan "Landmark
smoothing" menghilangkan jitter -> pelakunya One-Euro M.

Akar masalah: `smooth()` lama membandingkan `np.allclose(M_s, M)` (rtol 1e-5),
praktis SELALU True, jadi SETIAP frame di-warp ulang dengan M yang beda sub-piksel
dari alignment asli. Pada footage yang sudah steady, One-Euro hanya menggeser crop
skala sub-piksel tiap frame -> resampling konstan = micro-jitter permanen.

Perbaikan: DEADBAND piksel di dalam `MatrixStabilizer.smooth`:
- Ukur pergeseran maksimum keempat sudut crop (dalam ruang frame) antara M mentah
  vs M ter-smooth -- metrik piksel yang benar untuk affine (bukan bandingkan
  elemen matriks yang campur radian/log-skala/piksel).
- Jika < `deadband_px` (default 1.2) -> kembalikan M MENTAH + flag changed=False;
  pemanggil MELEWATI warp-ulang sepenuhnya -> nol resampling tambahan, crop asli
  dipakai apa adanya.
- Jika >= threshold -> smoothing dipakai (changed=True).
Sweep 0.8-2.0px: 1.2 = clean footage hanya ~2% re-warp (praktis diam) sementara
jitter nyata tetap 99% ditangani (damp 0.27x) dan pan tetap diikuti (lag 2.2px).
`smooth()` kini mengembalikan (M, changed); pemanggil di `process_face` hanya
warp-ulang saat changed=True.

Validasi: still footage 0/120 re-warp; sub-px noise 7/300; jitter nyata 298/300
damp 0.27; pan 119/120 lag 2.22px; first-frame (M, False); multi-face independen.
Integrasi process_face: steady 0/55 re-warp, jittery 53/55.

## 2. FIX video tidak auto-stop + error "JSON.parse unexpected character"
Gejala: video selesai proses tapi tombol tetap "processing"; menekan Stop -> 
"Could not parse server response: SyntaxError: JSON.parse ...". User ingat ini
pernah di-fix di handoff v2 (regresi).

Akar masalah (off-by-one, warisan upstream yang saya flag di batch 1 tapi belum
diperbaiki): di `run_batch_inmem`, `frame_count = (frame_end - frame_start) + 1`
dipakai sebagai target tqdm & progress Gradio, TAPI `read_frames_thread` hanya
membaca `frame_end - frame_start` frame (`endframe` = COUNT dari
get_video_frame_total, jadi range = [start, end) = end-start frame). Progress
mentok di N/(N+1), generator Gradio tak pernah 100%, UI tetap "processing", dan
Stop lalu mengembalikan respons non-JSON -> error parse.
Perbaikan: buang `+1`. Target progress kini = jumlah frame yang benar-benar dibaca
= selaras dengan restore_audio & average_fps (yang juga pakai end-start). Run
mencapai 100% dan tombol kembali idle sendiri. (Verifikasi: 2431 == 2431.)

## 3. MERGE menu (lanjutan diet)
- "Stabilization" DIHAPUS sebagai grup terpisah; kedua kontrolnya (Landmark +
  warp smoothing + strength) pindah ke grup alignment. Grup di-rename
  "Alignment & detection" -> "Alignment & stabilization".
- "Expression restorer — tuning" DIHAPUS sebagai grup terpisah; semua isinya
  (expression smoothing, power, pose lock, pose gate) pindah ke grup ekspresi.
  Grup di-rename "Expression" -> "Expression Restorer".
Hasil: 5 grup accordion -> 3 di area itu, tanpa kehilangan satu kontrol pun.

## Validasi build
`py_compile` seluruh tree; build-test Gradio 5.9.1 (3 tab, 2 accordion ter-merge
benar, semua kontrol ter-wire); integrasi process_face (deadband + off-by-one).

## Default
deadband_px=1.2 (baru). Perilaku lain tak berubah. Untuk menonaktifkan One-Euro M
sepenuhnya seperti sebelumnya: matikan "Landmark + warp smoothing" (juga mematikan
landmark smoothing), atau set deadband_px sangat besar di file bila hanya ingin
mematikan warp-smoothing sambil mempertahankan landmark smoothing.
