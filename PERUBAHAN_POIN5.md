# Poin 5 — kalibrasi paste-back full pipeline (diagnostik + tuning live)

Regresi full pipeline (wajah geser/membesar/miring) berasal dari offset framing
bawaan generator LP yang teramplifikasi crop 2.3x saat di-paste balik. Geometri
back_M sendiri benar; yang kurang = koreksi sistematis kecil.

## Tambahan
- `live_portrait_util.apply_back_calibration(back_M,w,h,scale,dx,dy,rot)`:
  similarity koreksi (scale+rotasi terhadap pusat crop + translasi) lalu
  `C o back_M`. dx/dy = fraksi ukuran crop (bebas-resolusi). Identity (1,0,0,0)
  tak mengubah apa pun.
- `Expression_LivePortrait.Run()`: terapkan kalibrasi saat paste-back full
  pipeline; print `[lp-geom]` (back_M scale/rot/translasi + nilai cal) saat
  expression_debug aktif.
- globals: `expression_lp_cal_scale=1.0`, `_dx=0.0`, `_dy=0.0`, `_rot=0.0`.
- Settings → Expression: 4 slider live (cal scale / shift X / shift Y / rotation).

## Cara pakai (di L4)
1. Settings → centang "Full LivePortrait pipeline".
2. Preview satu wajah frontal yang jelas → akan tampak geser/membesar.
3. Geser 4 slider kalibrasi sambil lihat preview sampai wajah pas:
   scale (besar-kecil), shift X/Y (geser), rotation (miring).
4. Centang "Debug log" untuk melihat `[lp-geom]` per frame.
5. Lapor nilai yang pas → di-bake jadi default.

Catatan: kalibrasi ini satu affine global. Bila offset generator konsisten lintas
wajah/pose, satu set nilai cukup. Bila bervariasi dgn pose, ini memperbaiki kasus
umum tapi mungkin tak 100% di semua sudut (kita lihat dari hasil uji).

Validasi: py_compile + uji numerik kalibrasi (identity/scale-about-center/shift) +
build penuh gradio 5.9.1 OK.
