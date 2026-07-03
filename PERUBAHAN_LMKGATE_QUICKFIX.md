# PERUBAHAN — LANDMARK SANITY GATE + BATCH QUICK-FIX (post-handoff-v2)

## 1. Landmark sanity gate (FITUR BARU — target: distorsi "landmark salah" di frame sulit)
Di frame pose ekstrem / blur / wajah ter-stretch, landmark 68pt gagal LEBIH DULU
daripada 5 kps detektor → 68→5 rusak → affine salah → swap terlihat geser/stretch.
Gate per-frame di `ProcessMgr.process_face`: bandingkan 5 titik hasil 68→5 dengan
kps detektor (keduanya menandai mata/hidung/mulut yang sama, gratis).
- Trip bila `mean(disagreement)/face_size > thr` ATAU `max per-titik > 2×thr`
  (kriteria max menangkap kegagalan paling umum: SATU titik liar yang oleh
  rata-rata 5 titik justru terdilusi — divalidasi numerik).
- Saat trip → fallback alignment kps detektor untuk frame itu saja.
- Globals: `landmark_sanity_gate=True`, `landmark_sanity_threshold=0.08`.
- UI: accordion "Alignment & detection" → checkbox + slider threshold (live).
- Debug: print `[lmk-gate] ... mean=.. max=..` tiap trip saat `expression_debug=True`.
Catatan A/B: gate ini netral terhadap 2dfan4 ON/OFF — dia menilai landmark 68
apa pun sumbernya, jadi bisa dipakai untuk menguji ulang 2dfan4 secara objektif.

## 2. FIX jalur auto-rotate melewati refinement (keluhan wajah horizontal)
`process_face` rotated-retry: `rotface` dari `get_first_face` dulu TIDAK melewati
`refine_faces_landmark68` → justru frame tersulit memakai landmark paling mentah.
Kini refinement dipanggil (stabilizer sengaja TIDAK — koordinat rotcutframe beda
ruang dengan frame penuh, track-nya akan mismatch). Gate #1 juga aktif di jalur ini.

## 3. FIX cv2.resize interpolasi diabaikan diam-diam (6 lokasi)
`cv2.resize(img, size, cv2.INTER_CUBIC)` menaruh flag di argumen `dst` → OpenCV
diam-diam memakai INTER_LINEAR (diverifikasi empiris: positional == linear).
Diperbaiki ke `interpolation=cv2.INTER_CUBIC` di:
- `ProcessMgr.py` upscale crop→512 sebelum paste (paling berdampak ke ketajaman)
- `Enhance_GFPGAN/CodeFormer/RestoreFormerPPlus` pre-resize 512
- `Mask_XSeg`, `Mask_FFXSeg_base` resize 256 (efek kecil, konsistensi)
Efek visual: hasil TANPA enhancer sedikit lebih tajam/halus dari build sebelumnya.

## 4. FIX enum `SKIP_FRAME_IF_DISSIMILAR = 3,` (koma nyasar → tuple)
Perbandingan int==tuple selalu False → mode itu diam-diam berperilaku seperti
Retry-rotated dan tak pernah men-skip frame dissimilar. Kini int 3, mode berfungsi.

## 5. FIX chaining frame-processor di `process_frames` (jalur extract-frames)
`frame = p.Run(temp_frame)` tidak meng-chain → hanya efek processor TERAKHIR yang
dipakai. Kini chain seperti `process_videoframes`. + lookup source→target via dict
(hapus `source_files.index(f)` yang O(n²) untuk batch panjang).

## 6. Mask oklusi: 1× inference, apply 2×
Saat mask jalan setelah enhancer, model mask dijalankan DUA KALI pada crop yang
sama (untuk fake_frame dan enhanced_frame). `process_mask` dipecah jadi
`compute_mask` (inference) + `apply_mask` (blend); inference kini sekali.
`process_mask` tetap ada sebagai wrapper kompatibilitas. Output identik
(diverifikasi array-equal), hanya lebih cepat saat mask-after-enhancer + enhancer.

## 7. Guard `blur_area` terhadap matte tanpa piksel ==255
`np.max` pada array kosong crash bila user imagemask ter-antialias (max 254).
Kini fallback ke area >0; matte kosong dikembalikan apa adanya.

## 8. Bersih-bersih
Duplikat ~19 baris ekor `paste_upscale` yang unreachable di `_match_color_masked`
(setelah `return`) dihapus.

## Validasi
- Math (numpy/cv2): gate 5 skenario (good, offset-definisional 4%, 1-titik liar,
  2-titik liar, full drift) semua benar; blur_area 3 kasus; mask refactor
  array-equal dengan jalur lama; chaining (10+1)*2=22; resize cubic ≠ linear.
- `py_compile` seluruh tree OK.
- Build-test Gradio **5.9.1** (versi user): 3 tab + theme + css + js terbangun,
  kontrol lmk-gate ter-wire live.

## Default TIDAK berubah
2dfan4 tetap OFF (sesuai observasi user), pose gate tetap OFF, cudnn tetap
EXHAUSTIVE + max_workspace (pilihan speed user). Tidak ada menu opsional yang
dijadikan default.
