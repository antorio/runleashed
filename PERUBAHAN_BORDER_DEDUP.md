# FIX — garis hitam di kotak swap (#3) + confidence rendah merusak landmark (#4)

## #3 Garis hitam di kotak hasil swap
Akar: `align_crop` / `align_crop_robust` / `transform` memotong dengan
`borderValue=0.0`. Bila crop melewati tepi frame, area yang hilang diisi HITAM.
Hitam itu ikut melewati swap, lalu saat paste-back matte yang sudah di-Gaussian-
blur melebar sedikit MELEWATI kotak crop, dan di sana `BORDER_REPLICATE`
menyalin piksel tepi crop -- yaitu hitam tadi -- sehingga muncul garis gelap di
sisi kotak swap. Lebih kentara pada subsample 128/256 karena crop kecil
diperbesar lebih banyak saat warp balik, sehingga pita hitamnya lebih lebar
dalam piksel frame.
FIX: ketiganya memakai `borderMode=cv2.BORDER_REPLICATE`. Area di luar frame
kini berisi salinan piksel tepi (warna kulit/latar nyata), bukan hitam.
Validasi: crop pada wajah yang menempel tepi frame -> 0 piksel hitam di
size 128/256/512, juga untuk align_crop_robust dan transform().
REGRESI: untuk wajah di TENGAH frame crop-nya IDENTIK bit-per-bit dengan
sebelumnya (selisih maks 0) -- perubahan ini hanya aktif saat crop melewati tepi.

## #4 Menurunkan Detection confidence justru memperburuk landmark
Dua sebab, yang kedua ada di kode kita:
1. Skor SCRFD adalah keyakinan anchor, dan anchor berkeyakinan rendah juga
   MEREGRESI KEYPOINT lebih buruk -- skor berkorelasi dengan kualitas lokalisasi.
2. `_dedup_faces` dulu memprioritaskan deteksi sudut-0 SECARA MUTLAK:
       return (0 if ang == 0 else 1, -s)
   Jadi begitu ambang diturunkan, deteksi sudut-0 yang LEMAH ikut lolos dan
   MENGALAHKAN deteksi rotasi yang jauh lebih kuat untuk wajah yang sama.
   Hasilnya: menurunkan ambang bukan sekadar "lebih permisif", tapi benar-benar
   memilih deteksi yang lebih jelek.
FIX yang saya coba: sudut-0 diberi BONUS skor (0.10), bukan hak veto.
>>> DICABUT LAGI. Hasil uji user: yang tadinya bagus di 0.5 jadi rusak -- hanya
area HIDUNG yang ter-swap, dan baru pulih di 0.6.
Sebabnya: prioritas mutlak sudut-0 itu PENGAMAN, bukan kelalaian. Footage
berorientasi normal, jadi deteksi sudut-0 adalah kebenaran; pass rotasi hanya
cadangan untuk wajah yang memang miring DI DALAM frame. Dengan skema bonus,
deteksi rotasi berskor tinggi -- yang bisa jadi positif-palsu yang mendarat di
sebagian wajah -- merebut wajah dari deteksi sudut-0 yang benar. Kps-nya lalu
berkerumun, similarity fit mem-zoom habis, dan hanya hidung yang ter-swap.
Sebab #1 (skor rendah = keypoint kurang presisi) tetap berlaku dan itu SIFAT
DETEKTOR, bukan bug kita -- tidak ada yang perlu diubah di kode untuk itu.
Validasi setelah revert: sudut-0 menang di semua kombinasi skor (0.38 vs 0.82,
0.78 vs 0.82, seri, 0.95 vs 0.40); rotasi tetap dipakai bila sudut-0 tak
mendeteksi; dua wajah terpisah tetap utuh.

## Validasi umum
py_compile seluruh tree; nol `borderValue=0.0` tersisa di face_util;
process_face x6 pada jalur alignment kps; tidak ada regresi.
