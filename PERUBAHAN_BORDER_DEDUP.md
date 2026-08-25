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
FIX: sudut-0 kini mendapat BONUS skor (0.10), bukan hak veto. Ia tetap menang
pada skor sama atau selisih tipis, tapi deteksi rotasi yang jelas lebih yakin
menang atas dasar kualitas.
Validasi: sudut-0 0.38 vs rotasi 0.82 -> rotasi menang (dulu sudut-0 menang);
sudut-0 0.78 vs rotasi 0.82 -> sudut-0 menang (bonus, perilaku lama terjaga);
skor sama -> sudut-0 menang; dua wajah terpisah tetap keduanya disimpan.

## Validasi umum
py_compile seluruh tree; nol `borderValue=0.0` tersisa di face_util;
process_face x6 pada jalur alignment kps; tidak ada regresi.
