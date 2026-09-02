# FIX — garis tipis di tepi swap: "Blur size" dipisahkan dari erosi

## Keluhan
Terlihat garis tipis di pipi yang memisahkan warna/tekstur swapped face dan
target face. User ingin transisi lebih halus TANPA mengurangi area swap.

## Akar masalah: satu angka mengendalikan dua hal, dan TERBALIK
    k(erosi) = max(mask_size // (blur_amount // 2), blur_amount // 2)
    k(blur)  = max(mask_size // blur_amount,        blur_amount // 5)
Keduanya memakai `blur_amount` (slider "Blur size") secara BERBANDING TERBALIK.
Terukur pada wajah 300px:
    Blur size  10 -> erosi 60px | blur 30px
    Blur size  20 -> erosi 30px | blur 15px   (default)
    Blur size  30 -> erosi 20px | blur 10px
    Blur size  40 -> erosi 20px | blur  8px
Jadi MENAIKKAN slider justru MENGECILKAN feather, dan satu-satunya cara menambah
feather (menurunkan slider) melipatgandakan erosi -> area swap menyusut. Persis
yang user tidak inginkan. Permintaan "lebih halus tanpa mengurangi area" memang
mustahil dengan rumus ini.

## Perbaikan: dipisahkan
    erosi  -> kernel tetap max(mask_size//10, 10), banyaknya diatur slider Erosion
    blur   -> k2 = max(mask_size * blur_amount // 400, 4), naik monoton
Nilai default blur_amount=20 menghasilkan erosi 30 / blur 15 -- PERSIS angka lama.
Slider "Blur size" diperlebar 10-50 menjadi 4-100 agar rentangnya berguna, dengan
info singkat "edge softness only; does not shrink the swap".

## Validasi
- default (blur=20): matte hasil IDENTIK bit-per-bit dengan perilaku lama
  (selisih maks 0)
- menaikkan blur 4 -> 100: lebar transisi 10-90% naik monoton 5px -> 59px
- tepi 50% mask BERGESER 0px di seluruh rentang -> area swap tidak menyusut
- slider Erosion tetap berfungsi terpisah (tepi 50%: 685 -> 671 -> 657 utk 1/2/3x)
- py_compile + build Gradio 5.9.1

## Cara pakai
Garis tipis di pipi -> naikkan "Blur size" (coba 40, lalu 60). Area swap tetap.
Kalau swap perlu dipersempit -> pakai slider "Erosion", bukan Blur size.
