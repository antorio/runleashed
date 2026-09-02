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

## Lanjutan: Erosion & Blur size jadi GLOBAL
Masalah yang dilaporkan user: kedua nilai ini tersimpan per SOURCE FACE
(`faceset.faces[0].mask_offsets[4]/[5]`), TAPI tidak ada satu pun handler yang
membaca-balik nilainya saat source face diganti. Jadi slider selalu menampilkan
angka terakhir yang digeser, apa pun face yang sedang dipilih -- user tak punya
cara tahu setting mana yang sedang berlaku. Lebih buruk: `set_mask_offset()`
diam-diam TIDAK melakukan apa pun bila belum ada source face dimuat.

FIX: keduanya dipindah ke globals (`mask_erosion_iterations`, `mask_blur_size`).
Ini memang tempat yang benar: erosi/feather menggambarkan cara swap DIBAURKAN ke
target, bukan properti source face. `paste_upscale` kini membaca globals; indeks
4/5 pada mask_offsets tidak lagi dipakai (nol referensi tersisa).

Layout: info-text di bawah "Blur size" dihapus sehingga bentuknya sama persis
dengan "Erosion" di sebelahnya (info-text itulah yang membuat tinggi kedua
kontrol tidak sejajar). Nilai awal kedua slider kini dibaca dari globals.

Validasi: globals default 1/20; handler menulis ke globals bukan ke faceset;
setting berlaku MESKI belum ada source face (transisi 7px -> 34px untuk blur
10 -> 60); mask_offsets[4]/[5] terbukti diabaikan (dua nilai berbeda memberi
hasil identik); build Gradio OK; slider Blur size tak lagi punya info-text.
