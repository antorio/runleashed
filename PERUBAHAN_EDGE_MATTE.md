# FIX — hasil swap terpotong di pinggir frame ("bingkai dalam")

## Gejala
Wajah yang berada di tepi frame: hasil swap tidak mencapai garis pinggir gambar,
seolah ada frame lain yang lebih ke dalam.

## Akar masalah (terukur)
`paste_upscale` menghitamkan 1 piksel terluar frame SEBELUM `blur_area`:
    img_matte[:1,:] = img_matte[-1:,:] = img_matte[:,:1] = img_matte[:,-1:] = 0
Sendirian itu tak terlihat. Tapi `blur_area` berikutnya melakukan EROSI dengan
kernel yang skalanya mengikuti ukuran wajah (k = max(mask_size//10, 10)), lalu
Gaussian blur. Erosi memakan ke dalam DARI nol yang baru dibuat di tepi frame
itu, dan blur melebarkannya lagi.
Terukur (wajah menempel tepi kiri):
    wajah 200px -> swap baru mulai di x=10px
    wajah 300px -> x=15px
    wajah 400px -> x=20px
Jadi ada pita mati 10-20px, makin lebar makin besar wajahnya.

## Perbaikan
1. `blur_area` kini mem-PAD matte dengan BORDER_REPLICATE sebelum erosi+blur,
   lalu memotong pad-nya kembali. Area di luar gambar tidak lagi dianggap
   background, jadi matte yang memang menyentuh tepi frame tetap mencapainya.
   Pad = k*iterasi + 2*k_blur + 4 (cukup untuk kedua operasi).
2. Penghitaman 1px tepi frame DIPINDAH ke SETELAH feather selesai. Tujuan
   aslinya (mencegah matte bocor keluar frame saat blending float) tetap
   tercapai, tapi sekarang benar-benar hanya 1px dan tidak jadi bahan erosi.

## Validasi
A. wajah menempel tepi kiri -> swap mulai di x=1px (dulu 10/15/20px) utk wajah
   200/300/400px.
B. tepi wajah yang BUKAN batas frame -> feather tetap 9/13/17px dan erosi tetap
   14/21/28px, sama seperti sebelumnya (tidak ada perubahan perilaku).
C. wajah di TENGAH frame -> selisih maksimum vs perilaku lama = 0 (IDENTIK).
D. baris/kolom terluar tetap 0 -> tidak ada kebocoran ke luar frame.
E. matte kosong / matte tanpa piksel 255 -> tidak crash.
F. integrasi paste_upscale dengan M yang menempel tepi -> swap terlihat dari x<=2.
