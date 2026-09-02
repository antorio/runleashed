# Subsample upscale: tambah 768px & 1024px (+ perbaiki bug parsing)

## Bug yang ditemukan lebih dulu
`roop.globals.subsample_size = int(upsample[:3])` -- memotong 3 KARAKTER pertama.
  '128px'[:3]='128'  '512px'[:3]='512'  TAPI  '1024px'[:3]='102'
Jadi menambahkan 1024 begitu saja akan diam-diam menjadikan subsample 102 (bukan
kelipatan 128 -> reshape pixel-boost pecah). Diperbaiki di 2 tempat menjadi
`int(str(upsample).lower().replace('px','').strip())`.

## Bagaimana subsample sebenarnya bekerja (diverifikasi numerik)
Namanya "subsample", BUKAN tiling. Crop pada ukuran N dipecah menjadi (N/128)^2
sub-gambar 128x128, masing-masing berisi WAJAH UTUH yang diambil dengan FASE
PIKSEL berbeda (sub#0 ambil piksel 0,2,4..., sub#1 ambil 1,3,5...). Terbukti:
selisih antar sub-image = 1 piksel, rentang nilai tiap sub mencakup gambar penuh,
dan implode->explode identik bit-per-bit.
Konsekuensi penting: inswapper SELALU melihat wajah lengkap berapa pun faktornya
-- tidak ada masalah "potongan terlalu ter-zoom". Menaikkan faktor aman secara
struktural.

## Biaya (kuadratik)
  128px  ->  1 inferensi inswapper per wajah per frame
  256px  ->  4
  512px  -> 16
  768px  -> 36   (2.25x dari 512)
  1024px -> 64   (4x dari 512)
Semua ukuran diverifikasi kelipatan 128 dan round-trip pixel-boost-nya identik.

## Ekspektasi kualitas -- jujur
Yang BERTAMBAH: resolusi keluaran swap, sehingga tekstur/detail halus dari wajah
TARGET lebih terjaga dan hasilnya tidak selembek 128px yang di-upscale.
Yang TIDAK bertambah: detail identitas dari SOURCE. inswapper mengenali source
lewat embedding 512-dimensi dan menghasilkan 128x128 per fase; menaikkan faktor
tidak menambah informasi source. Jadi "lebih true to the original source face"
tidak dijamin -- yang dijamin lebih tajam adalah hasil akhirnya.
Risiko pada faktor tinggi: tiap fase adalah decimation TANPA filter anti-alias,
dan tiap fase di-swap independen. Perbedaan kecil antar fase bisa muncul sebagai
pola halus/moire saat di-interleave kembali. Karena itu 768/1024 perlu diuji
visual, bukan diasumsikan lebih baik.

## Validasi
parsing 5 opsi benar; pixel-boost round-trip identik di 128/256/512/768/1024;
py_compile; build Gradio 5.9.1; tidak ada sisa `int(upsample[:3])`.
