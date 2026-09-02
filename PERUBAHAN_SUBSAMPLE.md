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

## HOTFIX — 768/1024 menghasilkan LAYAR HITAM (analisis awal saya tidak menyeluruh)
Saya memverifikasi implode/explode secara TERISOLASI dan menyimpulkan "aman
secara struktural", tanpa memeriksa apakah bagian LAIN pipeline punya asumsi
ukuran. Ada dua, dan keduanya di jalur paste-back.

### Sebab 1 (penyebab layar hitam): pembagian INTEGER
    scale_factor = int(upscale / orig_width)      # upscale = 512
      128 -> 4 | 256 -> 2 | 512 -> 1 | 768 -> 0 | 1024 -> 0
Lalu `paste_upscale` melakukan `M_scale = M * scale_factor`. Dengan 0, SELURUH
matriks affine jadi nol, dan `invertAffineTransform` atas matriks nol
menghasilkan transformasi degenerate -> paste menyebar ke seluruh frame.
Terukur: di 768 perubahan mencakup x[1..1278] y[1..718] (seluruh gambar),
sedangkan di 512 hanya x[468..731] (kotak wajah).
FIX: `scale_factor = float(upscale) / float(orig_width)`.

### Sebab 2 (lebih dalam): 768/1024 diturunkan lagi ke 512
`upscale = 512` di-hardcode dan `fake_frame` selalu di-resize ke situ. Jadi
bahkan setelah sebab 1 diperbaiki, hasil 768/1024 akan di-DOWNSCALE ke 512
sebelum ditempel -- seluruh komputasi tambahannya terbuang, nol manfaat.
FIX: `upscale = max(512, orig_width)`. 512 tetap menjadi LANTAI (crop kecil
128/256 tetap diperbesar dulu agar warp balik mulus, seperti semula) tapi bukan
lagi PLAFON.

### Nilai sebelum vs sesudah
    subsample   upscale|scale_factor  LAMA -> BARU
      128         512|4   ->   512|4.00     (identik)
      256         512|2   ->   512|2.00     (identik)
      512         512|1   ->   512|1.00     (identik)
      768         512|0   ->   768|1.00     (dulu rusak)
     1024         512|0   ->  1024|1.00     (dulu rusak)
128/256/512 tidak berubah sama sekali.

### Validasi
process_face dengan swapper tiruan di 128/256/512/768/1024: kelimanya menempel
di kotak wajah yang sama (x[468..732] y[212..475]), tidak ada lagi penyebaran ke
seluruh frame; paste_upscale + geometri matte identik antar ukuran; implode/
explode round-trip 0 selisih; py_compile + build Gradio.

## HOTFIX 2 — layar hitam TERNYATA hanya saat ENHANCER aktif
Bug yang sama (pembagian integer) ada di TIGA tempat lain yang saya lewatkan:
`Enhance_GFPGAN.py`, `Enhance_CodeFormer.py`, `Enhance_RestoreFormerPPlus.py`
semuanya menghitung
    scale_factor = int(result.shape[1] / input_size)
Enhancer SELALU mengeluarkan 512, sedangkan input_size = ukuran subsample:
    128->4  256->2  512->1  768->0  1024->0
Nilai 0 itu masuk ke paste_upscale sebagai `M_scale = M * 0` -> matriks affine nol
-> invers degenerate -> layar hitam. Karena itu gejalanya hanya muncul saat
enhancer ON. Ketiganya diperbaiki jadi pembagian float.

## HOTFIX 3 — blending enhancer meleset di 768/1024
`IM` dikalibrasi untuk ukuran `upsk_face`. Dengan enhancer, upsk_face = 512
sedangkan fake_face (swap mentah) = 768/1024; me-warp dua ukuran berbeda dengan
matriks yang sama membuat wajah non-enhanced ter-zoom dan blending meleset.
FIX: fake_face di-resize ke ukuran upsk_face sebelum warp.
Terverifikasi: subsample 512/768/1024 + enhancer -> kotak paste sama
(x[468..731]) dan nilai blend benar (210 & 90 -> 150).

## TEMUAN PENTING untuk pemakaian (jawaban #1)
Ketiga enhancer adalah model 512-in/512-out (`cv2.resize(temp_frame,(512,512))`).
Jadi dengan ENHANCER AKTIF, subsample 768/1024 TIDAK menambah resolusi akhir --
hasil swap yang 768/1024 justru diturunkan ke 512 oleh enhancer. Biaya
komputasinya (36x/64x inferensi swap) hampir seluruhnya terbuang.
  enhancer OFF -> 768/1024 benar-benar menaikkan resolusi keluaran
  enhancer ON  -> plafon efektif tetap 512; pakai 512 saja

## Concurrency preview dinaikkan
6 event preview: `concurrency_limit` 1 -> 3 (concurrency_id "preview" dan
trigger_mode "always_last" tetap). Batas 1 menyerialkan setiap perubahan setting
sehingga preview terasa lambat; batas 3 mengizinkan tumpang tindih terbatas tapi
tetap mencegah 6 swap GPU berjalan sekaligus (penyebab "error connecting" dulu).
Bila "error connecting" muncul lagi, turunkan ke 2.

## Enhancer selalu 512 -- terbukti dari kode
`Enhance_*.Run()` baris pertama: `input_size = temp_frame.shape[1]` lalu
`cv2.resize(temp_frame, (512,512))`. Modelnya 512-in/512-out, jadi APA PUN
ukuran masuknya, keluarnya 512:
    subsample 128  -> enhancer menaikkan 128 -> 512   (menguntungkan)
    subsample 256  -> menaikkan 256 -> 512            (menguntungkan)
    subsample 512  -> pas, tanpa resize
    subsample 768  -> MENURUNKAN 768 -> 512           (detail dibuang)
    subsample 1024 -> MENURUNKAN 1024 -> 512          (detail dibuang)
Jadi 768/1024 + enhancer BUKAN tidak bisa (sudah tidak layar hitam lagi), tapi
sia-sia: biaya swap 36x/64x dibayar lalu hasilnya diturunkan ke 512 oleh enhancer.
