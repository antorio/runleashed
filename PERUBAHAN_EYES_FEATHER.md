# PERUBAHAN — feather Restore eyes diperlebar (batas mask tak terlihat saat jitter)

## Masalah
Batas mask mata terlalu kelihatan, terutama saat face/ER jitter. Feather lama:
`create_feathered_mask` dengan feather = min(30, box_w//15, box_h//15) -> untuk
box mata ~66x31 hanya ~2px = praktis tepi tajam.

## Referensi FaceFusion (dicek)
FF tidak pakai box tajam: region mask dari face-parser (per-piksel) + `face_mask_blur`
(default 0.3) yang mem-blur SELURUH tepi mask. Dokumen FF menyebut solusi untuk
"original face briefly returns / flickering" persis = NAIKKAN mask blur (~0.5).
Prinsipnya: andalkan tepi mask yang lembut & lebar, bukan bentuk kotak.

## Perubahan (mengambil prinsip FF tanpa menambah model parser)
1. Padding box mata dilonggarkan 0.25 -> 0.45 (memberi RUANG untuk feather memudar).
   Alis tetap tereksklusi: landmark mata duduk cukup di bawah alis, box top tetap
   di bawah garis alis (divalidasi: y0 > 270 saat alis di y=255).
2. `create_soft_eye_mask` BARU khusus mata: elips inti kecil (0.26x box) +
   GaussianBlur radius ~0.5x sisi terpendek. Hasil: tepi gradien lebar (band
   0.9->0.1 ~7-15px tergantung ukuran, vs ~2px sebelumnya), center tetap penuh
   (mata ter-restore 100%), border memudar ke 0 (tak ada cutoff keras di tepi box).
3. `apply_eyes_area` kini pakai soft mask ini (bukan create_feathered_mask mulut).
   Blend + color transfer tetap sama.
Efek: batas mata jauh lebih halus, tetap tak terlihat walau posisi bergeser saat
jitter (tepi lebar mentoleransi pergeseran sub-box).

## Lain-lain
- Info-text pada toggle "Restore eyes" dihapus (sesuai permintaan).
- Bonus dari sesi sebelumnya tetap: guard divide-by-zero di apply_color_transfer.

## Validasi
Soft mask di 4 ukuran box: center penuh, border 0, edge band melebar dengan ukuran.
Brow excluded @0.45 padding. Mata ter-paste (channel warna naik). Graceful tanpa
kps/landmark. py_compile + build Gradio 5.9.1 OK.

## Catatan tuning
Jika masih kurang/kelebihan di L4: dua angka mudah disetel di create_soft_eye_mask
(elips 0.26, blur 0.5) dan padding 0.45 di create_eyes_masks.
