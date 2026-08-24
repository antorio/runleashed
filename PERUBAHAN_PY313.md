# requirements untuk Python 3.13 (Colab "Latest")

## Kenapa
Colab menaikkan runtime ke Python 3.13. Pin lama dibuat untuk 3.12 dan sebagian
tidak punya wheel 3.13, sehingga install gagal/compile-dari-source.

Catatan: Colab MASIH menyediakan runtime lama lewat menu Runtime > Change runtime
type > Runtime version (mis. 2026.04). Itu jalan tercepat kalau butuh cepat.
File ini untuk pindah ke 3.13 secara permanen, karena image lama tidak selamanya
tersedia.

## Perubahan pin
| paket | lama | baru | alasan |
|---|---|---|---|
| numpy | 1.26.4 | 2.1.3 | TIDAK ADA numpy 1.x dengan wheel cp313 |
| onnx | 1.16.1 | 1.18.0 | 1.17 pun belum punya cp313 |
| onnxruntime-gpu | 1.19.0 | 1.21.0 | cp313 baru ada sejak 1.20 |
| torch / torchvision | 2.5.1+cu124 / 0.20.1+cu124 | TIDAK DIPIN | lihat bawah |
| albucore | 0.0.16 | DIHAPUS | bentrok: albumentations modern menuntut >=0.0.24 |
| --extra-index-url pytorch | ada | DIHAPUS | tak perlu lagi tanpa pin cu124 |
Sisanya tetap (gradio 5.9.1, opencv-headless 4.10.0.84, psutil, tqdm, pydantic
2.10.6 -- semuanya pure-python atau abi3, jalan di 3.13).

## torch/torchvision tidak dipin -- disengaja
Colab sudah membawa torch build CUDA yang cocok dengan Python-nya sendiri.
Memin +cu124 memaksa unduhan ~900MB yang wheel-nya belum tentu ada untuk
interpreter yang sedang dipakai (persis kegagalan torchvision 0.20.1+cu124).
Di kode ini torch hanya dipakai oleh mask engine Clip2Seg (opsional) dan
pengecekan cuda.is_available(). Efek samping: install jauh lebih cepat.

## Yang SUDAH diverifikasi (di sandbox, Python 3.12 + numpy 2.1.3)
- insightface 0.7.3 BERHASIL dikompilasi dari source terhadap numpy 2.1.3
  (ini risiko terbesar, karena insightface tak punya wheel dan lahir di era numpy 1)
- Seluruh stack ter-import bersama: numpy 2.1.3, cv2 4.10, onnx 1.18, onnxruntime,
  insightface, gradio 5.9.1, pydantic 2.10.6, albumentations 2.0.8, skimage, scipy
- Interop numpy2 <-> cv2: resize + estimateAffinePartial2D OK
- Kode RunLeashed sendiri: import roop.*, align_crop, align_crop_robust,
  estimate_norm, landmark_68_to_5, LandmarkStabilizer, ExpressionSmoother,
  dan build UI Gradio 3 tab -- semua OK
- Ketersediaan wheel cp313 dicek satu per satu di PyPI untuk tiap pin baru

## Yang BELUM diverifikasi (jujur)
- Belum dijalankan di Python 3.13 sungguhan (sandbox saya 3.12). Yang dipastikan
  hanyalah setiap pin PUNYA wheel cp313; kompilasi insightface di 3.13 belum diuji.
- onnxruntime-gpu 1.21.0 belum diuji dengan GPU/CUDA nyata (sandbox tanpa GPU).
- torch dari PyPI (bukan +cu124) belum diuji di sini karena disk sandbox penuh.
- Peringatan skimage: `SimilarityTransform.estimate` deprecated (dihapus di
  skimage 2.2). Masih jalan sekarang; perlu diganti ke `from_estimate` sebelum
  skimage 2.2 mendarat.

## Tambahan: macOS
`onnxruntime-gpu` dilewati di darwin (tidak ada build CUDA untuk Mac), TAPI
sebelumnya tidak ada penggantinya sama sekali -- jadi setelah `pip install -r
requirements.txt` di Mac, onnxruntime tetap tidak terpasang dan run.py gagal.
Ditambahkan: `onnxruntime==1.25.1; sys_platform == 'darwin'` (versi terendah yang
punya wheel macOS cp313, tersedia arm64/universal2).
Selain itu torch/torchvision kini TANPA marker darwin (dulu dikecualikan di Mac),
jadi Mac ikut mendapatkannya dari PyPI.
config.yaml lokal sudah `provider: cpu`, cocok untuk Mac.

## Koreksi: onnxruntime macOS TIDAK boleh dipin
Saya sempat memin `onnxruntime==1.25.1` untuk darwin. Itu salah: saat mengecek
ketersediaan wheel macOS saya TIDAK membedakan arsitektur, hanya melihat ada
arm64/universal2. Plafonnya berbeda jauh:
    Apple Silicon (arm64) : sampai 1.29.0
    Intel Mac  (x86_64)   : berhenti di 1.23.2
Jadi pin apa pun akan gagal di salah satu arsitektur. Diubah jadi TANPA versi:
    onnxruntime; sys_platform == 'darwin'
pip akan mengambil yang tertinggi yang benar-benar ada di mesin tersebut.
Catatan: di Intel Mac, torch juga otomatis turun ke 2.2.2 (build x86_64 macOS
terakhir dari PyTorch) -- itu wajar dan cukup untuk mini-test di CPU.
