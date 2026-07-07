# PERUBAHAN — semua model dari satu repo HuggingFace milik user

## Kenapa
Download model saat pertama kali sering stuck DARI COLAB (tapi jalan dari browser).
Penyebab: 9 dari 15 model diunduh dari GitHub release assets
(facefusion-assets) yang redirect ke S3 dan sering rate-limit/timeout ke IP
datacenter Colab. HuggingFace CDN (CloudFront) ramah ke IP cloud -- itu sebabnya
yang dari HF jarang bermasalah.

## Perubahan
`pre_check()` di core.py ditulis ulang:
- Semua 15 model kini diunduh dari SATU repo: `antorio/runleashed-models`
  (di-mirror ke sana dari countfloyd/deepfake + facefusion-assets).
- Memakai `hf_hub_download` (resume + retry otomatis) -> koneksi putus tidak
  mengulang dari nol. Fallback ke direct `resolve/main` URL (tetap repo HF user)
  bila hf_hub_download gagal import/jalan.
- Penempatan file lokal tetap sama persis: root ../models/, kecuali
  liveportrait/* (4), CLIP/rd64-uni-refined.pth, CodeFormer/CodeFormerv0.1.onnx.
- Skip-if-exists dipertahankan (tidak mengunduh ulang yang sudah ada).
- `huggingface_hub` ditambahkan eksplisit ke requirements.txt (insightface juga
  menariknya, tapi eksplisit lebih aman).

## Yang harus diupload user ke repo HF (root, flat, nama apa adanya) -- 15 file:
inswapper_128.onnx, GFPGANv1.4.onnx, restoreformer_plus_plus.onnx, xseg.onnx,
xseg_1.onnx, xseg_2.onnx, xseg_3.onnx, bisenet_resnet_34.onnx, 2dfan4.onnx,
live_portrait_feature_extractor.onnx, live_portrait_motion_extractor.onnx,
live_portrait_generator.onnx, live_portrait_stitcher.onnx, rd64-uni-refined.pth,
CodeFormerv0.1.onnx

## Validasi
py_compile OK. Test pre_check (hf_hub_download di-mock): 15 file ke subdir benar,
run kedua 0 download, fallback -> resolve URL repo user. Semua lolos.

## Catatan
Total ~2.5GB, repo publik HF = gratis. reswapper TIDAK ada (sudah dicabut).
