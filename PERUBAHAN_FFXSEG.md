# Tambah FaceFusion XSeg occluder (xseg_1/2/3 + combined) ke Face Masking Engine

Model: FaceFusion's DeepFaceLab-XSeg occluder (arsitektur sama dgn xseg.onnx stok,
beda bobot). 256x256, /255, NHWC — interface identik dgn Mask_XSeg.

## Menu baru di "Face masking engine"
- FF XSeg 1  -> xseg_1.onnx
- FF XSeg 2  -> xseg_2.onnx
- FF XSeg 3  -> xseg_3.onnx
- FF XSeg (combined) -> ketiganya, di-interseksi (numpy.minimum face-mask =
  union occluder) untuk cakupan oklusi terluas (mode 'many' ala FaceFusion).

## File
- roop/processors/Mask_FFXSeg_base.py (logika bersama; io_binding per-call,
  thread-safe; intersect bila >1 model).
- roop/processors/Mask_FFXSeg1/2/3/Many.py (class tipis per varian).
- ProcessMgr.plugins: + mask_ffxseg1/2/3/_many.
- core.pre_check: unduh xseg_1/2 (models-3.1.0), xseg_3 (models-3.2.0) dari
  facefusion-assets (URL diverifikasi ada).
- faceswap_tab: dropdown + map_mask_engine + mapping on_preview_mask.

## Catatan
- Preprocessing disamakan PERSIS dgn Mask_XSeg stok (BGR, tanpa swap channel).
  FaceFusion mungkin pakai RGB; bila suatu saat mask FF terlihat salah/terbalik,
  cukup tambah cv2.cvtColor(BGR2RGB) sebelum normalisasi di Mask_FFXSeg_base.
- combined lebih berat (3x inferensi mask/wajah) -> render lebih lambat.

Validasi: py_compile semua + uji interseksi (combined=oklusi terluas) + build
faceswap_tab OK + URL download diverifikasi (403=ada vs 404=tidak).
