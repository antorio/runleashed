# Reorder occlusion mask → setelah Expression Restorer

Sebelumnya: swap → mask(oklusi) → ER → enhancer. Akibatnya ER (LivePortrait)
mewarp occluder (tangan/rambut) yang baru di-restore → occluder ikut terdistorsi.

Sekarang (default): swap → ER → mask(oklusi) → enhancer.
Opsional (toggle): swap → ER → enhancer → mask(oklusi) — occluder tidak ikut
disentuh/di-sharpen enhancer (tetap piksel asli).

## Perubahan
- `roop/core.py` `get_processing_plugins`: ER ditaruh tepat setelah swap; mask
  setelah ER. Posisi mask vs enhancer dikontrol `mask_after_enhancer`.
- `roop/ProcessMgr.py` `process_face` (branch 'mask'): bila `enhanced_frame`
  sudah ada (kasus mask-after-enhancer), mask juga diterapkan ke `enhanced_frame`
  — karena paste_upscale mem-blend enhanced+fake; tanpa ini occluder encer/hilang.
- `roop/globals.py`: `mask_after_enhancer = False` (default).
- Settings → "Face mask & paste-back — live tuning": checkbox "Occlusion mask
  after enhancer" (live, tanpa restart).

## Urutan terverifikasi
default            : faceswap → expression_restorer → mask → enhancer
mask_after_enhancer: faceswap → expression_restorer → enhancer → mask
tanpa ER           : faceswap → mask → enhancer
tanpa enhancer     : faceswap → expression_restorer → mask

Validasi: py_compile + build penuh gradio 5.9.1 OK.

## Hotfix (mask-after-enhancer broadcast error)
`process_mask` me-resize `frame` (sumber occluder asli) agar match resolusi
`target`. Sebelumnya error `(512,512,1) vs (256,256,3)` saat mask diterapkan ke
`enhanced_frame` (512) padahal aligned crop 256. Kasus normal (256 vs 256) tak
berubah.

## Fix run lokal (non-Colab)
`prepare_environment` kini fallback ke `./output` bila `output_folder` (default
Colab `/content/drive/...`) tak bisa dibuat di mesin lokal (Mac/Win) -- tidak lagi
crash `Read-only file system: /content`. `allowed_paths` juga otomatis menyertakan
output dir + cwd lokal. Perilaku Colab tidak berubah.
