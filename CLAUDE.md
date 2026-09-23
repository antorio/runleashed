# RunLeashed — catatan proyek (satu-satunya)

> Menggantikan semua `PERUBAHAN_*.md` dan `RUNLEASHED_HANDOFF_v2.md` (masih ada di git history s/d `32eb825`).
> Sebagian catatan lama keliru atau usang. **Kode adalah sumber kebenaran.** Isi di bawah dicek ke kode per 2026-09-23.

## Cara kerja
- Bahasa Indonesia, ringkas. Kalau salah, akui dengan jujur.
- Pisahkan **fakta terverifikasi** (kode/pengukuran) dari **hipotesis**. Jangan klaim perbaikan hanya dari simulasi.
- Perubahan yang memengaruhi visual (alignment, mask, ER, blending) wajib A/B visual di klip pendek dulu. Metrik agregat hanya untuk mendeteksi regresi.
- Fitur opsional jangan dijadikan default tanpa izin (enhancer tetap opsional).
- Commit/push hanya bila diminta. Uji nyata = push ke GitHub → Colab L4.

## Tujuan & penilaian user
- Ukuran kualitas utama: **kemiripan dengan source face**. Sekarang ±60/40 (source/target); target 90/10, idealnya 100/0.
- **FaceFusion bukan patokan hasil.** Menurut user swap-nya seperti blend 50/50, lebih lambat, dan UI-nya berantakan. Kodenya boleh dipakai sebagai referensi teknis, bukan standar kualitas.
- Expression Restorer di app ini (dengan setting tertentu) lebih terlihat efeknya dibanding FaceFusion.
- 2dfan4 bukan "yang terbaik"; kelemahannya sama dengan landmarker lain, hanya beda bentuk.
- Keluhan terbesar: alignment OFF / distorsi berlebihan saat wajah **miring atau mendongak** (±20–30% dari footage; footage beragam, kadang satu video penuh).
- Jitter: fix urutan frame (`FrameSequencer`, 26 Agu) menurut uji awal user sudah banyak menyelesaikan jitter (jitter parah awalnya dari tweak Claude sebelumnya). **Jangan dirusak**; perubahan di sekitarnya harus menjaga urutan smoothing persis.

## Lingkungan
- Fork roop-unleashed 4.4.x → v5.0.0. Baseline asli untuk diff: `../runleashed-440`. Referensi: `../facefusion` (3.4.1).
- Render nyata di **Colab L4**: notebook clone `antorio/runleashed` → `mv config_colab.yaml config.yaml` → `pip install -r requirements.txt` → `python run.py` (UI lewat gradio.live). Python 3.13, gradio 5.9.1, onnxruntime-gpu 1.21, insightface 0.7.3 (dikompilasi saat install).
- Lokal: Intel Mac (tanpa CUDA), tanpa folder `models/` → tidak bisa render; cukup untuk tes logika numpy/cv2.
- Model: 15 file dari HF `antorio/runleashed-models` (`core.pre_check`); `buffalo_l` diunduh insightface dari GitHub.

## Arsitektur singkat
- `ui/tabs/faceswap_tab.py` → `roop/core.py` (`batch_process_regular`, `live_swap`, `get_processing_plugins`) → `roop/ProcessMgr.py` (mesin).
- Per wajah (`process_face`): deteksi `face_util.get_*_faces_multi` (multi-angle `fallback`) → [urut frame] refine 2dfan4 (opsional) + `LandmarkStabilizer` (hanya render video in-memory) → align 68→5 (`landmark_68_to_5`) bila Landmark alignment ON, selain itu kps detektor → inswapper_128 + pixel boost (subsample N = (N/128)² inferensi) → ER LivePortrait (opsional) → mask oklusi ↔ enhancer (urutan via `mask_after_enhancer`, default ON = mask terakhir) → `paste_upscale` (matte kotak di-erode + blur; `mask_erosion_iterations`, `mask_blur_size` global) → restore mouth/eyes.
- Setelan: `roop/globals.py` = knob live (UI menulis langsung, **tidak tersimpan**, kembali ke default saat restart). `config.yaml` + `settings.py` = tersimpan (Apply Settings).

## Fakta terverifikasi
- inswapper hanya memakai embedding 512-d dari source; resolusi foto source tidak menambah detail.
- Enhancer (GFPGAN/CodeFormer/RF++) selalu 512→512: dengan enhancer, subsample >512 sia-sia.
- `ransacReprojThreshold=100` = praktis least-squares (sama dengan FaceFusion). Ambang kecil membuat inlier bolak-balik → jitter (sudah dicoba, di-rollback).
- Mencampur atau berganti basis 68→5 ↔ kps per frame = geometri tak valid / lompatan (crossfade & gate v2 di-rollback 3 Agu). Sanity gate kini opt-in, default OFF.
- One-Euro pada matriks M: dicoba lalu dicabut (4 Jul), micro-jitter menurut A/B user.
- insightface menjalankan tiap modul analisis secara independen; set modul tidak mengubah `landmark_2d_106`. "ER beda antar mode" dulu disebabkan bug mode selected yang memakai source berbeda (fixed 1 Agu).
- Generator LP (ekspor FaceFusion): input `source` = kp tujuan (ekspresi target), `target` = kp saat ini → `expression_invert_direction=True` benar.
- ER Strength dan Expression power adalah pengali yang sama (amt = factor × power).
- `angle0_bonus` hanya berpengaruh di multi-angle `always`; di `fallback`, deteksi sudut-0 dan rotasi tidak pernah bersaing.

## Tahap 1 — branch `perf/phase1` (menunggu uji di L4)
Perbaikan performa/bug yang **tidak mengubah gambar**; semuanya diuji tanpa GPU dengan `ProcessMgr` asli (lama dari `main` vs baru) + detektor/swapper tiruan:
- **Stall 10 dtk dihapus.** Frame tanpa wajah kini tetap mengambil gilirannya (`FrameSequencer.pass_turn`); frame yang terlambat tidak menunggu. Urutan dijaga untuk setiap frame video, termasuk saat smoothing OFF. Jalur render in-memory penuh, 160 frame + 4 wajah hilang: 30,4 dtk → 0,4 dtk. Landmark ter-smooth identik dengan smoothing berurutan ideal. Kode lama kadang juga memasukkan frame ke smoothing tidak berurutan setelah timeout (race).
- **Paste hanya di area wajah.** Warp dihitung dari titik (0,0) frame lalu dipotong → **bit-identik** dengan kode lama di OpenCV 4.10 dan 5.0 (ratusan kasus acak termasuk offset, erosi/blur, enhancer, color transfer, overlay, wajah di tepi/di luar frame). Median 2–2,5× lebih cepat per wajah (wajah 15–40% tinggi frame, posisi acak; tidak pernah lebih lambat). Varian yang menggeser matriks ke ROI TIDAK identik di OpenCV 4.10 (fixed-point bergantung posisi absolut); pada gambar mirip foto selisihnya maks 4 level di ≤0,21% piksel, jadi opsi itu ada kalau butuh ±1,5× lagi.
- **Cache frame video** (`roop/capturer.py`, dipakai preview, pemilih wajah, Face Manager): frame yang sama lagi / maju 1 frame tidak lagi seek + decode ulang dari keyframe. 1080p: 200 → 0 ms / 193 → 5 ms; 4K: 883 → 6 ms / 863 → 17 ms; lompat acak tetap perlu seek. 0 frame berbeda dari loader lama di 8 jenis video (MP4/MKV/WebM/MOV, H.264/HEVC/VP9, VFR) + uji 4 thread paralel.
- OpenCV 5.0 tidak dipakai: tidak lebih cepat untuk operasi app ini dan mengubah output warp/resize (maks 3–4 level di 19–50% piksel).
- **Preview tidak menyentuh render yang sedang jalan** (`is_processing` → tampilkan frame asli). Render juga memakai salinan daftar wajah sendiri.
- **Model dipakai ulang antar preview/render** (pool `_idle_processors`). Skenario uji: 10 → 4 kali pembuatan model. Keluar aplikasi = semua dilepas. Clip2Seg tidak lagi dimuat ulang tiap init.
- Retry rotated dilewati hanya bila multi-angle sudah mencoba 0/90/180/270 dan tidak menemukan wajah sama sekali (hasil sama, 12 → 4 deteksi). Mode Selected/`off` tetap retry.
- Salinan frame untuk "Use last swapped" hanya dibuat di mode itu. Komentar default yang salah dibetulkan.
- Uji di Colab: `runleashed_colab.ipynb` (param `BRANCH`); bandingkan output dengan `tools/compare_renders.py`.

## Masalah diketahui (belum diperbaiki)
1. **Keypoint ER** (dibandingkan ke FaceFusion): kp 0,3,4,5,7,8,9,10 selalu ditransfer. FaceFusion tidak pernah mentransfer [0,4,5,8,9] dan memakai batas per-keypoint `EXPRESSION_MIN/MAX` (di sini hanya clamp skalar, off). Hipotesis: ini sumber drift kepala yang ditambal pose-lock. Perlu A/B visual.
2. Analyser menjalankan landmark + recognition + genderage untuk SEMUA wajah tiap frame, walau tidak dipakai (dari kode). Ditunda: butuh model asli untuk verifikasi.
3. Jalur "Extract Frames" tidak memakai temporal smoothing, jadi hasilnya beda dengan in-memory (dari kode).
4. Kecil: "Wait for key press" memanggil `input()` (tidak berguna di Colab). `gr.Error` "No Target Face" tidak di-raise, jadi Start diam saja. `tools_jitter_probe.py` rusak (import `align_conditioner` yang sudah dihapus). Setting live (slider/checkbox) yang diubah di tengah render tetap langsung memengaruhi render.
