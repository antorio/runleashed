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

## Masalah diketahui (dicek 23 Sep 2026, belum diperbaiki)
1. **Stall 10 dtk** (disimulasikan dengan kelas asli): frame tanpa wajah tidak pernah masuk `FrameSequencer.in_order`, jadi frame berikutnya menunggu timeout 10 dtk. Terjadi pada render in-memory + smoothing ON (default). 200 frame: 0,7 dtk → 10,7 dtk per satu kejadian wajah hilang.
2. **Paste seluruh frame** (diukur): `paste_upscale`/`blur_area` bekerja di seluruh frame. Versi ROI 4,4–6,4× lebih cepat, selisih maks 1 level warna (CPU 1 thread: 1080p 145→32 ms/wajah, 4K 641→144 ms).
3. **Preview saat render** (dari kode): `live_swap` memakai `ProcessMgr` global yang sama, jadi `initialize()` mengganti options/processors/sequencer/stabilizer render yang sedang jalan. Preview tidak mengecek `is_processing`.
4. **Keypoint ER** (dibandingkan ke FaceFusion): kp 0,3,4,5,7,8,9,10 selalu ditransfer. FaceFusion tidak pernah mentransfer [0,4,5,8,9] dan memakai batas per-keypoint `EXPRESSION_MIN/MAX` (di sini hanya clamp skalar, off). Hipotesis: ini sumber drift kepala yang ditambal pose-lock. Perlu A/B visual.
5. Retry rotated (default no-face action) mengulang semua pass multi-angle: ±12 deteksi per frame tanpa wajah (dari kode).
6. Sesi ONNX dilepas di awal dan akhir tiap render, lalu dibangun ulang (+ pencarian cuDNN EXHAUSTIVE) tiap render dan preview pertama sesudahnya (dari kode; overhead belum diukur di L4).
7. Analyser menjalankan landmark + recognition + genderage untuk SEMUA wajah tiap frame, walau tidak dipakai (dari kode).
8. Jalur "Extract Frames" tidak memakai temporal smoothing, jadi hasilnya beda dengan in-memory (dari kode).
9. Kecil: `processorname` Clip2Seg tidak cocok sehingga model dimuat ulang tiap init (warisan upstream). "Wait for key press" memanggil `input()` (tidak berguna di Colab). `gr.Error` "No Target Face" tidak di-raise, jadi Start diam saja. `tools_jitter_probe.py` rusak (import `align_conditioner` yang sudah dihapus). Komentar usang: `core.py` menyebut `mask_after_enhancer` default off (nyatanya ON); docstring `tuned_execution_providers` menyebut HEURISTIC (nyatanya EXHAUSTIVE).
