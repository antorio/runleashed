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
- Lokal (Intel Mac, tanpa CUDA): `./runMacOS.sh` → `.venv` Python 3.12 (3.14 tidak bisa: onnxruntime/insightface tidak punya build Intel), install ulang otomatis bila `requirements.txt` berubah, model ±2,5 GB diunduh sekali ke `models/` (langsung, tanpa salinan di cache HF), buka http://127.0.0.1:7860 (`server_share: false` di `config.yaml`; Colab memakai `config_colab.yaml`). CPU saja (onnxruntime 1.23.2). PyTorch tidak dipasang di Intel Mac (tidak ada build >2.2 / NumPy 2) → torch opsional di `core.py`/`utilities.py`, Clip2Seg tidak tersedia (peringatan). Untuk UI, faceset, preview; render video tetap di Colab.
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
- **Toggle "Extend swap to the chin"** (`mask_bottom_to_chin`, Advanced Masking, default OFF, untuk A/B user). Crop arcface menaruh dagu di 87–92% tinggi crop, dan Erosion/Blur size bekerja di 4 sisi kotak, sehingga pita erosi+feather sisi bawah jatuh di dagu (Erosion 2 / Blur 20: dagu panjang hanya 21% ter-swap; Erosion 3: ±0%). Itulah "dagu target tembus + garis sambungan" yang dilaporkan user, termasuk di pose frontal. ON: tepi bawah tidak di-erode, memudar di 6% terakhir crop → dagu 99–100% ter-swap di semua kombinasi, 0 di bawah crop, tepi atas/samping tetap. OFF bit-identik dengan kode lama (300/300).

- **Koreksi (uji user di L4):** toggle dagu hampir tidak terlihat efeknya karena di setting default (Erosion 1 / Blur 20) dagu wajah biasa sudah 97–100% tertutup. "Dagu target" yang terlihat = inswapper mempertahankan BENTUK rahang/dagu target, bukan masalah mask.
- **Preview tidak menyentuh render yang sedang jalan** (`is_processing` → tampilkan frame asli). Render juga memakai salinan daftar wajah sendiri.
- **Model dipakai ulang antar preview/render** (pool `_idle_processors`). Skenario uji: 10 → 4 kali pembuatan model. Keluar aplikasi = semua dilepas. Clip2Seg tidak lagi dimuat ulang tiap init.
- Retry rotated dilewati hanya bila multi-angle sudah mencoba 0/90/180/270 dan tidak menemukan wajah sama sekali (hasil sama, 12 → 4 deteksi). Mode Selected/`off` tetap retry.
- Salinan frame untuk "Use last swapped" hanya dibuat di mode itu. Komentar default yang salah dibetulkan.
- Uji di Colab: `runleashed_colab.ipynb` (param `BRANCH`); bandingkan output dengan `tools/compare_renders.py`.


## Tahap 2 — kemiripan source + UX (24 Sep, branch `perf/phase1`, menunggu A/B di L4)
Semua default OFF/0 → **output identik dengan HEAD sebelumnya** (process_face penuh 24/24 kasus acak, paste 120/120 vs `main`, urutan smoothing tetap).
- **Face shape from source** (`face_shape_strength`, slider di Face selection; `roop/face_shape.py`): bentuk dibandingkan dalam **3D** (1k3d68, Umeyama pada hidung + sudut mata) → pose tidak terbaca sebagai bentuk (sintetis: 0,000 px di yaw ±40 / pitch ±30; versi 2D sebelumnya salah 17–54 px, dibuang). Yang di-warp adalah **frame** di sekitar wajah (TPS; alis/mata/hidung/bibir dipin ≤0,25 px → alignment tetap), sebelum crop; wajah hasil swap ditempel ke frame yang sudah di-warp. Dagu relatif ke bibir bawah (bukaan mulut target tetap). Maks geser 12% lebar rahang, cek lipatan (Jacobian ≥0,3), pudar di yaw 30–45 / pitch 20–35 dan sisi jauh wajah menoleh. Bentuk target dirata-rata per track (`TargetTracks`, video in-memory): jitter geser 5,8 → 0,3 px. Saat aktif, 1k3d68 selalu diminta (juga dengan 2dfan4; salinannya `landmark_3d_68_buffalo`). ±40–50 ms/wajah di CPU Mac. Hipotesis: belahan dagu = tekstur, belum tentu hilang.
- **Identity strength** (`identity_strength`): `e = norm(src + w·(src − tgt))` sebelum emap; tgt = embedding target dirata-rata per track (reset bila cos < 0,5 = ganti orang).
- **ER "Keep source face structure"** (`expression_keep_structure`): kp LivePortrait 0,4,5,8,9 tetap dari hasil swap (seperti FaceFusion). Hipotesis: kp ini ikut menarik struktur wajah ke target.
- **Face-aligned mask edges** (`mask_face_aligned`): matte miring vs tegak beda 27–35% (lama) → 0,05%; juga lebih cepat dari jalur lama.
- `TargetTracks` terpisah dari `LandmarkStabilizer` (stabilizer tidak diubah sama sekali).
- **Frame preview cepat:** saat video target dipilih, salinan GOP 8 dibuat di latar (ffmpeg, NVENC bila ada, `nice`), dipakai hanya bila jumlah frame sama dan 3 frame sampel cocok (VFR ditolak). Frame baru 4–8× lebih cepat (4K 1,4 dtk → 0,23 dtk; HEVC ala iPhone 0,64 → 0,06 dtk). Pemilih wajah / Face Manager / render tetap membaca file asli (`exact=True`). Flag `preview_seek_copy`. Log `[preview] frame N: load / swap / total ms`.
- **Analyser:** lock (fix race FileExistsError buffalo_l), warm-up di thread latar saat start, rebuild bila Force CPU berubah.
- **Colab:** proxy Colab di tab baru memang rusak (dimatikan Colab); notebook kini menjalankan `run.py` di latar dan menampilkan UI di iframe output sel (`serve_kernel_port_as_iframe`). gradio.live tetap aktif (server di Oregon). Cloudflare quick tunnel ditolak: GET/SSE di-buffer → progress & tombol Stop baru muncul setelah render selesai.
- Paste: campuran enhancer (`addWeighted`) dihitung sebelum dipotong ke ROI (versi phase-1 sempat ±1 level).
- **Face Management dirombak** (25 Sep, atas keluhan user: tombol Remove tidak jelas, tabel + nomor yang berubah saat foto dihapus = tidak bisa dipakai). Alur 1 Tambah → 2 Periksa → 3 Simpan: foto dicek otomatis saat masuk; status tertulis di foto itu sendiri (✓ / ⚠ alasan utama +n); klik foto → panel kanan (foto besar, penjelasan, pasangan duplikat, tombol *Remove this photo* / *Keep anyway*); setelah hapus, foto berikutnya otomatis terpilih (selected_index dikendalikan server). Filter *Needs attention*, *Remove N flagged*, Undo untuk semua penghapusan (termasuk *Start over* dengan konfirmasi dan membuka .fsz), simpan dengan nama (tidak menimpa). Tanpa nomor foto dan tanpa tabel. Diuji di Gradio 5.9.1 asli (lokal, port 7861): tambah, pilih, hapus, keep, undo, remove flagged, start over, simpan, buka .fsz, wajah dari video.
- **Check faceset / Remove flagged** (`roop/faceset_check.py`): per foto kemiripan leave-one-out, flag `other person?` (cos < 0,40 ke median identitas, butuh ≥3 foto), `unlike the others`, duplikat (cos > 0,95; foto terbaik grup dipertahankan), `small` (<128 px di foto asli), `blurry` (ketajaman < 40% median); info yaw/pitch/mulut/pemakaian Face shape. Ambang = heuristik. Remove flagged tidak pernah mengosongkan daftar. Saat .fsz dimuat di tab swap: ringkasan di console + gr.Info bila ada flag (hanya membaca, embedding rata-rata tidak berubah).
- **Face shape memilih foto source** (`pick_shape_photos`, dipakai juga oleh Check): hanya orang yang sama (cos ≥ 0,40 ke median), yaw ≤25°, pitch ≤20°, mulut tertutup (celah bibir dalam / jarak sudut mata luar ≤0,08); bila < 3, diisi foto terdekat. Log `[face-shape] source shape from K of N photos`.
- Riset faceset (literatur ArcFace + kode tool lain): rata-rata embedding jenuh sekitar ~50 foto; komposisi (usia/makeup/filter sama, variasi sudut) lebih penting daripada jumlah. Belum ada A/B inswapper.

## Tujuan kualitas: kemiripan source
- Source user: faceset .fsz; ER biasanya ON; mask DFL XSeg atau FF XSeg 3; swap steps jarang dipakai (wajah terlihat makin mengecil tiap step).
- Riset (kode + literatur): embedding ArcFace diambil dari crop 112 = crop128[0:112, 8:120]; dagu (lm 8) di y≈121 → embedding source praktis tidak memuat dagu/rahang. Semua trik latent (VisoMaster/Rope/Fusion) hanya memengaruhi wajah bagian dalam; tidak ada swapper one-shot yang memindahkan bentuk rahang. Karena itu bentuk = warp geometri (di atas).
- Rute ke ~100% termasuk bentuk: model per-identitas (DeepFaceLab SAEHD/AMP → .dfm): butuh ribuan gambar source + training ±1 hari+ GPU. FAQ Colab melarang pembuatan deepfake di semua runtime.

## Masalah diketahui (belum diperbaiki)
1. **Keypoint ER** (dibandingkan ke FaceFusion): default mentransfer kp 0,3,4,5,7,8,9,10; toggle "Keep source face structure" menahan [0,4,5,8,9]. FaceFusion juga memakai batas per-keypoint `EXPRESSION_MIN/MAX` (di sini hanya clamp skalar, off). Hipotesis: kp itu sumber drift kepala yang ditambal pose-lock. Perlu A/B visual.
2. Analyser menjalankan landmark + recognition + genderage untuk SEMUA wajah tiap frame, walau tidak dipakai (dari kode). Ditunda: butuh model asli untuk verifikasi.
3. Jalur "Extract Frames" tidak memakai temporal smoothing maupun `TargetTracks` (face shape / identity dihitung per frame), jadi hasilnya beda dengan in-memory (dari kode).
4. **Erosi matte bergantung rotasi wajah** (diukur) di jalur default: `blur_area` mengerosi di ruang frame dengan kernel kotak lurus → wajah miring 25° tererosi ±75% lebih dalam. Solusi = toggle "Face-aligned mask edges" (Tahap 2); jadikan default hanya setelah A/B user.
5. Kecil: "Wait for key press" memanggil `input()` (tidak berguna di Colab). `gr.Error` "No Target Face" tidak di-raise, jadi Start diam saja. `tools_jitter_probe.py` rusak (import `align_conditioner` yang sudah dihapus). Setting live (slider/checkbox) yang diubah di tengah render tetap langsung memengaruhi render.
