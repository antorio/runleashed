# RUNLEASHED — DOKUMEN HANDOFF v2 (LENGKAP)
*(Embed file ini di awal thread baru sebagai basis. Berisi konteks proyek, semua
yang sudah dikerjakan sampai versi 5.0.0, fakta terverifikasi, kondisi terkini,
dan langkah berikutnya. Ini menggantikan handoff v1; bagian "PEKERJAAN LAMA"
meringkas v1, sisanya pekerjaan baru.)*

---

## 1. PROYEK
**"RunLeashed"** = fork dari roop-unleashed v4.4.1. Inti: InsightFace `buffalo_l`
analyser + `inswapper_128.onnx` (face-swap), GUI Gradio. Versi sekarang **5.0.0**.

**Lingkungan eksekusi (punya user):** Google Colab **L4 GPU 24GB**, CUDA 12.4,
`onnxruntime-gpu 1.19.0`, `insightface 0.7.3`, PyTorch 2.5.1+cu124, gradio 5.9.1.
**Juga jalan di lokal Mac** sekarang (lihat §6.J). Notebook: `!git clone
github.com/antorio/runleashed` → `!mv config_colab.yaml config.yaml` → `!python run.py`
(Gradio `share=True`, tunnel gradio.live).

**Gaya kerja & preferensi user (PENTING, pertahankan):**
- Komunikasi **Bahasa Indonesia**, jawaban **ringkas**.
- User **perfeksionis** ("mencari kesempurnaan"). Bersandar pada **LivePortrait asli
  (KwaiVGI)**. (Model LP yang dipakai memang dari KwaiVGI; FaceFusion cuma meng-host
  file ONNX-nya.)
- Asisten **tidak punya GPU/model** di sandbox → validasi matematika via numpy/cv2
  (`pip install --break-system-packages`). User menguji tiap build di L4-nya lalu
  mengirim balik **console log**. Pola kerja: **diagnostik dulu** (instrumentasi,
  ukur, baru perbaiki) — berkali-kali jadi kunci menemukan akar masalah.
- Pola tiap fix: edit → validasi math (numpy/cv2) → `py_compile` → build-test stub
  Gradio → bersihkan cache → zip → `present_files` → user uji di L4 + kirim log.
- **Akui kesalahan jujur.** Jangan jadikan menu opsional user sebagai default tanpa
  izin. Enhancer tetap opsional.

---

## 2. STRUKTUR FILE PENTING (di `runleashed-main/`)
- `roop/ProcessMgr.py` — `process_face`, `swap_faces`, `paste_upscale`,
  `_build_hull_matte`, `process_expression`, `process_mask`. Map plugin key→class.
- `roop/core.py` — `pre_check` (unduh model ~line 130-150), `get_processing_plugins`
  (urutan pipeline), `run()` (line ~398 load `Settings('config.yaml')`).
- `roop/globals.py` — semua toggle/default (lihat §5).
- `roop/utilities.py` — `tuned_execution_providers()` (setting hemat-memori ORT),
  `convert_to_gradio_preview`, dll.
- `roop/face_util.py` — `get_face_analyser`, `align_crop`, deteksi. `g_desired_face_analysis`
  di-set di ProcessMgr:128 = `["landmark_3d_68","landmark_2d_106","detection","recognition"]`
  → **`face.pose` (pitch,yaw,roll) TERSEDIA**.
- `roop/processors/Expression_LivePortrait.py` — restorer ekspresi (pipeline LP).
- `roop/processors/live_portrait_util.py` — helper math LP (`lp_crop_matrix`,
  `compose_affine`, `apply_back_calibration`, `lock_pose`, `headpose_pred_to_degree`).
- `roop/processors/Mask_XSeg.py`, `Mask_FFXSeg_base.py` + `Mask_FFXSeg1/2/3/Many.py`,
  `Mask_FaceParser.py`, `Mask_Clip2Seg.py` — mask engines.
- `roop/processors/Enhance_GFPGAN.py`, `Enhance_CodeFormer.py`,
  `Enhance_RestoreFormerPPlus.py` — enhancer.
- `ui/main.py` (header/footer/launch/`prepare_environment`), `ui/theme.py`
  (`runleashed_theme` OBJECT, `runleashed_css`, `runleashed_js` sticky-JS),
  `ui/tabs/faceswap_tab.py`, `ui/tabs/settings_tab.py`, `ui/tabs/facemgr_tab.py`.
- Dokumen changelog: `PERUBAHAN_*.md` (BATCH, POSEGATE, MASKORDER, POIN5, FFXSEG,
  ROLL_ENHFIX). Baca ini untuk detail tiap fix.

---

## 3. PEKERJAAN LAMA (handoff v1, sudah stabil — ringkas)
- Alignment landmark→5pt RANSAC; convex-hull face matte; deteksi multi-angle;
  FaceSet.AverageEmbeddings robust; LandmarkStabilizer (video).
- Expression restorer LivePortrait (3 ONNX dari facefusion-assets models-3.0.0).
- Dibuang: DMDNet/GPEN/deoldify/real_esrgan/lsdir/Live Cam. Enhancer disisakan:
  None / Codeformer / GFPGAN / Restoreformer++.
- FIX preview lambat (akar: ORT `cudnn_conv_algo_search=EXHAUSTIVE` → `HEURISTIC`).
- FIX UI preview lambat (downscale JPEG preview).
- Expression restorer debug panjang (lihat §7 untuk arsitektur terverifikasi).
- FIX CUDA OOM render (buang `kSameAsRequested`, `cudnn_conv_use_max_workspace=0`).
- FIX crash illegal-memory (sanitasi NaN/Inf + serialisasi LP `_LP_LOCK`).

---

## 4. PEKERJAAN SESI INI (kronologis, semua sudah dikirim & sebagian dikonfirmasi user)

### A. UI RESKIN penuh (SELESAI, dikonfirmasi)
UI 3-kolom hasil rework user via Claude Design, diintegrasi sebagai file Gradio-native.
`theme.py` = OBJECT theme + css + js. Header kustom (`#app_header`), **footer bersih
"Use via API"** (`footer{display:none}` + `gr.HTML('<div class="rl-footer">Use via
API</div>')` di main.py). Sticky-JS butuh `elem_id="swap_row"` + `elem_id="center_stage"`.
Tab Extras DIHAPUS (file + import + download isnet). faceswap_tab = 3 kolom
(kiri input/masking, tengah preview/result, kanan accordion).

### B. Settings layout & default minor (SELESAI, dikonfirmasi)
- settings_tab 3 kolom: **Kol1** Performance+Output, **Kol2** Expression Restorer,
  **Kol3** Interface & maintenance.
- Default di `config_colab.yaml` (yang di-`mv` → config.yaml = default efektif):
  `max_threads: 8`, `output_video_codec: libx264`, `video_quality: 18`. (settings.py
  fallback disamakan.)
- Accordion kanan faceswap: **Model & frames** + **Video & output** = tertutup;
  **Face selection, Expression, Enhancement** = terbuka.
- Subsample upscale default **256px**. Action on no face default **Retry rotated**
  (`no_face_action=1`). Convex-hull face matte default **OFF** (`use_face_hull_mask=False`).
  Pose lock rotation default **ON** (`expression_pose_lock_rotation=True`, override user).
- Face Masking Engine default **DFL XSeg**. Version **5.0.0**.

### C. POSE GATE — fix smear ekspresi di sudut ekstrem (SELESAI, dikonfirmasi)
Di pitch/yaw ekstrem, LP out-of-distribution → warp meleleh. Gate di
`Expression_LivePortrait.Run()`: hitung `max(|pitch|,|yaw|)`, **< soft (45°)** efek
penuh, **soft..hard (45-65°)** fade halus (`factor*=gate`), **≥ hard (65°)** skip
(return swapped_crop). Ramah video. Globals `expression_pose_gate=True`,
`_soft=45`, `_hard=65`. UI: checkbox + 2 slider live. **+ REVISI (§4.K):** gate kini
juga pakai **pose detektor** (`face.pose`) karena estimator LP underestimate yaw di
view rear → `pose_mag = max(LP pitch/yaw, detektor pitch/yaw)`.

### D. REORDER occlusion mask → setelah ER (SELESAI)
Dulu: swap→mask→ER→enhancer (ER mewarp occluder restored). Sekarang **default:
swap→ER→mask→enhancer**. Toggle `mask_after_enhancer` (default False) → memindah mask
setelah enhancer juga. `process_mask` di-resize agar `frame` match `target` (fix
broadcast 256 vs 512 saat mask diterapkan ke `enhanced_frame`). **+ enhancer-aware
(§4.I):** RF++/CodeFormer DIPAKSA mask-after.

### E. POIN 5 — kalibrasi paste-back full pipeline (SELESAI/diterima)
Regresi shift/enlarge/rotate = offset framing bawaan generator LP. `back_M` benar
secara geometri. Tambah `apply_back_calibration(back_M,w,h,scale,dx,dy,rot)` +
globals `expression_lp_cal_scale/dx/dy/rot` + 4 slider live + print `[lp-geom]`.
**Hasil L4: kalibrasi tetap default (1,0,0,0), hasil sudah bagus** → `back_M` benar
per-frame, tak butuh koreksi konstan. Full pipeline **default OFF** (lebih berat;
opt-in). `kept` 103-112% (scale-lock sedikit memperkuat), `kabsch_rot` <2° normal.

### F. FF XSeg occluder — menu baru di Face Masking Engine (SELESAI)
4 menu: **FF XSeg 1/2/3** (bobot DFL-XSeg dari FaceFusion, arsitektur sama xseg.onnx)
+ **FF XSeg (combined)** = ketiganya di-interseksi (`numpy.minimum` face-mask = union
occluder, mode 'many'). File: `Mask_FFXSeg_base.py` (base, io_binding per-call) + 4
class tipis + ProcessMgr.plugins + pre_check download (xseg_1/2 dari
facefusion-assets models-3.1.0, xseg_3 dari models-3.2.0) + dropdown + 2 mapping.
**Caveat:** preprocessing disamakan Mask_XSeg (BGR). Bila mask FF terlihat salah,
tambah `cv2.cvtColor(BGR2RGB)` di base. (URL diverifikasi: 403=ada vs 404=tidak.)

### G. FIX enhancer jitter/stale-frame (SELESAI)
RestoreFormer++ & CodeFormer pakai **`self.io_binding` bersama** → race antar 8 thread
→ tiap beberapa ratus frame frame ketukar/stale. GFPGAN tidak (sudah per-call).
Fix: RF++ & CodeFormer kini **io_binding per-panggilan** (thread-safe).

### H. FIX OOM ke-2 (SELESAI)
OOM `FusedConv` balik (memory_usage ~6GB tapi VRAM habis). Akar: mask engine
(XSeg/FaceParser/FF XSeg) pakai **plain providers** → ORT default
`cudnn_conv_use_max_workspace=1` → workspace konv GB-an per sesi. Karena DFL XSeg jadi
**default**, model mask rakus-workspace resident tiap render (FF combined = 3 sesi =
3×). Fix: semua mask engine kini pakai `tuned_execution_providers()`.

### I. FIX RestoreFormer++ "burned color" saat occluder (SELESAI)
Bukan OOM. Akar: di urutan default (mask sebelum enhancer), piksel occluder di-blend
ke crop dulu, lalu RF++ (codebook) meng-enhance konten non-wajah → color burn. GFPGAN
tahan. Fix enhancer-aware di `get_processing_plugins`: **RF++ & CodeFormer DIPAKSA
mask-after-enhancer** (cuma lihat wajah bersih); **GFPGAN TIDAK diubah** (tetap ikut
toggle, default mask-sebelum). + guard di process_face: enhancer gagal/black →
fallback un-enhanced (anti kotak mentah).

### J. FIX run lokal non-Colab (SELESAI, dikonfirmasi)
`prepare_environment` fallback ke `./output` bila `output_folder` (default Colab
`/content/...`) tak bisa dibuat lokal. `allowed_paths` + output dir & cwd lokal.

### K. FIX roll ekstrem (full pipeline) + rear-view gate (SELESAI, MENUNGGU konfirmasi)
- **Roll ~90°:** `lp_crop_matrix` dulu tanpa rotasi → di roll besar generator LP dapat
  wajah menyamping → misalign/ghosting. Fix: `lp_crop_matrix(roll_deg=...)` roll-align
  (pakai rotasi dari `M_arc`) → wajah selalu TEGAK di crop LP. `back_M` tetap eksak.
  Global `lp_crop_roll_align=True`. Validasi: eye-line ~0° di roll 0/30/60/90.
- **Rear-view (wajah membelakangi):** gate lama miss karena LP underestimate yaw. Fix:
  gate gabung pose detektor (lihat §4.C revisi).

---

## 5. KONDISI GLOBALS TERKINI (`roop/globals.py` — state final v5.0.0)
```
# performa / memori
subsample_size = 128                          # UI default "256px" (lihat faceswap_tab)
cudnn_conv_algo_search = 'HEURISTIC'
cudnn_conv_use_max_workspace = False          # hemat memori
preview_max_height = 720

# mask / paste
use_face_hull_mask = False                    # default OFF
face_hull_forehead = 0.6 ; face_hull_dilate = 0.10 ; face_hull_min_area = 0.22
use_color_transfer = False
mask_after_enhancer = False                   # GFPGAN ikut ini; RF++/CodeFormer dipaksa True

# expression — pendekatan & tuning
expression_restorer = False                   # menu opsional user
expression_power = 2.0                         # 1.0-1.25 setia, 2.0 ekspresif (POIN 4, tunable)
expression_blend_border = 0.2
expression_invert_direction = True             # konvensi generator terbalik (BENAR)

# full pipeline LP (opsional, default OFF)
expression_full_pipeline = False
expression_stitching = False                   # log stitch sudah jujur (toggle, bukan model-loaded)
lp_crop_size = 512 ; lp_crop_scale = 2.3 ; lp_crop_vy = -0.125
lp_crop_roll_align = True                       # roll-align crop LP (fix roll ekstrem)
expression_lp_cal_scale = 1.0 ; _dx = 0.0 ; _dy = 0.0 ; _rot = 0.0   # kalibrasi paste-back

# pose lock deterministik (jalur in-place)
expression_pose_lock = True
expression_pose_lock_scale = True              # POIN 2, A/B tunable
expression_pose_lock_rotation = True           # user override (handoff v1 sarankan False)

# pose gate (anti smear sudut ekstrem)
expression_pose_gate = True
expression_pose_gate_soft = 45.0 ; expression_pose_gate_hard = 65.0

expression_debug = False                        # toggle [expr-delta]/[lp-geom]/[expr-posegate]/[enhancer]
expression_serialize = True                     # serialisasi LP (anti-crash multi-thread)
no_face_action = 1                              # Retry rotated
```
**Default = pendekatan ekspresi in-place + pose-lock(translasi+skala+rotasi) +
pose-gate(LP+detektor) + serialize ON; full pipeline & stitcher OFF; DFL XSeg default;
hull matte OFF.**

---

## 6. ARSITEKTUR EXPRESSION RESTORER (fakta terverifikasi — JANGAN diutak-atik tanpa paham)
- **Konvensi generator TERBALIK** vs penamaan: suapkan `'source'←kp_driving` (ekspresi
  target), `'target'←kp_source` (wajah swap). `expression_invert_direction=True`.
- Stitcher DEFAULT OFF (handoff v1 bilang "buyar"), **tapi** log stitch dulu bohong
  (lapor model-loaded); sudah dibetulkan. Layak A/B ulang dengan log jujur.
- Raw exp delta kecil (~0.01-0.03) → butuh `expression_power` (default 2.0).
- **Jalur LAMA (in-place, DEFAULT):** src=crop arcface swapped, drv=crop arcface target;
  `prepare_crop`→256; feature/motion extractor; kp_source & kp_driving berbagi
  rotasi/skala/translasi; generator (nama TERBALIK); `parse_output`; resize; `feather_blend`.
- **Jalur FULL pipeline (opsional):** rekonstruksi swapped_frame → crop ala-LP
  (`lp_crop_matrix`, skala 2.3, vy -0.125, 512, **kini roll-aligned**) → generator →
  paste-back via `back_M=compose_affine(M_arc, inv(M_lp))` (+ kalibrasi opsional).
- **Pose gate** (in-place & full): `max(|pitch|,|yaw|)` dari LP **dan** detektor
  (`face.pose`), fade soft..hard, skip ≥hard.
- **`.fsz` faceset** dipakai HANYA untuk embedding identitas (recognition align ke
  112×112 internal) → resolusi crop source TINGGI tak menambah detail swap. Detail
  output dari sisi TARGET (subsample/enhancer).
- **inswapper** output 128px; source cuma "menyumbang identitas", bukan tekstur.

---

## 7. PENDING / LANGKAH BERIKUTNYA
1. **[MENUNGGU]** Konfirmasi user: fix roll-align (§4.K) + rear-view gate menghilangkan
   distorsi LP saat kepala miring/membelakangi.
2. **[MENUNGGU]** Konfirmasi RF++ "burned color" hilang setelah enhancer-aware mask-after
   (§4.I) + OOM mask-provider fix (§4.H).
3. **[TERBUKA] A/B stitch on/off** dengan log yang sudah jujur (handoff v1 "OFF lebih
   baik" mungkin tak berlaku lagi setelah full-pipeline + pose-lock).
4. **[TERBUKA] POIN 2 (scale-lock) & POIN 4 (power)** = kenop tuning user (live di
   Settings + Debug log). Bila ketemu kombinasi enak → bake jadi default.
5. **[OPSI] Tambah GPEN** sebagai enhancer (ONNX, cepat, kadang blend lebih mulus) untuk
   A/B vs RF++ — ditawarkan, user belum minta.
6. **[OPSI] FF XSeg RGB**: bila mask FF terlihat salah, tambah BGR→RGB di
   `Mask_FFXSeg_base.py`.
7. **[DEFER]** Full pipeline tetap OPSIONAL (lebih berat). Kalibrasi paste-back tersedia
   bila perlu.

---

## 8. HASIL RISET WEB (sesi ini — untuk referensi)
- **XSeg lebih baru:** model yang dipakai = DFL XSeg generik. Upgrade termudah =
  FaceFusion **xseg_1/2/3** (sudah ditambahkan, §4.F). Cakupan occluder lebih luas
  butuh **face parser** (BiSeNet `bisenet_resnet_34.onnx`, atau SegFace 2025) sebagai
  channel mask kedua. SAM/SAM2/FastSAM = salah alat (bukan occluder aligned-crop).
- **RestoreFormer++:** **sudah versi final** (TPAMI 2023, `wzhouxiff/RestoreFormerPlusPlus`).
  Tidak ada yang lebih baru dari penulis aslinya.
- **GFPGAN:** **v1.4 = final** (TencentARC tak rilis lagi). Tidak ada v1.5.
- Field restorasi pindah ke **diffusion** (DiffBIR/DifFace/PMRF/OSDFace/CodeFormer++ 2025)
  — kualitas tinggi tapi lambat/flicker/jarang ONNX → **kurang cocok untuk swap
  video per-frame**. Trio GAN/codebook (GFPGAN/CodeFormer/RF++) masih pilihan tepat.
- **inswapper 256/512:** InsightFace sengaja menahan rilis; file yang beredar =
  bocoran. **Asisten tidak mencarikan/memberi link file bocoran.** Alternatif legal
  resolusi-tinggi (mis. ReSwapper) tersedia bila user mau.

---

## 9. KONVENSI / ATURAN MAIN
- Jawab **Bahasa Indonesia**, ringkas. Akui kesalahan jujur.
- Jangan jadikan menu opsional user sebagai default tanpa izin. Enhancer opsional.
- Folder uploads read-only; kerja di working copy (`/home/claude/runleashed-main`).
- Tiap fix: edit → validasi math (numpy/cv2, `pip --break-system-packages`) →
  `py_compile` → build-test stub Gradio (mock `roop.*`, build 3 tab + theme + css + js)
  → bersihkan cache → zip → `present_files` → user uji di L4 + kirim log.
- Resep paket: `find runleashed-main -name "*.pyc" -delete; rm -rf __pycache__; cd
  /home/claude && zip -rq /mnt/user-data/outputs/runleashed-main.zip runleashed-main
  -x "*.pyc" -x "*__pycache__*" -x "*.DS_Store"`. Cek zip bersih dari pyc/extras/isnet.
- Mask engine & enhancer baru: pakai `tuned_execution_providers()` (hemat memori) dan
  **io_binding per-panggilan** (thread-safe), JANGAN `self.io_binding` bersama.
- Provider: enhancer/LP/swap/mask SEMUA pakai `tuned_execution_providers()`.
- Network sandbox: github.com OK (api.github.com utk cek asset), huggingface.co DIBLOK
  (403 ≠ file hilang). Verifikasi URL download via 403(ada)-vs-404(tidak) di GitHub.
- Diagnostik (cetak `[expr-delta]`, `[lp-geom]`, `[expr-posegate]`, `[enhancer]`,
  `[timing]`) saat `expression_debug=True` — berkali-kali jadi kunci.

---
*Akhir handoff v2. Lampirkan juga zip build terakhir (`runleashed-main.zip`) bila ingin
asisten baru membaca kode langsung. Build terakhir = v5.0.0 dengan semua fix di atas.*
