# PERUBAHAN — batch ekspresi + UI settings + output path

## A. 5 poin ekspresi (yang bisa di-default, di-default)
- **Poin 1 (DONE):** `expression_pose_lock_rotation` default `True → False` di `globals.py`.
  Kabsch di jalur in-place bisa menyerap ekspresi lokal jadi rotasi palsu.
- **Poin 2 (live A/B):** `pose_lock_scale` dibiarkan `True` — ini keputusan visual,
  bukan default pasti. Toggle live di Settings; diagnostik menunjukkan efeknya.
- **Poin 3 (visibilitas):** tidak ada default yang pantas di-bake (subsample = trade-off
  per-job). Diagnostik kini mencetak `crop=WxH` (resolusi input restorer) → kalau
  128x128, itu bottleneck; naikkan "Subsample upscale to" ke 256/512.
- **Poin 4 (preferensi):** `expression_power` dibiarkan `2.0` (tegas). ~1.0–1.25 = setia.
  Tunable live; diagnostik mencetak nilai power.

## B. Diagnostik [expr-delta] diperkaya (flag baru `expression_debug`)
`roop/processors/Expression_LivePortrait.py` — per frame mencetak:
`exp|max`, `sig_before`/`sig_after` (sinyal ekspresi sebelum/sesudah pose-lock),
`kept%`, `kabsch_rot` (sudut rotasi yang dibuang — besar = rotation_lock memakan
ekspresi), `crop=WxH`, status lock/scale/rot, `power`, `factor`, `full`, `stitch`.
Aktifkan via checkbox **"Debug log [expr-delta]"** di Settings → Expression Restorer
(atau `profile_timings`). Untuk A/B: nyalakan, ubah toggle, baca console.

## C. Globals penting kini ada di UI Settings (live, reset saat restart)
Compartment baru (semua `.open=False`, set `roop.globals` saat runtime):
- **Alignment & detection** (kolom 1): `use_landmark_alignment`,
  `multi_angle_detection_mode`, `use_color_transfer`.
- **Face mask & paste-back** (kolom 2): `use_face_hull_mask`, `face_hull_forehead`,
  `face_hull_dilate`, `face_hull_min_area`.
- **Stabilization & faceset** (kolom 3): `landmark_smoothing`,
  `landmark_smoothing_strength`, `faceset_average_mode`, `faceset_outlier_threshold`.
Catatan: toggle ini live (efek di frame berikut), TAPI tidak persist — restart balik
ke default `globals.py`. Kalau ketemu kombinasi enak → bake permanen.

## D. Output path
- Setting baru **"Output Folder"** di Settings → Output, default
  `/content/drive/MyDrive/c` (string sudah benar: forward-slash, tanpa escape).
- CFG-backed: `settings.py` (load+save) + `config_colab.yaml` → persist via Apply Settings.
- `ui/main.py prepare_environment()` set `output_path` dari `CFG.output_folder`
  (fallback `<cwd>/output` bila kosong). Edit kotak → langsung update live + tersimpan saat Apply.

## E. Default value lain (dari permintaan sebelumnya, tetap)
`max_threads: 8`, `output_video_codec: libx264`, `video_quality: 18`.
