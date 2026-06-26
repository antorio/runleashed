# Pose gate — fix smear restorer ekspresi di sudut ekstrem

Masalah: di pitch/yaw sangat ekstrem (kepala mendongak jauh / profil kuat),
LivePortrait out-of-distribution → warp meleleh/smear. Swap tanpa restorer bersih.

Fix: gerbang pose di `Expression_LivePortrait.Run()` (jalur in-place & full).
- Hitung max(|pitch|,|yaw|) dari motion extractor (sudah tersedia).
- < soft (45°): efek penuh (pose normal tak tersentuh — tetap "udah oke").
- soft..hard (45–65°): efek **memudar halus** (factor *= gate) → ramah video.
- >= hard (65°): **skip**, kembalikan swap bersih (return swapped_crop).
- Print `[expr-posegate]` saat `expression_debug` aktif.

Globals baru (default): `expression_pose_gate=True`, `_soft=45.0`, `_hard=65.0`.
UI: checkbox "Pose gate" + 2 slider (start fade / full skip) di accordion
Expression Restorer (live-tunable tanpa restart).

Validasi: py_compile + build penuh gradio 5.9.1 OK; sanity gate (1.0<45, fade,
0.0>=65) benar.
