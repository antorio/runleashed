# ROLLBACK — semua tweak jitter dicabut, perbaikan source-face dipertahankan

## Kenapa
Setelah rangkaian tweak jitter, hasil render user jauh MEMBURUK: "micro jitter di
mana-mana, bagian dalam wajah selalu bergetar, crumble distorted junk landmark",
lebih buruk daripada versi awal yang sudah lebih baik dari FaceFusion.

## Diagnosis kesalahan saya
1. CROSSFADE cacat secara konsep. Gate mem-blend titik 68->5 dengan kps detektor.
   Keduanya bukan dua estimasi dari hal yang sama -- mereka DUA KONVENSI berbeda
   yang berjarak sistematis 9-29px (baseline terukur 0.044-0.143 x ukuran wajah).
   Blending dua konvensi tidak menghasilkan titik tengah yang valid; ia
   menghasilkan geometri yang bukan milik keduanya -> template arcface menerima
   himpunan tak koheren -> affine salah -> "crumble/distorted". Aktif di 11-17%
   frame (30-50 flip) = ratusan frame rusak per klip.
2. METRIK FLICKER menipu. Blending adalah low-pass filter: ia PASTI menurunkan
   selisih antar-frame. Jadi "flicker -62..-69%" terjadi JUSTRU KARENA blending,
   bukan karena hasilnya benar. Alignment yang salah-konsisten punya flicker
   rendah tapi terlihat mengerikan. Saya mengoptimalkan kehalusan, bukan
   kebenaran, lalu melaporkannya sebagai lampu hijau.
3. Sinyal yang saya abaikan: M_pipeline sempat > M_raw (44.8 vs 42.7) -- itu
   alarm regresi yang saya sebut "masih oke".
4. Klaim "2dfan4 paling akurat" tidak pernah saya verifikasi di pipeline ini.
   Observasi user (luas wajah berbeda, hasil lebih masuk akal TANPA hi-landmarker,
   baseline disagreement berayun 0.033->0.143 antar klip) justru mengindikasikan
   masalah integrasi 2dfan4 -- kemungkinan pemetaan balik crop 256->frame atau
   ketidakcocokan konvensi 68-titik (iBUG) dengan landmark_68_to_5 yang ditulis
   untuk buffalo 1k3d68. BELUM diinvestigasi; jangan dianggap fakta.

## DICABUT (kembali ke build referensi runleashed-main_47_)
- Align5Conditioner (smoothing 5 titik align + gate v2 + crossfade + reset
  diskontinuitas) -> file roop/align_conditioner.py DIHAPUS.
- Sanity gate v2 -> kembali ke gate sederhana batch-1 (ambang tetap).
- Urutan bbox-smoothing-sebelum-landmarker (stabilize(faces, refine_fn=...)) ->
  kembali ke stabilize(faces) asli; call site kembali refine lalu stabilize.
- RANSAC threshold 0.05/0.08 * image_size -> kembali ke 100 (perilaku asli).
- alpha_min clamp 0.02 -> kembali 0.08.
- landmark_smoothing_deadzone 0.012 -> kembali 0.006.
- tools_jitter_probe.py DIHAPUS (metriknya yang menyesatkan).
- Changelog tweak jitter (ALIGN5_GATEV2, JITTER_PROBE, JITTER_ROOTCAUSE) dihapus.

## DIPERTAHANKAN (permintaan user + tidak menyentuh geometri)
- Selected-mode: source face yang dipilih SELALU dipakai (dulu 2+ target face
  membuang pilihan user dan memakai index target). + tidak ada wajah ter-swap dua
  kali + index out-of-range jatuh ke 0.
- Modul analisis wajah KONSTAN di semua mode (+genderage selalu) -> analyser tak
  di-rebuild saat ganti mode, ER konsisten antar mode deteksi.
- UI: input_faces.select -> .then(preview) (ganti source face langsung terlihat),
  6 event preview trigger_mode="always_last" + concurrency_limit=1 (klik tak
  dibuang, tak menumpuk -> tak ada "error connecting").
- Semua fitur non-alignment: Restore eyes + feather, model dari HF, perbaikan
  auto-stop/UI idle, guard divide-by-zero apply_color_transfer, dsb.
- Slider "Still-face dead-zone" tetap ada di UI (default kembali 0.006) sebagai
  knob opsional.

## Validasi
py_compile seluruh tree; globals (deadzone 0.006, align5 hilang, analysis list
konstan); LandmarkStabilizer.stabilize kembali ke signature asli + alpha_min 0.08;
nol referensi align5/crossfade/refine_fn di seluruh source; build Gradio 5.9.1
(rantai source-select -> preview utuh, 6 event always_last); process_face x15 pada
pipeline hasil rollback; simulasi selected-mode mengikuti source pilihan.

## Aturan kerja ke depan (disepakati)
Tidak ada perubahan alignment yang masuk tanpa verifikasi VISUAL pada klip pendek
lebih dulu. Metrik agregat hanya untuk mendeteksi REGRESI, tidak pernah untuk
mengklaim perbaikan.
