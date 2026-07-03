# PERUBAHAN — CABUT ONE-EURO M + AUTO-STOP HARDENING + CABUT RESWAPPER (batch 4)

## 1. One-Euro M (MatrixStabilizer) DICABUT TOTAL
User konfirmasi via A/B: bahkan dengan deadband piksel, One-Euro M tetap
menimbulkan micro-jitter halus terus-menerus di permukaan wajah pada footage yang
alignment-nya sudah tenang. Mematikan landmark smoothing menghilangkannya. Karena
manfaatnya (meredam shimmer) tidak sebanding dengan cacat yang ditimbulkannya di
kasus umum, fitur ini dicabut seluruhnya:
- Kelas `MatrixStabilizer` + `_OneEuro` DIHAPUS dari `face_stabilizer.py`
  (tersisa hanya `LandmarkStabilizer` seperti sebelum batch 2).
- `self.m_stabilizer` (buat + reset + blok smoothing di `process_face`) DIHAPUS.
- Label UI dikembalikan: "Landmark smoothing (video)" (bukan "+ warp").
Landmark smoothing (LandmarkStabilizer) TIDAK tersentuh — tetap berfungsi seperti
build asli. Jitter kembali ke tingkat build pra-batch-1.

KENAPA "di batch 1 ER smoothing OFF tapi tidak jitter, sekarang jitter?" -- karena
batch 1 TIDAK punya One-Euro M sama sekali (itu ditambahkan di batch 2). Jadi
build batch-1 = perilaku alignment build asli = tidak ada micro-jitter tambahan.
Sumber jitter murni dari One-Euro M yang diperkenalkan batch 2, bukan dari
ada/tidaknya ER smoothing. Sekarang setelah dicabut, kondisinya kembali seperti
batch 1.

## 2. Auto-stop setelah video selesai — hardening berlapis
Off-by-one (batch 3) sudah benar, tapi user masih melihat: setelah video selesai
tombol tetap "processing", box "Processed file(s)" menampilkan "error", harus klik
Stop manual. Ini gejala event `start_swap` gagal di tahap FINISH (bukan saat
proses). Diperbaiki di tiga lapis:

a) `start_swap` (UI): gr.Files kini hanya diberi file DARI RUN INI (mtime >= waktu
   mulai), bukan seluruh isi output folder. Di Colab output folder = Google Drive
   yang menumpuk semua render lama; menyerahkan ratusan file lama ke gr.Files
   memaksa Gradio menyalin semuanya ke cache tiap selesai render -> makin lama &
   rawan error tepat di finish. Kini maksimal 50 file terbaru dari run ini.
   Warning + print diagnostik `[finish]` ditambahkan; UI SELALU kembali idle.

b) `update_progress` (ProcessMgr): panggilan progress ke Gradio dibungkus
   try/except + lock. Stream progress ke browser lewat tunnel gradio.live bisa
   mati di tengah render; tracker yang melempar exception TIDAK boleh membunuh
   worker thread (worker mati -> pipeline queue deadlock -> UI stuck selamanya).
   Bila tracker gagal: matikan progress, lanjut render tanpa progress UI.

c) `run_batch_inmem` teardown (ProcessMgr): anti-deadlock menyeluruh.
   - read/write thread jadi daemon (tak pernah menyandera proses).
   - `read_frames_thread.put` pakai timeout 0.2s + re-check `processing`: kalau
     ada worker mati (queue-nya tak pernah dikuras) read thread tetap bisa
     menyadari abort dan keluar, bukan block selamanya di queue penuh.
   - `future.result()` dibungkus: worker yang crash -> set processing=False,
     catat error, lanjut ke teardown (bukan menggantung).
   - teardown menguras frames_queue sambil join (maks ~60s), memberi sentinel
     ke write thread bila ada worker mati, lalu raise error worker di akhir
     supaya lapisan (a) menangkap & mengembalikan UI ke idle.
Divalidasi via simulasi pipeline thread (normal/crash/single-thread): tak ada
deadlock, error ter-surface, progres mencapai 100%.

## 3. ReSwapper DICABUT TOTAL
Per keputusan user (kualitas jauh di bawah InSwapper):
- `pre_check` (core.py): dua unduhan `reswapper_128.onnx` & `reswapper_256.onnx`
  DIHAPUS -> tidak ada auto-download di depan lagi.
- `ProcessMgr` initialize: cabang mapping ReSwapper 128/256 DIHAPUS; swap model
  selalu `inswapper_128.onnx`.
- UI: `model_swap_choices = ["InSwapper 128"]`; dropdown pemilih model di-hide
  (`visible=False`) TAPI komponen dipertahankan agar semua event wiring yang
  memakainya sebagai input (previewinputs, preview mask, start_swap) tetap valid.
File `reswapper_*.onnx` lama di ./models (bila ada) tidak dihapus otomatis; boleh
dihapus manual, tak lagi dipakai.

## Validasi
`py_compile` seluruh tree; face_stabilizer hanya expose LandmarkStabilizer;
build-test Gradio 5.9.1 (3 tab, model tunggal, ReSwapper hilang); integrasi
process_face 8 frame tanpa m_stabilizer (tak ada AttributeError); simulasi
teardown thread (normal/crash/1-thread) tanpa deadlock.

## Default
Tidak ada One-Euro M. Landmark smoothing tetap seperti semula. Model swap tunggal.
Semua fix batch 1-3 lain (INTER_CUBIC, gate, pose lock adaptif, off-by-one,
mask 1x, dsb) tetap ada.
