# PERUBAHAN — "Restore eyes" + hapus "Show mask overlay in preview"

## Restore eyes (Opsi A: dua box mata terpisah, alis/kelopak dikecualikan)
Toggle baru "Restore eyes" di UI, tepat di bawah "Restore original mouth area".
Independen (jalan terlepas dari Expression Restorer), sama seperti mouth restore.
Yang di-restore: interior mata ASLI dari target ditempel balik ke hasil swap.

Implementasi (ProcessMgr):
- `create_eyes_masks`: alih-alih hard-code indeks 2d106 (berisiko salah tempel
  bila konvensi indeks beda), titik mata dicari GEOMETRIS -- dari 106 landmark,
  ambil yang terdekat ke tiap kps mata detektor (radius 0.22x jarak antar-mata).
  Robust terhadap layout indeks, dan alis otomatis tereksklusi (titik alis jauh
  di atas pusat mata, di luar radius). Menghasilkan DUA box terpisah (kiri+kanan),
  padding kecil (0.25x, lebih ketat dari mouth) supaya kelopak/alis tak ikut.
- `apply_eyes_area`: pakai blend feather + color transfer yang sama dengan mouth.
- Dipanggil di `process_face` tepat setelah mouth restore.

Plumbing: ProcessOptions.restore_original_eyes -> batch_process_regular (core.py)
-> start_swap + preview (faceswap_tab). Preview juga menampilkan efeknya.

## BUG FIX (ditemukan saat test) — apply_color_transfer divide-by-zero
`apply_color_transfer` menghitung target_std/source_std; bila source uniform
(std=0), hasilnya NaN -> restore diam-diam gagal (ter-catch). Ini juga
memengaruhi MOUTH restore pada rongga gelap seragam. Diperbaiki: source_std
di-floor ke 1e-3. Divalidasi: uniform source -> hasil finite.

## Hapus "Show mask overlay in preview"
Checkbox dihapus dari UI. Komponennya dipertahankan sebagai state tersembunyi
(visible=False, selalu False) supaya wiring preview yang memakainya tetap valid.

## Validasi
Math: dua box mata terpisah, alis dikecualikan (box y0=284 vs brow y=240),
mata asli tertempel (channel warna naik), brow tak tersentuh, no-kps/no-landmark
-> [] graceful, color-transfer guard finite. py_compile + build Gradio 5.9.1 OK.
