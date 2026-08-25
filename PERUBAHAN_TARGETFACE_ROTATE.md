# FIX — "use face from this frame" tak menemukan wajah miring

## Gejala
Wajah ~90 derajat: render/swap menemukannya, tapi tombol "use face from this
frame" melaporkan "no face detected". Akibatnya wajah itu tak bisa ditandai
sebagai target, sehingga di mode "selected" ia tak punya embedding untuk
dicocokkan -> tidak ter-swap, atau tercocokkan ke target lain yang salah.

## Akar masalah: DUA jalur deteksi yang berbeda
  render/swap  -> get_all_faces_multi / get_first_face_multi  (0 + 90/270/180)
  UI face picker -> extract_face_images -> get_all_faces       (0 saja)
Jadi apa yang bisa DIPILIH tidak sama dengan apa yang bisa DI-SWAP.
Catatan: preview SWAP sendiri (gambar hasil di panel) memakai live_swap ->
process_frame, jadi ia sudah konsisten dengan render. Yang tidak konsisten
hanya pengambil wajah.

## Perbaikan (tepat di tempat masalahnya)
`extract_face_images(..., use_multi_angle=False)` -- parameter baru, DEFAULT OFF.
Diaktifkan HANYA di satu tempat: pengambilan TARGET face dari frame video
(faceswap_tab, `(True, selected_frame)`), dan mengikuti setting
`multi_angle_detection_mode` yang sudah ada.
Sengaja TIDAK diaktifkan untuk source photo maupun face manager: gambar itu
tegak, jadi mencoba 4 orientasi hanya menambah waktu dan berpotensi memunculkan
wajah palsu dari sudut aneh.

## Validasi
- default parameter = False (6 pemanggil lain tak berubah perilakunya)
- dari 7 pemanggil extract_face_images, TEPAT 1 memakai use_multi_angle=True,
  dan itu jalur target-frame
- jalur yang benar-benar terpanggil dibuktikan dengan menyadap fungsinya:
  default -> get_all_faces; flag ON -> get_all_faces_multi(fallback);
  setelah mode diubah ke 'always' -> get_all_faces_multi(always)
- py_compile + build Gradio 5.9.1
