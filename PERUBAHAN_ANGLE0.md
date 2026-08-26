# "Upright detection priority" — bonus sudut-0 dikembalikan sebagai slider

## Kenapa dikembalikan
Saya sempat mengubah dedup (bonus skor untuk sudut-0), lalu MENCABUTNYA setelah
user melaporkan regresi "hanya hidung yang ter-swap". Pencabutan itu didasarkan
pada sebab-akibat yang SALAH: setelahnya kami membuktikan wajah-palsu-di-hidung
punya IoU 0.09 terhadap wajah asli -- jauh di bawah ambang dedup 0.5 -- sehingga
ia lolos sebagai "wajah lain" APA PUN urutan prioritasnya. Jadi perubahan dedup
bukan penyebab regresi itu.

Sementara gejala yang dilaporkan user ("berubah-ubah padahal posisi wajah kurang
lebih sama") justru yang seharusnya dibantu skema bonus: pada kepala miring,
deteksi tegak kadang lolos tipis (kps jelek) dan kadang gagal (dipakai deteksi
rotasi yang bagus). Prioritas MUTLAK memaksa yang tipis selalu menang saat ia
muncul -> alignment berganti-ganti tiap beberapa frame.

## Bentuknya sekarang: satu angka, bisa digeser
`roop.globals.angle0_bonus`, slider "Upright detection priority" (0.0-1.0,
default 1.0) di grup Alignment & stabilization.
  1.0  = sudut-0 SELALU menang (perilaku saat ini, default -> tak ada yang berubah)
  0.10 = sudut-0 menang pada seri & selisih tipis; rotasi yang jelas lebih yakin menang
  0.0  = murni skor detektor
Seri selalu jatuh ke sudut-0.

## Validasi
default 1.0 -> sudut-0 menang di SEMUA kombinasi termasuk ekstrem 0.00 vs 1.00
  (identik dengan perilaku sebelum perubahan ini);
0.10 -> 0.38 vs 0.82 dimenangkan rotasi, 0.78 vs 0.82 tetap sudut-0;
0.0  -> 0.79 vs 0.80 dimenangkan rotasi, seri jatuh ke sudut-0;
dua wajah terpisah tetap utuh; deteksi rotasi tetap dipakai bila sudut-0 tak ada;
FP hidung tetap lolos (memang di luar wewenang dedup -- atasi lewat ambang
confidence atau penyaringan ukuran).
py_compile + build Gradio 5.9.1.

## Cara uji
Klip yang sama, ubah HANYA slider ini: 1.0 -> 0.10 -> 0.0.
Perhatikan frame berwajah miring: apakah alignment berhenti berganti-ganti.
Kalau FP hidung muncul, itu masalah TERPISAH -> naikkan Detection confidence.
