# Fix full-pipeline di roll ekstrem + guard enhancer (anti kotak)

## 1. Full LP pipeline misalign saat kepala miring ~90 deg (roll)
Akar: lp_crop_matrix dulu crop scale+translate TANPA rotasi, sedang jalur in-place
memberi LP crop arcface yang sudah di-upright. Di roll besar, generator LP (dilatih
wajah tegak) dapat wajah menyamping -> warp misalign/ghosting. Pose gate hanya
cek pitch/yaw, bukan roll.
Fix: lp_crop_matrix kini roll-align (param roll_deg) memakai rotasi dari M_arc, jadi
wajah selalu TEGAK di crop LP berapa pun roll kepala. Round-trip back_M tetap eksak
(aljabar affine umum). Global toggle: lp_crop_roll_align=True.
Validasi numerik: eye-line di crop LP ~0 deg pada head-roll 0/30/60/90.

## 2. RestoreFormer++ kadang "kotak/mentah" tiap beberapa ratus frame
Akar (dugaan kuat): akibat OOM FusedConv intermiten -> konv enhancer gagal ->
NaN -> frame hitam -> ke-paste jadi kotak. Intervalnya cocok dgn error OOM.
Fix berlapis:
- (turn ini) mask engine kini pakai tuned_execution_providers -> OOM jauh berkurang.
- Guard di process_face: bila enhancer gagal / mengembalikan frame kosong, lewati
  enhancement frame itu (paste wajah un-enhanced) alih-alih garbage. Satu frame
  sedikit lebih lembut jauh lebih tidak kentara dari kotak mentah.

Validasi: py_compile semua OK + uji numerik roll-align.

## 3. RestoreFormer++ "burned color" saat ada occluder (REVISI diagnosa)
Ternyata bukan OOM. Akar: di urutan default (mask SEBELUM enhancer), piksel asli
occluder di-blend ke crop dulu, lalu RestoreFormer++ meng-enhance konten NON-wajah
itu -> color burn. RF++/CodeFormer (codebook) sensitif ke input non-wajah; GFPGAN
tahan -> tak burn.
Fix (enhancer-aware order di get_processing_plugins): untuk RestoreFormer++ &
CodeFormer, occlusion mask DIPAKSA berjalan SETELAH enhancer -> mereka cuma melihat
wajah swap bersih. GFPGAN TIDAK diubah (tetap ikut toggle mask_after_enhancer,
default mask-sebelum). Jadi GFPGAN yang sudah benar tidak terganggu.

## 4. Pose gate tidak menangkap "wajah dari belakang" (rear/extreme yaw)
Gejala: LP distorsi saat wajah hampir membelakangi kamera; tanpa LP bagus.
Akar: pose gate lama hanya pakai pitch/yaw dari motion-extractor LP, yang
UNDERESTIMATE yaw di view rear (wajah terlalu tersembunyi) -> gate tak ter-trigger.
Fix: gate kini juga memakai pose dari face analyser (landmark_3d_68: pitch,yaw),
yang andal lebih jauh dari profil, lalu ambil sinyal terkuat
(pose_mag = max(LP pitch/yaw, detektor pitch/yaw)). Jadi saat kepala membelakangi,
detektor melaporkan yaw besar -> gate skip -> swap bersih dipakai (tak distorsi).
Berlaku di jalur in-place & full pipeline. context kini membawa target_face.pose.
Debug: [expr-posegate] kini cetak lp(...) dan det(...) terpisah.

Validasi: py_compile + uji gate gabungan (rear ter-skip walau LP underestimate).
