# PERBAIKAN BERSIH — "Error" di Processed Files + Stop merusak UI

## Berpikir ulang dari nol (mengabaikan revisi-revisi sebelumnya)
Petunjuk dari screenshot user: box "Processed File(s)" menampilkan "Error"
(BUKAN kosong). Itu bukan "yield hilang" -- itu gr.Files GAGAL saat menerima
nilai. Dua akar terpisah:

### Akar A: "Error" di box hasil
`start_swap` yield terakhirnya mengirim `gr.Files(value=outfiles)` dengan
`outfiles = rglob(seluruh output folder)`. Output folder di Colab = Google Drive
yang menumpuk SEMUA render lama; menyerahkan seluruh isinya ke gr.Files membuat
Gradio mencoba mem-provision setiap file (besar/banyak/path aneh) ke client ->
komponen error. (Persis yang user duga soal gr.file.)
FIX: `batch_process_regular` sudah mencatat file hasil tiap entri di
`ProcessEntry.finalname`. Kumpulkan HANYA itu (file run ini, biasanya 1) alih-alih
rglob. Bersih, benar, dan tak membebani gr.Files.

### Akar B: Stop merusak queue -> "error connecting" -> harus restart
`bt_stop.click(..., cancels=[start_event, after_swap_event])`. `start_swap` adalah
generator yang saat proses sedang BLOCKING di dalam `batch_process_regular` (bukan
di titik `yield`). Gradio tidak bisa membatalkan fungsi Python yang blocking di
tengah -- cancel hanya efektif di `yield`. Akibatnya cancel meninggalkan event
queue dalam keadaan setengah-batal -> klik berikutnya "error connecting" -> UI
wedge -> restart run.py. (User konfirmasi versi lama pun bermasalah soal Stop --
memang cancels ini biang historisnya.)
FIX: HAPUS `cancels`. Stop jadi SOFT-STOP murni: `roop.globals.processing=False`,
worker mengeceknya dan berhenti sendiri, lalu `start_swap` selesai natural.
`queue=False` menjaga Stop tetap responsif.

## Auto-idle tombol (dipertahankan dari perbaikan sebelumnya)
`reset_event = start_event.then(reset_buttons_idle, outputs=[bt_start, bt_stop])`.
`.then()` dijamin jalan setelah generator berakhir (sukses/error/cancel), jadi
tombol SELALU balik idle walau yield terakhir tak sampai. `reset_buttons_idle`
juga membersihkan `is_processing` + `roop.globals.processing` -> Start berikutnya
tak pernah terblokir, tak perlu restart. Terverifikasi: fn ber-trigger_after
start_swap; TIDAK ada `cancels` di dependency mana pun.

## Yang TIDAK diubah
Success-path teardown thread (join tanpa timeout, sentinel dijamin) tetap -> video
utuh tersimpan. Perubahan ini murni UI/event + sumber daftar file.

## Validasi sandbox
py_compile; build Gradio 5.9.1 (3 tab); event graph: reset_buttons_idle
trigger_after=start_swap, cancels=[] di semua dependency; koleksi finalname:
hanya file yang benar-benar ada, entri gagal/tak diproses dilewati.

## Batas kejujuran
Ini masalah client-server yang tak bisa direproduksi penuh di sandbox. Yang
terbukti: struktur event benar + tak ada cancels + sumber file aman. Konfirmasi
final di L4: render pendek -> box menampilkan file (bukan "Error"), tombol balik
Start SENDIRI, Stop saat proses tidak bikin "error connecting", dan bisa render
lagi tanpa restart.
