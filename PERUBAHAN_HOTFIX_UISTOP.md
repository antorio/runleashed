# HOTFIX — UI tidak auto-idle setelah render selesai (akar sebenarnya: event graph)

## Diagnosis ulang (holistik, dari akar)
Fakta kunci yang tadinya kuabaikan: video SUDAH tersimpan di output -> artinya
`batch_process_regular` SUDAH return normal. Jadi bug BUKAN di thread/pipeline
ProcessMgr (semua tambalanku di sana mengejar tempat yang salah). Bug ada di
lapisan EVENT GRADIO.

`start_swap` adalah generator: yield-AWAL set tombol ke processing (sampai ke
UI), lalu render, lalu yield-AKHIR set tombol balik idle. Dengan progress
callback manual aktif (`self.progress_gradio`) + `show_progress='full'`, Gradio
5.9.1 bisa menutup event-stream generator begitu progress mencapai 100% SEBELUM
mengirim yield-AKHIR. Akibatnya: render selesai + video tersimpan, tapi tombol
tak pernah balik idle. Tekan Stop -> `cancels=[start_event]` membatalkan event
yang di server sudah selesai -> respons kosong/aneh ("error" di box) -> tombol
tetap stuck -> harus restart run.py.

## Perbaikan (tidak bergantung pada yield generator)
Reset tombol dipindah ke event TERPISAH via `.then()`:
`reset_event = start_event.then(fn=reset_buttons_idle, outputs=[bt_start, bt_stop])`
`.then()` DIJAMIN berjalan setelah generator berakhir -- sukses, error, ATAU
cancel -- jadi tombol SELALU balik idle walau yield-akhir hilang. `reset_buttons_idle`
juga membersihkan `is_processing` + `roop.globals.processing` sehingga run yang
stuck tak pernah bisa memblok Start berikutnya.
Terverifikasi di struktur Gradio: fn `reset_buttons_idle` terdaftar dengan
`trigger_after=<start_swap>` (dijadwalkan setelah start_swap).

Lapisan pengaman: `stop_swap` tetap mereset tombol langsung; `reset_event` TIDAK
masuk daftar cancels stop, jadi ia tetap jalan saat Stop ditekan.

## Yang TIDAK diubah
gr.Files tetap seperti yang kamu setujui. Teardown thread success-path (join
tanpa timeout, sentinel dijamin) dari hotfix sebelumnya tetap -> video utuh
tersimpan. Perbaikan ini murni di wiring event UI.

## Validasi
py_compile; build Gradio 5.9.1 3 tab; event graph menunjukkan reset_buttons_idle
ber-trigger_after start_swap; reset_buttons_idle membersihkan kedua flag +
mengembalikan 2 update tombol.

## Catatan uji
Karena `.then()` adalah mekanisme client-server, konfirmasi finalnya ada di L4:
render pendek -> tombol harus balik idle SENDIRI tanpa klik Stop, dan Start bisa
langsung dipakai untuk render berikutnya tanpa restart.
