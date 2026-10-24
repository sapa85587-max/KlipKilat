# Klipkilat: cara pasang gratis dari HP

Klipkilat memotong video panjang jadi klip pendek. Aplikasi menulis transkrip, memberi skor potensi viral di tiap bagian, lalu memotong klip 9:16 dengan subtitle yang menempel di video.

## Pasang di Hugging Face Space (gratis, cukup dari browser HP)

1. Daftar atau masuk di huggingface.co.
2. Buka menu **New Space**. Isi nama bebas (misalnya `klipkilat`), pilih **Docker** lalu **Blank**, hardware **CPU basic (free)**. Pilih **Private** supaya hanya kamu yang bisa memakai.
3. Di tab **Files**, pilih **Add file** lalu **Upload files**. Unggah file ini: `Dockerfile`, `requirements.txt`, `main.py`, `pipeline.py`, `scoring.py`, `render.py`, `index.html`. Semuanya di satu level, tanpa folder. Jangan menimpa `README.md` milik Space, karena isinya pengaturan Docker.
4. Tunggu tab **App** selesai membangun (sekitar 5 sampai 10 menit pertama kali). Setelah itu buka Space dan pakai.

## Cara pakai

- **Dari link**: tempel link video lalu tekan Cari klip terbaik.
- **Dari file**: pilih video dari HP. Cocok kalau link diblokir.
- Hasilnya urut dari skor tertinggi. Ketuk satu klip untuk menonton, geser Mulai dan Selesai, pilih subtitle dan format, lalu Buat klip dan Unduh klip.

## Penilaian viral

Tanpa kunci, skor dihitung dari pembuka kalimat, kata emosional, angka, tempo bicara, jeda, energi suara, dan apakah klip utuh. Dengan kunci Gemini gratis dari aistudio.google.com/apikey (tempel di Pengaturan), 12 klip teratas dinilai ulang oleh AI dan judulnya dibuat lebih menarik. Skor hanya perkiraan.

## Batas yang perlu diketahui

- Video lebih dari 90 menit ditolak. Ubah lewat variabel `MAX_MINUTES`.
- Hasil dan file sementara dihapus setelah 3 jam atau saat Space restart.
- Link YouTube sering diblokir dari server gratis. Kalau gagal, unduh videonya lalu unggah sebagai file.
- Space gratis bisa tidur kalau lama tidak dipakai, dan bangun lagi saat dibuka.
- Pakai hanya untuk video milik sendiri atau yang kamu punya izinnya.
