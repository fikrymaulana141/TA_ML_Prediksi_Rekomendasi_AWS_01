# Proyek Prediksi Cuaca dan Rekomendasi Penyiraman Otomatis

Proyek ini menggunakan model machine learning untuk memprediksi kondisi cuaca dan memberikan rekomendasi penyiraman. Proses ini dijalankan secara otomatis menggunakan GitHub Actions.

## Fitur
- **Prediksi Cuaca:** Memprediksi suhu, kelembapan, dan angin.
- **Rekomendasi Penyiraman:** Memberikan rekomendasi berdasarkan hasil prediksi.
- **Otomatisasi:** Berjalan secara otomatis sesuai jadwal via GitHub Actions.
- **Pencatatan Riwayat:** Menyimpan setiap hasil prediksi sebagai log di Firebase.

## Setup
1. **Clone Repositori:** `git clone https://github.com/NAMA_ANDA/NAMA_REPO_ANDA.git`
2. **Buat Secret:**
   - Buka repositori di GitHub, pergi ke **Settings > Secrets and variables > Actions**.
   - Buat secret baru bernama `FIREBASE_CREDENTIALS`.
   - Isinya adalah output dari perintah **Base64** pada file kredensial `.json` Anda.

## Status Workflow
![Nama Workflow](https://github.com/NAMA_ANDA/NAMA_REPO_ANDA/actions/workflows/run_prediction.yml/badge.svg)
