import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# ===================================================================
# FUNGSI-FUNGSI BANTU
# ===================================================================

def prediksi_cuaca(data_realtime, model, scaler_X, scaler_y):
    """Fungsi ini menjalankan prediksi cuaca menggunakan model yang telah dilatih."""
    features = ['TN', 'TX', 'RR', 'SS', 'FF_X']
    df_input = pd.DataFrame([data_realtime], columns=features)
    input_scaled = scaler_X.transform(df_input)
    pred_scaled = model.predict(input_scaled, verbose=0)
    pred_final = scaler_y.inverse_transform(pred_scaled)
    
    hasil_numerik = {
        'TAVG': pred_final[0][0],
        'RH_AVG': pred_final[0][1],
        'FF_AVG_KNOT': pred_final[0][2],
        'DDD_X': int(pred_final[0][3])
    }
    return hasil_numerik

def get_rekomendasi_sacha_inchi(prediksi_numerik, input_cuaca):
    """
    Fungsi ini memberikan rekomendasi penyiraman berbasis skor
    yang disesuaikan untuk tanaman Sacha Inchi.
    """
    skor = 0
    suhu = prediksi_numerik['TAVG']
    kelembapan = prediksi_numerik['RH_AVG']
    kecepatan_angin_knot = prediksi_numerik['FF_AVG_KNOT']
    curah_hujan = float(input_cuaca['RR'])
    kecepatan_angin_kmh = kecepatan_angin_knot * 1.852

    if suhu > 30: skor += 3
    elif suhu >= 24: skor += 2
    else: skor += 1

    if kelembapan < 70: skor += 3
    elif kelembapan <= 85: skor += 2
    else: skor += 1

    if kecepatan_angin_kmh > 20: skor += 3
    elif kecepatan_angin_kmh >= 10: skor += 2
    else: skor += 1

    if curah_hujan > 5: skor -= 10
    elif curah_hujan >= 1: skor -= 4
    
    if skor <= 0:
        rekomendasi = "Tidak Perlu Penyiraman"
        detail = f"Total Skor: {skor}. Diperkirakan akan turun hujan yang cukup."
    elif skor <= 4:
        rekomendasi = "Penyiraman Ringan"
        detail = f"Total Skor: {skor}. Cukup jaga kelembapan media tanam."
    elif skor <= 7:
        rekomendasi = "Penyiraman Normal"
        detail = f"Total Skor: {skor}. Lakukan penyiraman sesuai jadwal rutin."
    else:
        rekomendasi = "Penyiraman Intensif"
        detail = f"Total Skor: {skor}. Cuaca sangat panas/kering, tanaman butuh air ekstra."

    return rekomendasi, detail

# === FUNGSI BARU UNTUK KLASIFIKASI CUACA ===
def klasifikasi_cuaca(prediksi_numerik, input_cuaca):
    """
    Fungsi ini mengambil hasil prediksi dan data input untuk memberikan
    satu label klasifikasi cuaca yang komprehensif.
    """
    suhu = prediksi_numerik['TAVG']
    kelembapan = prediksi_numerik['RH_AVG']
    kecepatan_angin_knot = prediksi_numerik['FF_AVG_KNOT']
    curah_hujan = float(input_cuaca['RR'])

    if kecepatan_angin_knot < 0:
        kecepatan_angin_knot = 0
    
    kecepatan_angin_kmh = kecepatan_angin_knot * 1.852
    klasifikasi = ""

    # 1. Cek Curah Hujan Terlebih Dahulu (berdasarkan standar BMKG)
    if curah_hujan > 50:
        klasifikasi = "Hujan Lebat"
    elif curah_hujan >= 20:
        klasifikasi = "Hujan Sedang"
    elif curah_hujan >= 5:
        klasifikasi = "Hujan Ringan"
    
    # 2. Jika tidak hujan, tentukan berdasarkan kelembapan dan suhu
    else:
        if kelembapan > 85:
            klasifikasi = "Sangat Lembap / Berawan"
        elif kelembapan > 65:
            klasifikasi = "Cerah Berawan"
        else:
            klasifikasi = "Cerah / Kering"
    
    # 3. Tambahkan keterangan suhu jika ekstrem
    if suhu > 33:
        klasifikasi += " & Panas"
        
    # 4. Tambahkan keterangan angin (berdasarkan Skala Beaufort)
    if kecepatan_angin_kmh > 20:
        klasifikasi += " & Berangin"
            
    return klasifikasi

# ===================================================================
# BLOK EKSEKUSI UTAMA
# ===================================================================
try:
    # --- Langkah 1: Inisialisasi dan Muat Aset ---
    print("--- Memulai Proses Prediksi, Rekomendasi, dan Klasifikasi ---")
    DATABASE_URL = 'https://tugas-akhir-64cd9-default-rtdb.asia-southeast1.firebasedatabase.app/'
    
    cred = credentials.Certificate("serviceAccountKey.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
    print("✅ Berhasil terhubung ke Firebase.")

    model = tf.keras.models.load_model('model_h20_p50.h5')
    scaler_X = joblib.load('scaler_X_4var.pkl')
    scaler_y = joblib.load('scaler_y_4var.pkl')
    print("✅ Model terbaik dan scaler berhasil dimuat.")

    # --- Langkah 2: Ambil Data Mentah ---
    ref_input = db.reference('/').order_by_key().limit_to_last(1)
    data_terbaru_dict = ref_input.get()

    if not data_terbaru_dict:
        print("❌ Tidak ada data yang ditemukan di root database.")
        exit()

    key = list(data_terbaru_dict.keys())[0]
    data_mentah = data_terbaru_dict[key]
    
    print("\n[INFO] Data mentah terbaru berhasil diambil (key:", key, ")")

    # --- Langkah 3: Konversi Data Mentah ke Format Input Model ---
    suhu_data = data_mentah.get('suhu', {})
    angin_data = data_mentah.get('angin', {})
    hujan_data = data_mentah.get('hujan', {})
    cahaya_data = data_mentah.get('cahaya', {})

    data_input_model = {
        'TN':   float(suhu_data.get('min', 0.0)),
        'TX':   float(suhu_data.get('max', 0.0)),
        'RR':   float(hujan_data.get('total_harian_mm', 0.0)),
        'FF_X': float(angin_data.get('gust_kmh', 0.0)) * 0.54,
        'SS':   float(cahaya_data.get('avg', 0.0))
    }
    
    print("[INFO] Data setelah dikonversi untuk input model:")
    print(data_input_model)

    # --- Langkah 4: Jalankan Semua Fungsi ---
    hasil_prediksi_numerik = prediksi_cuaca(data_input_model, model, scaler_X, scaler_y)
    rekomendasi, detail_skor = get_rekomendasi_sacha_inchi(hasil_prediksi_numerik, data_input_model)
    # Panggil fungsi klasifikasi yang baru
    klasifikasi = klasifikasi_cuaca(hasil_prediksi_numerik, data_input_model)

    # --- Langkah 5: Tampilkan Hasil Akhir ---
    kecepatan_angin_kmh_prediksi = hasil_prediksi_numerik['FF_AVG_KNOT'] * 1.852
    
    print("\n" + "="*40)
    print("--- HASIL PREDIKSI, KLASIFIKASI, & REKOMENDASI ---")
    print(f"Klasifikasi Cuaca: {klasifikasi}")
    print(f"- Prediksi Suhu Rata-rata (TAVG): {hasil_prediksi_numerik['TAVG']:.2f} °C")
    print(f"- Prediksi Kelembapan Rata-rata (RH_AVG): {hasil_prediksi_numerik['RH_AVG']:.2f} %")
    print(f"- Prediksi Kecepatan Angin Rata-rata (FF_AVG): {kecepatan_angin_kmh_prediksi:.2f} km/jam")
    print(f"- Prediksi Arah Angin Dominan (DDD_X): {hasil_prediksi_numerik['DDD_X']}°")
    print(f"Rekomendasi Penyiraman: {rekomendasi} ({detail_skor})")
    print("="*40)
    
except Exception as e:
    print(f"\n❌ Terjadi error pada proses utama: {e}")
