# ===================================================================
# BAGIAN 1: IMPORT LIBRARY
# ===================================================================
import pandas as pd
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ===================================================================
# BAGIAN 2: FUNGSI-FUNGSI (DEFINISI)
# ===================================================================

def prediksi_cuaca(data_realtime, model, scaler_X, scaler_y):
    """Menjalankan prediksi cuaca menggunakan model yang telah dilatih."""
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

def get_rekomendasi_penyiraman(prediksi_numerik, input_cuaca):
    """Memberikan rekomendasi penyiraman berdasarkan parameter ideal Sacha Inchi."""
    skor = 0
    suhu = prediksi_numerik['TAVG']
    kelembapan = prediksi_numerik['RH_AVG']
    kecepatan_angin_knot = prediksi_numerik['FF_AVG_KNOT']
    curah_hujan = float(input_cuaca['RR'])
    kecepatan_angin_kmh = kecepatan_angin_knot * 1.852
    if suhu >= 32: skor += 3
    elif suhu >= 28: skor += 2
    elif suhu >= 24: skor += 1
    else: skor += 0
    if kelembapan < 60: skor += 3
    elif kelembapan < 70: skor += 2
    elif kelembapan <= 85: skor += 1
    else: skor += 0
    if kecepatan_angin_kmh > 20: skor += 3
    elif kecepatan_angin_kmh >= 10: skor += 2
    else: skor += 1
    if curah_hujan > 5: skor -= 10
    elif curah_hujan >= 1: skor -= 5
    
    # === LOGIKA REKOMENDASI BARU ===
    if skor > 4:
        rekomendasi = "Optimal"
    elif skor > 2:  # Artinya skor adalah 3 atau 4
        rekomendasi = "Sedang"
    else:  # Artinya skor adalah 2, 1, 0, atau lebih rendah
        rekomendasi = "Tidak Menguntungkan"
        
    detail = f"Total Skor: {skor}"
    return rekomendasi, detail

def get_klasifikasi_cuaca(prediksi_numerik, input_cuaca, intensitas_cahaya):
    """Mengklasifikasikan kondisi cuaca berdasarkan prediksi dan input."""
    # Mengambil nilai-nilai yang dibutuhkan
    suhu = prediksi_numerik['TAVG']
    kelembapan = prediksi_numerik['RH_AVG']
    kecepatan_angin_kmh = prediksi_numerik['FF_AVG_KNOT'] * 1.852
    curah_hujan_input = float(input_cuaca['RR'])
    # 1. Prioritas utama: Cek apakah sedang hujan (berdasarkan data input)
    if curah_hujan_input > 10.0:
        return "Hujan Lebat"
    elif curah_hujan_input > 2.5:
        return "Hujan Sedang"
    elif curah_hujan_input > 0.1:
        return "Hujan Ringan"
    # 2. Jika tidak hujan, cek kondisi angin
    kecepatan_meter_per_detik = kecepatan_angin_kmh / 3.6
    if kecepatan_meter_per_detik > 10:
        return "Berangin"
    # 3. Cek kondisi suhu ekstrem (Panas/Sejuk)
    if suhu > 34.0:
        return "Panas"
    elif suhu < 22.0:
        return "Sejuk"
    # 4. Cek apakah sekarang malam hari (berdasarkan waktu eksekusi skrip)
    waktu_sekarang = datetime.now(ZoneInfo("Asia/Jakarta"))
    jam_sekarang = waktu_sekarang.hour
    # Malam hari dianggap dari jam 18:00 sore hingga 05:59 pagi
    if jam_sekarang >= 18 or jam_sekarang < 6:
        # Logika malam hari: gunakan kelembapan untuk klasifikasi
        if kelembapan > 85.0:
            return "Berawan (Malam)"
        else:
            return "Cerah (Malam)"
    # 5. Jika SIANG HARI dan tidak ekstrem, gunakan intensitas cahaya
    if intensitas_cahaya > 40000:
        return "Cerah"
    elif intensitas_cahaya > 10000:
        return "Cerah Berawan"
    else:
        return "Berawan"

def konversi_derajat_ke_arah_angin(derajat):
    """Mengubah derajat arah angin menjadi 8 arah mata angin."""
    if 337.5 <= derajat <= 360 or 0 <= derajat < 22.5: return "Utara"
    elif 22.5 <= derajat < 67.5: return "Timur Laut"
    elif 67.5 <= derajat < 112.5: return "Timur"
    elif 112.5 <= derajat < 157.5: return "Tenggara"
    elif 157.5 <= derajat < 202.5: return "Selatan"
    elif 202.5 <= derajat < 247.5: return "Barat Daya"
    elif 247.5 <= derajat < 292.5: return "Barat"
    elif 292.5 <= derajat < 337.5: return "Barat Laut"
    else: return "Tidak Terdefinisi"

# ===================================================================
# BAGIAN 3: BLOK EKSEKUSI UTAMA
# ===================================================================
def jalankan_program():
    try:
        # Inisialisasi Firebase
        cred = credentials.Certificate("firebase_credentials.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://tugas-akhir-64cd9-default-rtdb.asia-southeast1.firebasedatabase.app/'
            })
        
        # Muat model dan scaler
        model = tf.keras.models.load_model('model_predict_weather_h20_n10_.h5')
        scaler_X = joblib.load('scaler_X_predict_weather_.pkl')
        scaler_y = joblib.load('scaler_y_predict_weather_.pkl')
        
        # Ambil data terbaru dari Firebase
        ref_input = db.reference('aws_01').order_by_key().limit_to_last(1)
        data_terbaru_dict = ref_input.get()
        if not data_terbaru_dict:
            print("Tidak ada data sensor yang ditemukan.")
            return

        key = list(data_terbaru_dict.keys())[0]
        print(f"[INFO] Data diambil dari path: '/aws_01' (key: {key})")
        data_mentah = data_terbaru_dict[key]
        
        # Ekstraksi dan pemetaan data input
        suhu_data = data_mentah.get('suhu', {})
        angin_data = data_mentah.get('angin', {})
        hujan_data = data_mentah.get('hujan', {})
        cahaya_data = data_mentah.get('cahaya', {})
        intensitas_cahaya = float(cahaya_data.get('avg', 0.0))
        
        # Konversi intensitas cahaya ke nilai SS
        if intensitas_cahaya > 20000: nilai_ss_konversi = 8.0
        elif intensitas_cahaya > 5000: nilai_ss_konversi = 5.0
        elif intensitas_cahaya > 1000: nilai_ss_konversi = 2.0
        else: nilai_ss_konversi = 0.5
        
        print(f"Intensitas cahaya: {intensitas_cahaya}, dikonversi menjadi nilai SS: {nilai_ss_konversi}")
        
        data_input_model = {
            'TN': float(suhu_data.get('min', 0.0)),
            'TX': float(suhu_data.get('max', 0.0)),
            'RR': float(hujan_data.get('total_harian_mm', 0.0)),
            'FF_X': float(angin_data.get('gust_kmh', 0.0)) * 0.54,
            'SS': nilai_ss_konversi
        }
        
        print("Data yang dimasukkan ke model (setelah pemetaan):")
        print(f"{data_input_model}")
        
        # Jalankan prediksi dan analisis
        prediksi_numerik = prediksi_cuaca(data_input_model, model, scaler_X, scaler_y)
        rekomendasi_siram, detail_skor = get_rekomendasi_penyiraman(prediksi_numerik, data_input_model)
        klasifikasi_cuaca_hasil = get_klasifikasi_cuaca(prediksi_numerik, data_input_model, intensitas_cahaya)
        
        arah_angin_teks = konversi_derajat_ke_arah_angin(prediksi_numerik['DDD_X'])
        kecepatan_angin_kmh_unrounded = prediksi_numerik['FF_AVG_KNOT'] * 1.852
        
        # Tampilkan hasil ke konsol
        print("\n--- HASIL PREDIKSI CUACA ---")
        print(f"- Klasifikasi Cuaca: {klasifikasi_cuaca_hasil}")
        print(f"- Suhu_AVG_C: {prediksi_numerik['TAVG']}")
        print(f"- RH_AVG_Persen: {prediksi_numerik['RH_AVG']}")
        print(f"- FF_AVG_kmh: {kecepatan_angin_kmh_unrounded}")
        print(f"- DDD_X_Derajat: {prediksi_numerik['DDD_X']} ({arah_angin_teks})")
        
        print("\n--- REKOMENDASI KONDISI TANAH ---")
        print(f"Rekomendasi: {rekomendasi_siram} ({detail_skor})")

        # ==============================================================================
        # >> BLOK KODE YANG DIMODIFIKASI <<
        # Logika untuk menentukan timestamp jadwal prediksi berikutnya
        # ==============================================================================
        waktu_sekarang = datetime.now(ZoneInfo("Asia/Jakarta"))
        jadwal_prediksi_jam = [0, 3, 6, 9, 12, 15, 18, 21]
        target_jam = None

        # Cari jadwal terdekat SETELAH jam saat ini
        for jam in jadwal_prediksi_jam:
            if jam > waktu_sekarang.hour:
                target_jam = jam
                break

        waktu_target = waktu_sekarang
        # Jika tidak ditemukan (misal, eksekusi jam 22:00), maka targetnya adalah jam 00:00 hari berikutnya
        if target_jam is None:
            target_jam = jadwal_prediksi_jam[0]
            waktu_target += timedelta(days=1)

        # Buat objek waktu yang presisi sesuai jadwal (menit dan detik di-nol-kan)
        waktu_prediksi_final = waktu_target.replace(hour=target_jam, minute=0, second=0, microsecond=0)
        timestamp_key = waktu_prediksi_final.strftime('%Y-%m-%d_%H-%M-%S')
        # ==============================================================================
        
        kecepatan_angin_kmh_prediksi = prediksi_numerik['FF_AVG_KNOT'] * 1.852
        
        data_untuk_disimpan = {
            'Klasifikasi_Cuaca': klasifikasi_cuaca_hasil,
            'Prediksi_Cuaca': {
                'Suhu_AVG_C': float(round(prediksi_numerik['TAVG'], 2)),
                'RH_AVG_Persen': float(round(prediksi_numerik['RH_AVG'], 2)),
                'FF_AVG_kmh': float(round(kecepatan_angin_kmh_prediksi, 2)),
                'DDD_X_Derajat': int(prediksi_numerik['DDD_X']),
                'Arah_Angin_Teks': arah_angin_teks
            },
            'Rekomendasi_Penyiraman': {
                'Rekomendasi': rekomendasi_siram,
                'Detail_Skor': detail_skor,
            }
        }
        
        # Simpan hasil ke path baru di Firebase
        path_baru = f'/Hasil_Prediksi_Rekomendasi_Penyiraman_AWS_01/{timestamp_key}'
        db.reference(path_baru).set(data_untuk_disimpan)
        
        print(f"\nData berhasil diproses dan disimpan ke Firebase di path: {path_baru}")

    except Exception as e:
        print(f"Terjadi error pada proses utama: {e}")

# ===================================================================
# BAGIAN 4: TITIK MASUK PROGRAM
# ===================================================================

if __name__ == "__main__":
    jalankan_program()
