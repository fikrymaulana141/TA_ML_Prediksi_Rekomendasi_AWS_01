# ===================================================================
# BAGIAN 1: IMPORT LIBRARY
# ===================================================================
import pandas as pd
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
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
    if skor <= 2: rekomendasi = "Tidak Perlu Penyiraman"
    elif skor <= 5: rekomendasi = "Penyiraman Ringan"
    elif skor <= 7: rekomendasi = "Penyiraman Sedang"
    else: rekomendasi = "Penyiraman Intensif"
    detail = f"Total Skor: {skor}"
    return rekomendasi, detail

def get_klasifikasi_cuaca(prediksi_numerik, input_cuaca):
    """Memberikan klasifikasi cuaca berdasarkan prioritas: Hujan > Suhu/Kelembapan + Angin."""
    suhu = prediksi_numerik['TAVG']
    kelembapan = prediksi_numerik['RH_AVG']
    kecepatan_angin_knot = prediksi_numerik['FF_AVG_KNOT']
    curah_hujan = float(input_cuaca['RR'])
    kecepatan_angin_kmh = kecepatan_angin_knot * 1.852
    if curah_hujan >= 50: return "Hujan Lebat"
    if curah_hujan >= 20: return "Hujan Sedang"
    if curah_hujan >= 5: return "Hujan Ringan"
    klasifikasi_utama = ""
    if suhu >= 33: klasifikasi_utama = "Panas"
    elif suhu < 24: klasifikasi_utama = "Sejuk"
    else:
        if kelembapan > 85: klasifikasi_utama = "Berawan"
        elif kelembapan < 65: klasifikasi_utama = "Cerah"
        else: klasifikasi_utama = "Cerah Berawan"
    if kecepatan_angin_kmh >= 20: return f"{klasifikasi_utama} & Berangin"
    else: return klasifikasi_utama

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
        print("Memulai proses...")
        cred = credentials.Certificate("firebase_credentials.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://tugas-akhir-64cd9-default-rtdb.asia-southeast1.firebasedatabase.app/'
            })
        model = tf.keras.models.load_model('model_h20_p50.h5')
        scaler_X = joblib.load('scaler_X_4var.pkl')
        scaler_y = joblib.load('scaler_y_4var.pkl')
        print("Inisialisasi berhasil.")

        ref_input = db.reference('aws_01').order_by_key().limit_to_last(1)
        data_terbaru_dict = ref_input.get()
        if not data_terbaru_dict:
            print("Tidak ada data sensor yang ditemukan.")
            return

        key = list(data_terbaru_dict.keys())[0]
        data_mentah = data_terbaru_dict[key]
        
        suhu_data = data_mentah.get('suhu', {})
        angin_data = data_mentah.get('angin', {})
        hujan_data = data_mentah.get('hujan', {})
        cahaya_data = data_mentah.get('cahaya', {})
        intensitas_cahaya = float(cahaya_data.get('avg', 0.0))
        if intensitas_cahaya > 20000: nilai_ss_konversi = 8.0
        elif intensitas_cahaya > 5000: nilai_ss_konversi = 5.0
        elif intensitas_cahaya > 1000: nilai_ss_konversi = 2.0
        else: nilai_ss_konversi = 0.5
        data_input_model = {
            'TN': float(suhu_data.get('min', 0.0)),
            'TX': float(suhu_data.get('max', 0.0)),
            'RR': float(hujan_data.get('total_harian_mm', 0.0)),
            'FF_X': float(angin_data.get('gust_kmh', 0.0)) * 0.54,
            'SS': nilai_ss_konversi
        }
        
        prediksi_numerik = prediksi_cuaca(data_input_model, model, scaler_X, scaler_y)
        rekomendasi_siram, detail_skor = get_rekomendasi_penyiraman(prediksi_numerik, data_input_model)
        klasifikasi_cuaca_hasil = get_klasifikasi_cuaca(prediksi_numerik, data_input_model)
        arah_angin_teks = konversi_derajat_ke_arah_angin(prediksi_numerik['DDD_X'])
        
        timestamp_key = datetime.now(ZoneInfo("Asia/Jakarta")).strftime('%Y-%m-%d_%H-%M-%S')
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
        
        # --- PENYESUAIAN PATH PENYIMPANAN ---
        path_baru = f'/Hasil_Prediksi_Rekomendasi_Penyiraman/{timestamp_key}'
        db.reference(path_baru).set(data_untuk_disimpan)
        
        print("\n--- HASIL PREDIKSI & REKOMENDASI ---")
        print(f"Klasifikasi Cuaca      : {klasifikasi_cuaca_hasil}")
        print(f"Arah Angin Prediksi    : {arah_angin_teks} ({prediksi_numerik['DDD_X']}°)")
        print(f"Rekomendasi Penyiraman : {rekomendasi_siram} ({detail_skor})")
        
        # --- PENYESUAIAN PESAN OUTPUT ---
        print(f"Data berhasil diproses dan disimpan ke Firebase di path: {path_baru}")

    except Exception as e:
        print(f"Terjadi error pada proses utama: {e}")

# ===================================================================
# BAGIAN 4: TITIK MASUK PROGRAM
# ===================================================================

if __name__ == "__main__":
    jalankan_program()
    print("Program selesai.")
