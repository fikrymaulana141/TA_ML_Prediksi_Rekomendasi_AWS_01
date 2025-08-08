# ===================================================================
# BAGIAN 1: IMPORT DAN INISIALISASI
# ===================================================================

# --- Import Library ---
import pandas as pd
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from zoneinfo import ZoneInfo # <-- PERUBAHAN 1: Import ZoneInfo

# ===================================================================
# BAGIAN 2: FUNGSI-FUNGSI (DEFINISI)
# ===================================================================

# Fungsi 1: Melakukan Prediksi Cuaca
def prediksi_cuaca(data_realtime, model, scaler_X, scaler_y):
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

# Fungsi 2: Memberikan Rekomendasi Penyiraman
def get_rekomendasi_penyiraman(prediksi_numerik, input_cuaca):
    skor = 0
    suhu = prediksi_numerik['TAVG']
    kelembapan = prediksi_numerik['RH_AVG']
    kecepatan_angin_knot = prediksi_numerik['FF_AVG_KNOT']
    curah_hujan = float(input_cuaca['RR'])
    kecepatan_angin_kmh = kecepatan_angin_knot * 1.852

    if suhu > 30: skor += 3
    elif suhu >= 25: skor += 2
    else: skor += 1

    if kelembapan < 65: skor += 3
    elif kelembapan <= 80: skor += 2
    else: skor += 1

    if kecepatan_angin_kmh > 20: skor += 3
    elif kecepatan_angin_kmh >= 10: skor += 2
    else: skor += 1

    if curah_hujan > 5: skor -= 10
    elif curah_hujan >= 1: skor -= 5

    if skor <= 0: rekomendasi = "Tidak Perlu Penyiraman"
    elif skor <= 4: rekomendasi = "Penyiraman Ringan"
    elif skor <= 7: rekomendasi = "Penyiraman Sedang"
    else: rekomendasi = "Penyiraman Intensif"

    detail = f"Total Skor: {skor}"
    return rekomendasi, detail

# ===================================================================
# BAGIAN 3: BLOK EKSEKUSI UTAMA
# ===================================================================

def jalankan_program():
    """Fungsi utama yang menjalankan seluruh alur program."""
    try:
        # --- Inisialisasi Firebase dan Muat Model ---
        print("🚀 Memulai proses...")
        cred = credentials.Certificate("firebase_credentials.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://tugas-akhir-64cd9-default-rtdb.asia-southeast1.firebasedatabase.app/'
            })

        model = tf.keras.models.load_model('model_h20_p50.h5')
        scaler_X = joblib.load('scaler_X_4var.pkl')
        scaler_y = joblib.load('scaler_y_4var.pkl')

        print("✅ Inisialisasi berhasil.")

        # --- Ambil Data dari Firebase ---
        path_data_input = 'aws_01'
        ref_input = db.reference(path_data_input).order_by_key().limit_to_last(1)
        data_terbaru_dict = ref_input.get()

        if not data_terbaru_dict:
            print("❌ Tidak ada data yang ditemukan di Firebase.")
            return

        # --- Proses, Prediksi, dan Simpan Hasil ---
        key = list(data_terbaru_dict.keys())[0]
        data_mentah = data_terbaru_dict[key]

        print(f"\n[INFO] Data diambil dari path: '/{path_data_input}' (key: {key})")

        suhu_data = data_mentah.get('suhu', {})
        angin_data = data_mentah.get('angin', {})
        hujan_data = data_mentah.get('hujan', {})
        cahaya_data = data_mentah.get('cahaya', {})
        intensitas_cahaya = float(cahaya_data.get('avg', 0.0))

        if intensitas_cahaya > 20000:
            nilai_ss_konversi = 8.0
        elif intensitas_cahaya > 5000:
            nilai_ss_konversi = 5.0
        elif intensitas_cahaya > 1000:
            nilai_ss_konversi = 2.0
        else:
            nilai_ss_konversi = 0.5

        print(f"[INFO] Intensitas cahaya: {intensitas_cahaya}, dikonversi menjadi nilai SS: {nilai_ss_konversi}")

        data_input_model = {
            'TN': float(suhu_data.get('min', 0.0)),
            'TX': float(suhu_data.get('max', 0.0)),
            'RR': float(hujan_data.get('total_harian_mm', 0.0)),
            'FF_X': float(angin_data.get('gust_kmh', 0.0)) * 0.54,
            'SS': nilai_ss_konversi
        }

        print("\n[INFO] Data yang dimasukkan ke model (setelah pemetaan):")
        print(data_input_model)

        prediksi_numerik = prediksi_cuaca(data_input_model, model, scaler_X, scaler_y)
        rekomendasi_siram, detail_skor = get_rekomendasi_penyiraman(prediksi_numerik, data_input_model)

        # <-- PERUBAHAN 2: Membuat timestamp sesuai zona waktu Asia/Jakarta (WIB)
        timestamp_key = datetime.now(ZoneInfo("Asia/Jakarta")).strftime('%Y-%m-%d_%H-%M-%S')
        
        path_baru = f'/Hasil_Prediksi_Rekomendasi_Penyiraman/{timestamp_key}'

        kecepatan_angin_kmh_untuk_disimpan = prediksi_numerik['FF_AVG_KNOT'] * 1.852
        data_prediksi = {
            'Suhu_AVG_C': float(round(prediksi_numerik['TAVG'], 2)),
            'RH_AVG_Persen': float(round(prediksi_numerik['RH_AVG'], 2)),
            'FF_AVG_kmh': float(round(kecepatan_angin_kmh_untuk_disimpan, 2)),
            'DDD_X_Derajat': int(prediksi_numerik['DDD_X']),
        }

        print("\n--- HASIL PREDIKSI CUACA ---")
        for k, v in data_prediksi.items():
            print(f"- {k}: {v}")

        data_rekomendasi = {
            'Rekomendasi': rekomendasi_siram,
            'Detail_Skor': detail_skor,
        }
        data_untuk_disimpan = {
            'Prediksi_Cuaca': data_prediksi,
            'Rekomendasi_Penyiraman': data_rekomendasi
        }

        db.reference(path_baru).set(data_untuk_disimpan)

        print("\n--- REKOMENDASI PENYIRAMAN ---")
        print(f"Rekomendasi: {rekomendasi_siram} ({detail_skor})")
        print(f"\n✅ Data berhasil diproses dan disimpan ke Firebase di path: {path_baru}")

    except Exception as e:
        print(f"\n❌ Terjadi error pada proses utama: {e}")

# ===================================================================
# BAGIAN 4: TITIK MASUK PROGRAM
# ===================================================================

if __name__ == "__main__":
    jalankan_program()
    print("\n Program selesai.")
