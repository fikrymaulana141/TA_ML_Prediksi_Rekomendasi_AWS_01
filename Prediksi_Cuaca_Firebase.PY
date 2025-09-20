# SISTEM PREDIKSI CUACA DAN REKOMENDASI CERDAS DENGAN SENSOR TANAH
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime, timedelta
import pytz

# --- 1. KONFIGURASI ---
# Nama file kredensial sekarang dibaca dari environment variable untuk keamanan
CRED_FILE = 'firebase_credentials.json' 
DATABASE_URL = 'https://tugas-akhir-64cd9-default-rtdb.asia-southeast1.firebasedatabase.app/'
DATA_PATH = 'aws_01'
PREDICTION_PATH = 'prediksi/aws_01'
MODEL_FILE = 'model_peramal_cuaca_weighted_final.h5'
SCALER_FILE = 'scaler_engineered.pkl'
ENCODER_FILE = 'encoder_engineered.pkl'
FEATURE_ORDER = ['temp', 'humidity', 'precip', 'windspeed', 'solarradiation']


# --- 2. FUNGSI-FUNGSI UTAMA ---

def initialize_firebase():
    """Inisialisasi koneksi ke Firebase menggunakan file kredensial."""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(CRED_FILE)
            firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
        print("Koneksi ke Firebase berhasil diinisialisasi.")
        return True
    except Exception as e:
        print(f"Error saat inisialisasi Firebase: {e}")
        return False

def fetch_and_resample_data(ref_path, hours_needed=6, records_per_hour=12):
    """
    Mengambil data frekuensi tinggi, mengubahnya menjadi data per jam,
    dan mengembalikan 6 jam data terakhir.
    """
    num_records_to_fetch = int((hours_needed + 2) * records_per_hour * 1.5)
    print(f"Mengambil ~{num_records_to_fetch} record terakhir dari Firebase untuk di-resample...")

    try:
        ref = db.reference(ref_path)
        data = ref.order_by_key().limit_to_last(num_records_to_fetch).get()

        if not data:
            print("Error: Tidak ada data sama sekali di Firebase path yang ditentukan.")
            return None

        print(f"Berhasil mengambil {len(data)} record data.")
        
        records = []
        for timestamp_key, record_data in data.items():
            time_str = record_data.get('waktu')
            if not time_str:
                continue
            records.append({
                'datetime_str': time_str,
                'temp': record_data.get('suhu', {}).get('avg', 0),
                'humidity': record_data.get('kelembaban', {}).get('avg', 0),
                'precip': record_data.get('hujan', {}).get('total_harian_mm', 0),
                'windspeed': record_data.get('angin', {}).get('avg_kmh', 0),
                'solarradiation': record_data.get('cahaya', {}).get('avg', 0)
            })

        if not records:
            print("Error: Tidak ada record valid yang bisa diproses.")
            return None

        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime_str'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)
        df.set_index('datetime', inplace=True)
        df = df.drop(columns=['datetime_str'])

        print("Melakukan resampling data dari 5-menit ke per jam...")
        df_hourly = df.resample('H').mean()
        df_hourly.interpolate(method='linear', inplace=True)
        df_hourly.fillna(method='bfill', inplace=True)
        df_hourly.fillna(method='ffill', inplace=True)

        df_last_hours = df_hourly.tail(hours_needed)

        if len(df_last_hours) < hours_needed:
            print(f"\nPeringatan: Data setelah resampling hanya menghasilkan {len(df_last_hours)} jam (butuh {hours_needed} jam).")
            print("Akan dilakukan padding dengan data paling awal untuk melanjutkan proses prediksi.")
            
            first_row = df_last_hours.iloc[0:1]
            hours_to_pad = hours_needed - len(df_last_hours)
            padding_df = pd.concat([first_row] * hours_to_pad)
            df_padded = pd.concat([padding_df, df_last_hours])
            
            print("Padding berhasil. Data 6 jam sekarang tersedia.")
            return df_padded[FEATURE_ORDER]

        print("Resampling berhasil. Data 6 jam terakhir siap digunakan.")
        return df_last_hours[FEATURE_ORDER]

    except Exception as e:
        print(f"Error saat mengambil dan memproses data dari Firebase: {e}")
        return None

def predict_weather_3_hours_ahead(sequence_df, model, scaler, encoder):
    """Fungsi untuk mengambil data sekuens dan memprediksi cuaca."""
    processed_df = pd.DataFrame()
    for i in range(len(sequence_df) - 1, -1, -1):
        for col in sequence_df.columns:
            value = sequence_df[col].iloc[len(sequence_df)-1-i]
            processed_df[f'{col}_t-{i}'] = [value]
    
    processed_df['humidity_change_3hr'] = processed_df['humidity_t-0'] - processed_df['humidity_t-3']
    processed_df['solarradiation_change_3hr'] = processed_df['solarradiation_t-0'] - processed_df['solarradiation_t-3']
    processed_df['temp_change_3hr'] = processed_df['temp_t-0'] - processed_df['temp_t-3']
    processed_df['windspeed_change_3hr'] = processed_df['windspeed_t-0'] - processed_df['windspeed_t-3']
    
    normalized_data = scaler.transform(processed_df)
    probabilities = model.predict(normalized_data)[0]
    predicted_index = np.argmax(probabilities)
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    
    return predicted_label, probabilities, encoder.classes_

def fetch_latest_soil_moisture(ref_path):
    """
    Mengambil data terbaru dari Firebase dan mengekstrak nilai kelembapan tanah.
    """
    print("\nMengambil data sensor kelembapan tanah terbaru...")
    try:
        ref = db.reference(ref_path)
        latest_data = ref.order_by_key().limit_to_last(1).get()
        
        if not latest_data:
            print("Peringatan: Tidak ada data sensor tanah yang ditemukan.")
            return None

        key = list(latest_data.keys())[0]
        soil_moisture = latest_data[key].get('tanah', {}).get('kelembaban_persen')

        if soil_moisture is not None:
            print(f"Data kelembapan tanah berhasil didapatkan: {soil_moisture}%")
            return float(soil_moisture)
        else:
            print("Peringatan: Kunci 'kelembaban_persen' tidak ditemukan pada data terbaru.")
            return None
            
    except Exception as e:
        print(f"Error saat mengambil data sensor tanah: {e}")
        return None

def get_watering_recommendation(weather_prediction, current_soil_moisture):
    """
    Memberikan rekomendasi penyiraman dengan logika yang disempurnakan
    untuk menangani kasus tanah basah + prediksi hujan.
    """
    SOIL_MOISTURE_THRESHOLD = 50.0  #(%)
    is_rain_predicted = "hujan" in weather_prediction.lower()

    # KONDISI 1: Tanah sudah cukup lembab
    if current_soil_moisture >= SOIL_MOISTURE_THRESHOLD:
        if is_rain_predicted:
            # Tanah sudah basah dan akan hujan -> Berisiko overwatering
            rekomendasi = "Tidak Menguntungkan"
            alasan = f"Kondisi berisiko. Kelembapan tanah sudah tinggi ({current_soil_moisture}%) dan diperkirakan akan turun hujan."
        else:
            # Tanah sudah basah dan tidak akan hujan -> Kondisi aman
            rekomendasi = "Sedang"
            alasan = f"Kelembapan tanah ({current_soil_moisture}%) sudah cukup. Penyiraman tidak diperlukan."
    
    # KONDISI 2: Tanah kering
    else:
        if is_rain_predicted:
            # Tanah kering, tapi akan hujan -> Tunda penyiraman
            rekomendasi = "Tidak Menguntungkan"
            alasan = f"Sebaiknya tunda penyiraman. Tanah kering, namun diperkirakan akan turun hujan ({weather_prediction})."
        else:
            # Tanah kering dan tidak akan hujan -> Waktu terbaik untuk menyiram
            rekomendasi = "Optimal"
            alasan = f"Waktu terbaik untuk menyiram. Tanah kering ({current_soil_moisture}%) dan tidak ada prediksi hujan."
            
    return rekomendasi, alasan

def save_prediction_to_firebase(ref_path, prediction, probabilities, class_names, recommendation_result):
    """
    Menyimpan hasil prediksi dan rekomendasi ke Firebase sesuai jadwal.
    """
    print(f"Menyimpan hasil lengkap ke Firebase path: '{ref_path}'...")
    try:
        ref = db.reference(ref_path)
        wib = pytz.timezone('Asia/Jakarta')
        execution_time = datetime.now(wib)
        
        next_schedule_hour = ((execution_time.hour // 3) + 1) * 3
        target_time = execution_time.replace(minute=0, second=0, microsecond=0)
        if next_schedule_hour >= 24:
            target_time += timedelta(days=1)
            target_time = target_time.replace(hour=0)
        else:
            target_time = target_time.replace(hour=next_schedule_hour)

        timestamp_str = target_time.strftime('%Y-%m-%d_%H-%M-%S')
        
        prediction_data = {
            'prediksi_cuaca': prediction,
            'rekomendasi_penyiraman': recommendation_result,
            'timestamp_prediksi': target_time.isoformat(),
            'timestamp_eksekusi': execution_time.isoformat(),
            'probabilitas': {name: float(f"{prob*100:.2f}") for name, prob in zip(class_names, probabilities)}
        }
        
        ref.child(timestamp_str).set(prediction_data)
        print(f"Prediksi dan rekomendasi untuk jadwal {target_time.strftime('%Y-%m-%d %H:%M:%S')} berhasil disimpan.")
        return True
    except Exception as e:
        print(f"Error saat menyimpan ke Firebase: {e}")
        return False


# --- 3. PROSES UTAMA ---
if __name__ == "__main__":
    if initialize_firebase():
        try:
            print("\nMemuat model, scaler, dan encoder...")
            final_model = tf.keras.models.load_model(MODEL_FILE)
            engineered_scaler = joblib.load(SCALER_FILE)
            engineered_encoder = joblib.load(ENCODER_FILE)
            
            input_df = fetch_and_resample_data(DATA_PATH, hours_needed=6)
            current_soil_moisture = fetch_latest_soil_moisture(DATA_PATH)
            
            if input_df is not None and current_soil_moisture is not None:
                print("\nData input yang akan digunakan (6 jam terakhir setelah resampling):")
                print(input_df)
                print("\nMelakukan prediksi untuk 3 jam ke depan...")
                prediction, probs, class_names = predict_weather_3_hours_ahead(
                    input_df, final_model, engineered_scaler, engineered_encoder
                )
                
                print("------")
                print("HASIL PREDIKSI & REKOMENDASI")
                print(f"Prediksi Cuaca: {prediction.upper()}")
                print(f"Nilai Sensor Kelembapan Tanah: {current_soil_moisture}%")

                rekomendasi, alasan = get_watering_recommendation(prediction, current_soil_moisture)
                
                print(f"Kondisi Penyiraman: {rekomendasi}")
                print(f"Alasan: {alasan}")
                print("------")

                save_prediction_to_firebase(PREDICTION_PATH, prediction, probs, class_names, 
                                            {"rekomendasi": rekomendasi, "alasan": alasan, "kelembapan_terbaca": current_soil_moisture})
            else:
                print("\nProses dibatalkan karena data input (cuaca atau tanah) tidak lengkap.")
            
        except FileNotFoundError:
            print("\nERROR: Salah satu file model (.h5, .pkl) tidak ditemukan.")
        except Exception as e:
            print(f"\nTerjadi error pada proses utama: {e}")