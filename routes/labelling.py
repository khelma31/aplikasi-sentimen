from flask import Blueprint, redirect, url_for, jsonify
from db import get_db_connection

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

label = Blueprint('label', __name__)

@label.route("/labelling", methods=["POST"])
def labelling():
    connection = get_db_connection()
    
    try:
        with connection.cursor() as cursor:
            # Ambil data dari tabel data_klasifikasi, data_training, dan data_testing
            cursor.execute("SELECT id, preprocessing_text FROM data_klasifikasi")
            data_klas = cursor.fetchall()
            
            cursor.execute("SELECT id, preprocessing_text FROM data_training")
            data_training = cursor.fetchall()

            cursor.execute("SELECT id, preprocessing_text FROM data_testing")
            data_testing = cursor.fetchall()
            
            cursor.execute("SELECT id, preprocessing_text FROM data_implementasi")
            data_implementasi = cursor.fetchall()

    finally:
        connection.close()

    # Buat DataFrame dari data_klasifikasi dan data_training serta data_testing
    df = pd.DataFrame(data_klas, columns=['id', 'preprocessing_text'])
    df_training = pd.DataFrame(data_training, columns=['id', 'preprocessing_text'])
    df_testing = pd.DataFrame(data_testing, columns=['id', 'preprocessing_text'])
    df_implementasi = pd.DataFrame(data_implementasi, columns=['id', 'preprocessing_text'])
    
    # Load model sentiment analysis
    pretrained = "mdhugol/indonesia-bert-sentiment-classification"
    model = AutoModelForSequenceClassification.from_pretrained(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    # Label index untuk sentiment analysis
    label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
    
    # Fungsi untuk menganalisis sentimen
    def analyze_sentiment(text):
        result = sentiment_analysis(text)
        status = label_index[result[0]['label']]
        score = result[0]['score']
        return status, score
    
    # Menerapkan analisis sentimen ke data
    df['label'], df['score'] = zip(*df['preprocessing_text'].apply(analyze_sentiment))
    df_training['label'], df_training['score'] = zip(*df_training['preprocessing_text'].apply(analyze_sentiment))
    df_testing['label'], df_testing['score'] = zip(*df_testing['preprocessing_text'].apply(analyze_sentiment))
    df_implementasi['label'], df_implementasi['score'] = zip(*df_implementasi['preprocessing_text'].apply(analyze_sentiment))

    # Koneksi database untuk mengupdate data
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # Update data_klasifikasi dengan label
            for i, row in df.iterrows():
                cursor.execute(
                    "UPDATE data_klasifikasi SET label = %s WHERE id = %s",
                    (row['label'], row['id'])
                )
            
            # Update data_training dengan label
            for i, row in df_training.iterrows():
                cursor.execute(
                    "UPDATE data_training SET label = %s WHERE id = %s",
                    (row['label'], row['id'])
                )
                
            # Update data_testing dengan label
            for i, row in df_testing.iterrows():
                cursor.execute(
                    "UPDATE data_testing SET label = %s WHERE id = %s",
                    (row['label'], row['id'])
                )
                
            # Update data_implementasi dengan label
            for i, row in df_implementasi.iterrows():
                cursor.execute(
                    "UPDATE data_implementasi SET label = %s WHERE id = %s",
                    (row['label'], row['id'])
                )
                
        connection.commit()
    except Exception as e:
        connection.rollback()
        print(f"Error occurred: {str(e)}")  # Cetak kesalahan untuk debugging
        return jsonify({"message": str(e)}), 500
    finally:
        connection.close()

    return redirect(url_for("training.klastraining", status="label_success"))
