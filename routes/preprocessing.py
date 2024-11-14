from sklearn.model_selection import train_test_split
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from db import get_db_connection

proses = Blueprint("preprocessing", __name__)

@proses.route("/preprocessing", methods=["POST"])
def preprocessing():
    connection = get_db_connection()
    
    with connection.cursor() as cursor:
            # Ambil data sentimen dari database
            cursor.execute("SELECT created_at, username, full_text FROM data_sentimen")
            data_sentimen = cursor.fetchall()
    connection.close()
    
    df = pd.DataFrame(data_sentimen)
    
    # ================================================================ CLEANING ================================================================ #
    # Cleaning full_text
    df['username'] = df['username'].astype('str')
    df['full_text'] = df['full_text'].astype('str')
    
    def clean_x_text(text):
        # Menghapus mention
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        # Menghapus hashtag
        text = re.sub(r'#\w+', '', text)
        # Menghapus retweet indicator
        text = re.sub(r'RT[\s]+', '', text)
        # Menghapus URL
        text = re.sub(r'https?://\S+', '', text)
        # Menghapus format tanggal (contoh: 26/8/2024 atau 26.8.2024)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
        # Menghapus karakter selain huruf, angka, dan spasi
        text = re.sub(r'[^A-Za-z0-9 ]', '', text)
        # Menghapus spasi berlebihan dan menjaga spasi antar kata
        text = re.sub(r'\s+', ' ', text).strip()
    
        return text
    
    # Mengimplementasikan fungsi cleaning text pada data
    df['cleaned_text'] = df['full_text'].apply(clean_x_text)

    # Melakukan lower case text pada data
    df['cleaned_text'] = df['cleaned_text'].str.lower()
    
    # Menghapus adanya nilai kosong pada data
    df = df.dropna(subset=['created_at'])
    
    # Menghapus adanya duplikasi pada data
    df = df.drop_duplicates()
    # ================================================================== CLEANING =================================================================== #
    
    # ================================================================ NORMALISASI ================================================================ #
    # Membaca file kamus alay
    kamus_alay = pd.read_csv('lexicon/colloquial-indonesian-lexicon.csv', encoding='utf-8')

    # Konversi kamus alay ke dalam bentuk dictionary
    alay_dict = dict(zip(kamus_alay['slang'], kamus_alay['formal']))

    # Fungsi normalisasi menggunakan kamus alay
    def normalisasi(text):
        words = text.split()
        normalized_words = [alay_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)

    # Terapkan normalisasi pada kolom 'full_text'
    df['normalized_text'] = df['cleaned_text'].apply(normalisasi)
    # ================================================================ NORMALISASI ================================================================ #
    
    # ================================================================ TOKENISASI ================================================================= #
    # Mendefinisikan fungsi tokenize
    def tokenize(text):
        tokens = text.split()
        return tokens

    df['tokenized_text'] = df['normalized_text'].apply(tokenize)
    # ================================================================ TOKENISASI ================================================================= #
    
    # ================================================================= STOPWORD ================================================================== #
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

    df['stopword_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])
    # ================================================================= STOPWORD ================================================================== #
    
    # ================================================================= STEMMING ================================================================== #
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    df['stemmed_text'] = df['stopword_text'].apply(lambda x: [stemmer.stem(word) for word in x])
    df['preprocessing_text'] = df['stemmed_text'].apply(lambda tokens: ' '.join(tokens))
    # ================================================================= STEMMING ================================================================== #
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE data_klasifikasi")
        cursor.execute("TRUNCATE TABLE data_training")
        cursor.execute("TRUNCATE TABLE data_testing")
        
    # Bagi data menjadi 90% training dan 10% testing
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # Masukkan data klasifikasi
            data_klas = [
                (row["created_at"], row["username"], row["full_text"], row["preprocessing_text"])
                        for i, row in df.iterrows()
            ]
            if data_klas:
                cursor.executemany(
                    "INSERT INTO data_klasifikasi (created_at, username, full_text, preprocessing_text) VALUES (%s, %s, %s, %s)",
                            data_klas,
                )
            
            # Masukkan data training
            data_train = [
                (row["created_at"], row["username"], row["full_text"], row["preprocessing_text"])
                        for i, row in df_train.iterrows()
            ]
            if data_train:
                cursor.executemany(
                    "INSERT INTO data_training (created_at, username, full_text, preprocessing_text) VALUES (%s, %s, %s, %s)",
                            data_train,
                )
            
            # Masukkan data testing
            data_test = [
                (row["created_at"], row["username"], row["full_text"], row["preprocessing_text"])
                        for i, row in df_test.iterrows()
            ]
            if data_test:
                cursor.executemany(
                    "INSERT INTO data_testing (created_at, username, full_text, preprocessing_text) VALUES (%s, %s, %s, %s)",
                            data_test,
                )
                
            # Masukkan data implementasi
            data_implementasi = [
                (row["created_at"], row["username"], row["full_text"], row["preprocessing_text"])
                        for i, row in df_test.iterrows()
            ]
            if data_implementasi:
                cursor.executemany(
                    "INSERT INTO data_implementasi (created_at, username, full_text, preprocessing_text) VALUES (%s, %s, %s, %s)",
                            data_implementasi,
                )
        connection.commit()
    except Exception as e:
        connection.rollback()
        print(f"Error occurred: {str(e)}")  # Cetak kesalahan untuk debugging
        return jsonify({"message": str(e)}), 500
    finally:
        connection.close()
    
    return redirect(url_for("training.datatraining", status="preprocess_success")) 