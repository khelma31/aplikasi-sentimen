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

from .decorators import role_required

proses = Blueprint("preprocessing", __name__)

# Super Admin
@proses.route("/super/preprocessing", methods=["POST"])
@role_required('super')
def preprocessingSuper():
    connection = get_db_connection()
    
    with connection.cursor() as cursor:
            cursor.execute("SELECT created_at, username, full_text, label FROM data_sentimen")
            data_sentimen = cursor.fetchall()
    connection.close()
    
    df = pd.DataFrame(data_sentimen)
    
    # ================================================================ CLEANING ================================================================ #
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['username'] = df['username'].astype('str')
    df['full_text'] = df['full_text'].astype('str')
    
    def clean_x_text(text):
        
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'RT[\s]+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
        text = re.sub(r'[^A-Za-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    
        return text
    
    df['cleaned_text'] = df['full_text'].apply(clean_x_text)

    df['cleaned_text'] = df['cleaned_text'].str.lower()
    
    df = df.dropna(subset=['created_at'])
    
    # ================================================================ NORMALISASI ================================================================ #
    
    kamus_alay = pd.read_csv('lexicon/colloquial-indonesian-lexicon.csv', encoding='utf-8')

    alay_dict = dict(zip(kamus_alay['slang'], kamus_alay['formal']))

    def normalisasi(text):
        words = text.split()
        normalized_words = [alay_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)

    df['normalized_text'] = df['cleaned_text'].apply(normalisasi)
    
    # ================================================================ TOKENISASI ================================================================= #

    def tokenize(text):
        tokens = text.split()
        return tokens

    df['tokenized_text'] = df['normalized_text'].apply(tokenize)
    
    # ================================================================= STOPWORD ================================================================== #
   
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

    df['stopword_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])
    
    # ================================================================= STEMMING ================================================================== #
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    df['stemmed_text'] = df['stopword_text'].apply(lambda x: [stemmer.stem(word) for word in x])
    df['preprocessing_text'] = df['stemmed_text'].apply(lambda tokens: ' '.join(tokens))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE data_klasifikasi")
        cursor.execute("TRUNCATE TABLE data_training")
        cursor.execute("TRUNCATE TABLE data_testing")
        
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            
            data_klas = [
                (row["created_at"], row["username"], row["full_text"], row["preprocessing_text"], row["label"])
                        for i, row in df.iterrows()
            ]
            if data_klas:
                cursor.executemany(
                    "INSERT INTO data_klasifikasi (created_at, username, full_text, preprocessing_text, label) VALUES (%s, %s, %s, %s, %s)",
                            data_klas,
                )
            
            data_train = [
                (row["created_at"], row["username"], row["full_text"], row["preprocessing_text"], row["label"])
                        for i, row in df_train.iterrows()
            ]
            if data_train:
                cursor.executemany(
                    "INSERT INTO data_training (created_at, username, full_text, preprocessing_text, label) VALUES (%s, %s, %s, %s, %s)",
                            data_train,
                )
            
            data_test = [
                (row["created_at"], row["username"], row["full_text"], row["preprocessing_text"], row["label"])
                        for i, row in df_test.iterrows()
            ]
            if data_test:
                cursor.executemany(
                    "INSERT INTO data_testing (created_at, username, full_text, preprocessing_text, label) VALUES (%s, %s, %s, %s, %s)",
                            data_test,
                )
                
        connection.commit()
    except Exception as e:
        connection.rollback()
        print(f"Error occurred: {str(e)}")
        return jsonify({"message": str(e)}), 500
    finally:
        connection.close()
    
    return redirect(url_for("training.datatrainingSuper", status="preprocess_success"))

# Admin
@proses.route("/admin/preprocessing", methods=["POST"])
@role_required('admin')
def preprocessingAdmin():
    connection = get_db_connection()
    
    with connection.cursor() as cursor:
            cursor.execute("SELECT created_at, username, full_text, label FROM data_sentimen")
            data_sentimen = cursor.fetchall()
    connection.close()
    
    df = pd.DataFrame(data_sentimen)
    
    # ================================================================ CLEANING ================================================================ #
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['username'] = df['username'].astype('str')
    df['full_text'] = df['full_text'].astype('str')
    
    def clean_x_text(text):
        
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'RT[\s]+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
        text = re.sub(r'[^A-Za-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    
        return text
    
    df['cleaned_text'] = df['full_text'].apply(clean_x_text)

    df['cleaned_text'] = df['cleaned_text'].str.lower()
    
    df = df.dropna(subset=['created_at'])
    
    # ================================================================ NORMALISASI ================================================================ #
    
    kamus_alay = pd.read_csv('lexicon/colloquial-indonesian-lexicon.csv', encoding='utf-8')

    alay_dict = dict(zip(kamus_alay['slang'], kamus_alay['formal']))

    def normalisasi(text):
        words = text.split()
        normalized_words = [alay_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)

    df['normalized_text'] = df['cleaned_text'].apply(normalisasi)
    
    # ================================================================ TOKENISASI ================================================================= #

    def tokenize(text):
        tokens = text.split()
        return tokens

    df['tokenized_text'] = df['normalized_text'].apply(tokenize)
    
    # ================================================================= STOPWORD ================================================================== #
   
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

    df['stopword_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])
    
    # ================================================================= STEMMING ================================================================== #
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    df['stemmed_text'] = df['stopword_text'].apply(lambda x: [stemmer.stem(word) for word in x])
    df['preprocessing_text'] = df['stemmed_text'].apply(lambda tokens: ' '.join(tokens))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE data_klasifikasi")
        cursor.execute("TRUNCATE TABLE data_training")
        cursor.execute("TRUNCATE TABLE data_testing")
        
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            
            data_klas = [
                (row["created_at"], row["username"], row["full_text"], row["preprocessing_text"], row["label"])
                        for i, row in df.iterrows()
            ]
            if data_klas:
                cursor.executemany(
                    "INSERT INTO data_klasifikasi (created_at, username, full_text, preprocessing_text, label) VALUES (%s, %s, %s, %s, %s)",
                            data_klas,
                )
            
            data_train = [
                (row["created_at"], row["username"], row["full_text"], row["preprocessing_text"], row["label"])
                        for i, row in df_train.iterrows()
            ]
            if data_train:
                cursor.executemany(
                    "INSERT INTO data_training (created_at, username, full_text, preprocessing_text, label) VALUES (%s, %s, %s, %s, %s)",
                            data_train,
                )
            
            data_test = [
                (row["created_at"], row["username"], row["full_text"], row["preprocessing_text"], row["label"])
                        for i, row in df_test.iterrows()
            ]
            if data_test:
                cursor.executemany(
                    "INSERT INTO data_testing (created_at, username, full_text, preprocessing_text, label) VALUES (%s, %s, %s, %s, %s)",
                            data_test,
                )
                
        connection.commit()
    except Exception as e:
        connection.rollback()
        print(f"Error occurred: {str(e)}")
        return jsonify({"message": str(e)}), 500
    finally:
        connection.close()
    
    return redirect(url_for("training.datatrainingAdmin", status="preprocess_success"))