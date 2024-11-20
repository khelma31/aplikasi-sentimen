from flask import Blueprint, render_template, session, redirect, url_for
import json

from db import get_db_connection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from wordcloud import WordCloud
import os

from .decorators import role_required

implementasi = Blueprint("implementasi", __name__)

# Super Admin
@implementasi.route("/super/implementasihasil", methods=["GET", "POST"])
@role_required('super')
def implementasihasilSuper():
    # Cek apakah pengguna sudah login
    if 'email' not in session:
        return redirect(url_for('auth.login'))  # Arahkan ke halaman login jika belum login

    
    # Ambil data testing dari database
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT preprocessing_text, label FROM data_klasifikasi")
        data = cursor.fetchall()
    connection.close()
    
    df = pd.DataFrame(data, columns=['preprocessing_text', 'label'])

    # ========================================== TF-IDF ================================================

    documents = df['preprocessing_text'].tolist()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # ================================== Implementasi Random Forest ====================================

    # Pisahkan data untuk training dan testing
    X = tfidf_matrix  # Matriks TF-IDF
    y = df['label']   # Kelas sentimen (label)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi dan latih model Random Forest
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)  # Melatih model menggunakan data training

    # ======================================== Evaluasi Model ==========================================
    
    y_pred = rf_model.predict(X_test)  # Melakukan prediksi
    
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Hapus macro avg dan weighted avg
    report.pop('macro avg', None)
    report.pop('weighted avg', None)

    # Ubah nilai ke persentase
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Pastikan hanya memproses dictionary
            metrics['precision'] = round(metrics['precision'] * 100, 2)
            metrics['recall'] = round(metrics['recall'] * 100, 2)
            metrics['f1-score'] = round(metrics['f1-score'] * 100, 2)
    
    # ====================================== Confusion Matrix ==========================================
    
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

    # Membuat dan menyimpan confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
    cm_display.plot(cmap='Blues', values_format='d')
    confusion_matrix_path = os.path.join("static", "images", "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # =========================================== WordCloud ============================================
    
    # Mengambil teks untuk kelas 'positive', 'negative', 'neutral'
    positive_text = ' '.join(df[df['label'] == 'Positif']['preprocessing_text'])
    negative_text = ' '.join(df[df['label'] == 'Negatif']['preprocessing_text'])
    neutral_text = ' '.join(df[df['label'] == 'Netral']['preprocessing_text'])

    # Membuat WordCloud untuk setiap kelas
    positive_wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(positive_text)
    negative_wc = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(negative_text)
    neutral_wc = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(neutral_text)

    # Simpan gambar WordCloud ke static folder
    positive_wc_path = 'static/images/wordcloud_positive.png'
    negative_wc_path = 'static/images/wordcloud_negative.png'
    neutral_wc_path = 'static/images/wordcloud_neutral.png'
    
    positive_wc.to_file(positive_wc_path)
    negative_wc.to_file(negative_wc_path)
    neutral_wc.to_file(neutral_wc_path)

    # Buat dictionary untuk wordcloud images
    wordcloud_images = {
        'positive': 'images/wordcloud_positive.png',
        'negative': 'images/wordcloud_negative.png',
        'neutral': 'images/wordcloud_neutral.png'
    }

    # Kirim hasil ke template
    return render_template("super/hasil-analisis-super.html", accuracy=accuracy, report=report, wordcloud_images=wordcloud_images)

# Admin
@implementasi.route("/admin/implementasihasil", methods=["GET", "POST"])
@role_required('admin')
def implementasihasilAdmin():
    # Cek apakah pengguna sudah login
    if 'email' not in session:
        return redirect(url_for('auth.login'))  # Arahkan ke halaman login jika belum login

    
    # Ambil data testing dari database
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT preprocessing_text, label FROM data_klasifikasi")
        data = cursor.fetchall()
    connection.close()
    
    df = pd.DataFrame(data, columns=['preprocessing_text', 'label'])

    # ========================================== TF-IDF ================================================

    documents = df['preprocessing_text'].tolist()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # ================================== Implementasi Random Forest ====================================

    # Pisahkan data untuk training dan testing
    X = tfidf_matrix  # Matriks TF-IDF
    y = df['label']   # Kelas sentimen (label)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi dan latih model Random Forest
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)  # Melatih model menggunakan data training

    # ======================================== Evaluasi Model ==========================================
    
    y_pred = rf_model.predict(X_test)  # Melakukan prediksi
    
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Hapus macro avg dan weighted avg
    report.pop('macro avg', None)
    report.pop('weighted avg', None)

    # Ubah nilai ke persentase
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Pastikan hanya memproses dictionary
            metrics['precision'] = round(metrics['precision'] * 100, 2)
            metrics['recall'] = round(metrics['recall'] * 100, 2)
            metrics['f1-score'] = round(metrics['f1-score'] * 100, 2)
    
    # ====================================== Confusion Matrix ==========================================
    
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

    # Membuat dan menyimpan confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
    cm_display.plot(cmap='Blues', values_format='d')
    confusion_matrix_path = os.path.join("static", "images", "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # =========================================== WordCloud ============================================
    
    # Mengambil teks untuk kelas 'positive', 'negative', 'neutral'
    positive_text = ' '.join(df[df['label'] == 'Positif']['preprocessing_text'])
    negative_text = ' '.join(df[df['label'] == 'Negatif']['preprocessing_text'])
    neutral_text = ' '.join(df[df['label'] == 'Netral']['preprocessing_text'])

    # Membuat WordCloud untuk setiap kelas
    positive_wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(positive_text)
    negative_wc = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(negative_text)
    neutral_wc = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(neutral_text)

    # Simpan gambar WordCloud ke static folder
    positive_wc_path = 'static/images/wordcloud_positive.png'
    negative_wc_path = 'static/images/wordcloud_negative.png'
    neutral_wc_path = 'static/images/wordcloud_neutral.png'
    
    positive_wc.to_file(positive_wc_path)
    negative_wc.to_file(negative_wc_path)
    neutral_wc.to_file(neutral_wc_path)

    # Buat dictionary untuk wordcloud images
    wordcloud_images = {
        'positive': 'images/wordcloud_positive.png',
        'negative': 'images/wordcloud_negative.png',
        'neutral': 'images/wordcloud_neutral.png'
    }

    # Kirim hasil ke template
    return render_template("admin/hasil-analisis-admin.html", accuracy=accuracy, report=report, wordcloud_images=wordcloud_images)


# User
@implementasi.route("/user/implementasihasil", methods=["GET", "POST"])
@role_required('user')
def implementasihasilUser():
    # Cek apakah pengguna sudah login
    if 'email' not in session:
        return redirect(url_for('auth.login'))  # Arahkan ke halaman login jika belum login

    
    # Ambil data testing dari database
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT preprocessing_text, label FROM data_klasifikasi")
        data = cursor.fetchall()
    connection.close()
    
    df = pd.DataFrame(data, columns=['preprocessing_text', 'label'])

    # ========================================== TF-IDF ================================================

    documents = df['preprocessing_text'].tolist()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # ================================== Implementasi Random Forest ====================================

    # Pisahkan data untuk training dan testing
    X = tfidf_matrix  # Matriks TF-IDF
    y = df['label']   # Kelas sentimen (label)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi dan latih model Random Forest
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)  # Melatih model menggunakan data training

    # ======================================== Evaluasi Model ==========================================
    
    y_pred = rf_model.predict(X_test)  # Melakukan prediksi
    
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Hapus macro avg dan weighted avg
    report.pop('macro avg', None)
    report.pop('weighted avg', None)

    # Ubah nilai ke persentase
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Pastikan hanya memproses dictionary
            metrics['precision'] = round(metrics['precision'] * 100, 2)
            metrics['recall'] = round(metrics['recall'] * 100, 2)
            metrics['f1-score'] = round(metrics['f1-score'] * 100, 2)
    
    # ====================================== Confusion Matrix ==========================================
    
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

    # Membuat dan menyimpan confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
    cm_display.plot(cmap='Blues', values_format='d')
    confusion_matrix_path = os.path.join("static", "images", "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # =========================================== WordCloud ============================================
    
    # Mengambil teks untuk kelas 'positive', 'negative', 'neutral'
    positive_text = ' '.join(df[df['label'] == 'Positif']['preprocessing_text'])
    negative_text = ' '.join(df[df['label'] == 'Negatif']['preprocessing_text'])
    neutral_text = ' '.join(df[df['label'] == 'Netral']['preprocessing_text'])

    # Membuat WordCloud untuk setiap kelas
    positive_wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(positive_text)
    negative_wc = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(negative_text)
    neutral_wc = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(neutral_text)

    # Simpan gambar WordCloud ke static folder
    positive_wc_path = 'static/images/wordcloud_positive.png'
    negative_wc_path = 'static/images/wordcloud_negative.png'
    neutral_wc_path = 'static/images/wordcloud_neutral.png'
    
    positive_wc.to_file(positive_wc_path)
    negative_wc.to_file(negative_wc_path)
    neutral_wc.to_file(neutral_wc_path)

    # Buat dictionary untuk wordcloud images
    wordcloud_images = {
        'positive': 'images/wordcloud_positive.png',
        'negative': 'images/wordcloud_negative.png',
        'neutral': 'images/wordcloud_neutral.png'
    }

    # Kirim hasil ke template
    return render_template("user/hasil-analisis-user.html", accuracy=accuracy, report=report, wordcloud_images=wordcloud_images)