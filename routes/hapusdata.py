import os
from flask import Blueprint, redirect, url_for
from db import get_db_connection

hapus = Blueprint("hapus", __name__)

@hapus.route("/hapusdata", methods=["POST"])
def hapusdata():
    connection = get_db_connection()
    with connection.cursor() as cursor:
        # Hapus data sebelumnya pada semua tabel
        cursor.execute("TRUNCATE TABLE data_sentimen")
        cursor.execute("TRUNCATE TABLE data_training")
        cursor.execute("TRUNCATE TABLE data_testing")
        cursor.execute("TRUNCATE TABLE data_klasifikasi")
        cursor.execute("TRUNCATE TABLE data_implementasi")
    connection.commit()

    # Menghapus file gambar di static/images
    folder_images = 'static/images'  # Folder tempat gambar disimpan
    for filename in os.listdir(folder_images):
        file_path = os.path.join(folder_images, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Hapus file
        except Exception as e:
            print(f"Error saat menghapus file {file_path}: {e}")

    return redirect(url_for("sentimen.datasentimen", status="delete_success"))
