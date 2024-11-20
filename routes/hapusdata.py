import os
from flask import Blueprint, redirect, url_for
from db import get_db_connection

from .decorators import role_required

hapus = Blueprint("hapus", __name__)

# Super Admin
@hapus.route("/super/hapusdata", methods=["POST"])
@role_required('super')
def hapusdataSuper():
    connection = get_db_connection()
    with connection.cursor() as cursor:
        # Hapus data sebelumnya pada semua tabel
        cursor.execute("TRUNCATE TABLE data_sentimen")
        cursor.execute("TRUNCATE TABLE data_training")
        cursor.execute("TRUNCATE TABLE data_testing")
        cursor.execute("TRUNCATE TABLE data_klasifikasi")
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

    return redirect(url_for("sentimen.datasentimenSuper", status="delete_success"))

# Admin
@hapus.route("/admin/hapusdata", methods=["POST"])
@role_required('admin')
def hapusdataAdmin():
    connection = get_db_connection()
    with connection.cursor() as cursor:
        # Hapus data sebelumnya pada semua tabel
        cursor.execute("TRUNCATE TABLE data_sentimen")
        cursor.execute("TRUNCATE TABLE data_training")
        cursor.execute("TRUNCATE TABLE data_testing")
        cursor.execute("TRUNCATE TABLE data_klasifikasi")
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

    return redirect(url_for("sentimen.datasentimenAdmin", status="delete_success"))
