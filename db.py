from flask import current_app
import pymysql

# Inisialisasi koneksi PyMySQL
def get_db_connection():
    return pymysql.connect(
        host=current_app.config["MYSQL_HOST"],
        user=current_app.config["MYSQL_USER"],
        password=current_app.config["MYSQL_PASSWORD"],
        database=current_app.config["MYSQL_DB"],
        port=current_app.config["MYSQL_PORT"],
        cursorclass=pymysql.cursors.DictCursor,
    )