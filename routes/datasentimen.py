import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session
from db import get_db_connection

from .decorators import role_required

sentimen = Blueprint("sentimen", __name__)


# Super Admin
@sentimen.route("/super/datasentimen", methods=["GET", "POST"])
@role_required('super')
def datasentimenSuper():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]

        if file and file.filename.endswith(".csv"):

            df = pd.read_csv(file)
            
            connection = get_db_connection()
            try:
                with connection.cursor() as cursor:
                    
                    cursor.execute("TRUNCATE TABLE data_sentimen")
                    cursor.execute("TRUNCATE TABLE data_training")
                    cursor.execute("TRUNCATE TABLE data_testing")
                    cursor.execute("TRUNCATE TABLE data_klasifikasi")
                    
                    data_sentimen = [
                        (row["created_at"], row["username"], row["full_text"], row.get("label", None))
                        for i, row in df.iterrows()
                    ]
                    if data_sentimen:
                        cursor.executemany(
                            "INSERT INTO data_sentimen (created_at, username, full_text, label) VALUES (%s, %s, %s, %s)",
                            data_sentimen,
                        )
                connection.commit()
            except Exception as e:
                connection.rollback()
                return jsonify({"message": str(e)}), 500
            finally:
                connection.close()
                
            return redirect(url_for("sentimen.datasentimenSuper", status="import_success")) 
        else:
            return jsonify({"message": "Format file tidak valid."}), 400
        
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, created_at, username, full_text, label FROM data_sentimen")
        data = cursor.fetchall()
    connection.close()
        
    return render_template("super/data-sentimen-super.html", data=data)


# Admin
@sentimen.route("/admin/datasentimen", methods=["GET", "POST"])
@role_required('admin')
def datasentimenAdmin():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]

        if file and file.filename.endswith(".csv"):
            
            df = pd.read_csv(file)
            
            connection = get_db_connection()
            try:
                with connection.cursor() as cursor:
                    
                    cursor.execute("TRUNCATE TABLE data_sentimen")
                    cursor.execute("TRUNCATE TABLE data_training")
                    cursor.execute("TRUNCATE TABLE data_testing")
                    cursor.execute("TRUNCATE TABLE data_klasifikasi")
                    
                    data_sentimen = [
                        (row["created_at"], row["username"], row["full_text"])
                        for i, row in df.iterrows()
                    ]
                    if data_sentimen:
                        cursor.executemany(
                            "INSERT INTO data_sentimen (created_at, username, full_text) VALUES (%s, %s, %s)",
                            data_sentimen,
                        )
                connection.commit()
            except Exception as e:
                connection.rollback()
                return jsonify({"message": str(e)}), 500
            finally:
                connection.close()
                
            return redirect(url_for("sentimen.datasentimenAdmin", status="import_success")) 
        else:
            return jsonify({"message": "Format file tidak valid."}), 400
        
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, created_at, username, full_text FROM data_sentimen")
        data = cursor.fetchall()
    connection.close()
        
    return render_template("admin/data-sentimen-admin.html", data=data)


# User
@sentimen.route("/user/datasentimen", methods=["GET", "POST"])
@role_required('user')
def datasentimenUser():
    if 'email' not in session:
        return redirect(url_for('auth.login'))

    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, created_at, username, full_text FROM data_sentimen")
        data = cursor.fetchall()
    connection.close()
        
    return render_template("user/data-sentimen-user.html", data=data)