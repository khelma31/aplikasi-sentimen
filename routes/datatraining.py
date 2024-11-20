from flask import Blueprint, render_template, session, redirect, url_for
from db import get_db_connection

from .decorators import role_required

training = Blueprint("training", __name__)


# Super Admin
@training.route("/super/datatraining")
@role_required('super')
def datatrainingSuper():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, preprocessing_text FROM data_training")
        data = cursor.fetchall()
    connection.close()
    
    return render_template("super/pre-data-training-super.html", data=data)

@training.route("/super/klastraining")
@role_required('super')
def klastrainingSuper():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, label FROM data_training")
        data = cursor.fetchall()
    connection.close()
    return render_template("super/klas-data-training-super.html", status="label_success", data=data)


# Admin
@training.route("/admin/datatraining")
@role_required('admin')
def datatrainingAdmin():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, preprocessing_text FROM data_training")
        data = cursor.fetchall()
    connection.close()
    
    return render_template("admin/pre-data-training-admin.html", data=data)

@training.route("/admin/klastraining")
@role_required('admin')
def klastrainingAdmin():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, label FROM data_training")
        data = cursor.fetchall()
    connection.close()
    return render_template("admin/klas-data-training-admin.html", data=data)


# User
@training.route("/user/datatraining")
@role_required('user')
def datatrainingUser():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, preprocessing_text FROM data_training")
        data = cursor.fetchall()
    connection.close()
    
    return render_template("user/pre-data-training-user.html", data=data)

@training.route("/user/klastraining")
@role_required('user')
def klastrainingUser():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, label FROM data_training")
        data = cursor.fetchall()
    connection.close()
    return render_template("user/klas-data-training-user.html", data=data)
