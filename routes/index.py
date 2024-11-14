from flask import Blueprint, render_template, session, redirect, url_for
from db import get_db_connection

from .decorators import role_required

main = Blueprint("main", __name__)

# Super Admin
@main.route("/super/")
@role_required('super')
def indexSuper():
    if 'email' not in session:
        return redirect(url_for('auth.login')) 
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS jumlah FROM data_training")
        jumlah_training = cursor.fetchone()["jumlah"]

        cursor.execute("SELECT COUNT(*) AS jumlah FROM data_testing")
        jumlah_testing = cursor.fetchone()["jumlah"]
        
        cursor.execute("SELECT COUNT(*) AS jumlah FROM data_sentimen")
        jumlah_total = cursor.fetchone()["jumlah"]

    connection.close()

    return render_template(
        "super/index-super.html",
        jumlah_training=jumlah_training,
        jumlah_testing=jumlah_testing,
        jumlah_total=jumlah_total,
    )


# Admin
@main.route("/admin/")
@role_required('admin')
def indexAdmin():
    if 'email' not in session:
        return redirect(url_for('auth.login')) 
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS jumlah FROM data_training")
        jumlah_training = cursor.fetchone()["jumlah"]

        cursor.execute("SELECT COUNT(*) AS jumlah FROM data_testing")
        jumlah_testing = cursor.fetchone()["jumlah"]
        
        cursor.execute("SELECT COUNT(*) AS jumlah FROM data_sentimen")
        jumlah_total = cursor.fetchone()["jumlah"]

    connection.close()

    return render_template(
        "admin/index-admin.html",
        jumlah_training=jumlah_training,
        jumlah_testing=jumlah_testing,
        jumlah_total=jumlah_total,
    )


# User
@main.route("/user/")
@role_required('user')
def indexUser():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS jumlah FROM data_training")
        jumlah_training = cursor.fetchone()["jumlah"]

        cursor.execute("SELECT COUNT(*) AS jumlah FROM data_testing")
        jumlah_testing = cursor.fetchone()["jumlah"]
        
        cursor.execute("SELECT COUNT(*) AS jumlah FROM data_sentimen")
        jumlah_total = cursor.fetchone()["jumlah"]

    connection.close()

    return render_template(
        "user/index-user.html",
        jumlah_training=jumlah_training,
        jumlah_testing=jumlah_testing,
        jumlah_total=jumlah_total,
    )
