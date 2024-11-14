from flask import Blueprint, render_template, session, redirect, url_for
from db import get_db_connection

from .decorators import role_required

testing = Blueprint("testing", __name__)

# Super Admin
@testing.route("/super/datatesting")
@role_required('super')
def datatestingSuper():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, preprocessing_text FROM data_testing")
        data = cursor.fetchall()
    connection.close()
    
    return render_template("super/pre-data-testing-super.html", data=data)

@testing.route("/super/klastesting")
@role_required('super')
def klastrestingSuper():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, label FROM data_testing")
        data = cursor.fetchall()
    connection.close()
    return render_template("super/klas-data-testing-super.html", data=data)


# Admin
@testing.route("/admin/datatesting")
@role_required('admin')
def datatestingAdmin():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, preprocessing_text FROM data_testing")
        data = cursor.fetchall()
    connection.close()
    
    return render_template("admin/pre-data-testing-admin.html", data=data)

@testing.route("/admin/klastesting")
@role_required('admin')
def klastrestingAdmin():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, label FROM data_testing")
        data = cursor.fetchall()
    connection.close()
    return render_template("admin/klas-data-testing-admin.html", data=data)


# User
@testing.route("/user/datatesting")
@role_required('user')
def datatestingUser():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, preprocessing_text FROM data_testing")
        data = cursor.fetchall()
    connection.close()
    
    return render_template("user/pre-data-testing-user.html", data=data)

@testing.route("/user/klastesting")
@role_required('user')
def klastrestingUser():
    
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT full_text, label FROM data_testing")
        data = cursor.fetchall()
    connection.close()
    return render_template("user/klas-data-testing-user.html", data=data)
