from flask import Blueprint, render_template, session, redirect, url_for, flash
from db import get_db_connection

from .decorators import role_required

manajemen = Blueprint("manajemen", __name__)

@manajemen.route("/super/manajemenpengguna")
@role_required('super')
def manajemenSuper():
    if 'email' not in session:
        return redirect(url_for('auth.login'))
    
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT firstname, lastname, role, email FROM users WHERE role = 'user'")
        data = cursor.fetchall()
    connection.close()
    
    return render_template("super/manajemen-pengguna-super.html", data=data)

@manajemen.route("/delete_user/<string:email>", methods=['POST'])
def delete_user(email):
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM users WHERE email = %s", (email,))
        connection.commit()
    connection.close()
    flash("User deleted successfully!", "success")
    return redirect(url_for('manajemen.manajemenSuper'))