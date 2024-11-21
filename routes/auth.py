from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from db import get_db_connection

auth = Blueprint('auth', __name__)


@auth.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = 'user'
        
        # Cek apakah password sama
        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('auth.register'))
    
        connection = get_db_connection()
        cursor = connection.cursor()
    
        # Cek apakah email sudah ada di database
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()
    
        if existing_user:
            flash("Email is already registered", "danger")
            return redirect(url_for('auth.register'))
    
        # Simpan user baru ke database dengan hashing
        # hashed_password = generate_password_hash(password)
        cursor.execute("INSERT INTO users (firstname, lastname, email, password, role) VALUES (%s, %s, %s, %s, %s)", (firstname, lastname, email, password, role))
        connection.commit()
        cursor.close()
        connection.close()
        
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('auth.login'))
        
    return render_template("auth-register.html")
    
@auth.route("/", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user and user['password'] == password:  # Password tanpa hashing
            session['email'] = email  # Menyimpan email di session
            session['role'] = user['role']  # Menyimpan role di session
            session.permanent = True
            flash("Login successful!", "success")
            
            # Redirect berdasarkan role
            if user['role'] == 'super':
                return redirect(url_for('main.indexSuper'))
            elif user['role'] == 'admin':
                return redirect(url_for('main.indexAdmin'))
            elif user['role'] == 'user':
                return redirect(url_for('main.indexUser'))
        else:
            flash("Invalid email or password", "danger")
        
        conn.close()
        return redirect(url_for('auth.login'))
    
    return render_template("auth-login.html")

@auth.route("/logout")
def logout():
    session.clear()
    
    flash("You have been logged out.", "success")
    return redirect(url_for('auth.login'))