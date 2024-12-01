from flask import Flask
from datetime import timedelta

# Routing
from routes import register_blueprints

app = Flask(__name__)
application = app

# Konfigurasi koneksi ke database
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_PORT"] = 3307
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "flask"
app.secret_key = "your_secret_key"

# Atur durasi session, misalnya 30 menit
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

register_blueprints(app)

if __name__ == "__main__":
    app.run(debug=True)