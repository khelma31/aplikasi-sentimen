from routes.index import main
from routes.datasentimen import sentimen
from routes.datatraining import training
from routes.datatesting import testing
from routes.preprocessing import proses
from routes.hapusdata import hapus
from routes.implementasi import implementasi
from routes.auth import auth
from routes.manajemen import manajemen


def register_blueprints(app):
    app.register_blueprint(main)
    app.register_blueprint(sentimen)
    app.register_blueprint(training)
    app.register_blueprint(testing)
    app.register_blueprint(proses)
    app.register_blueprint(hapus)
    app.register_blueprint(implementasi)
    app.register_blueprint(auth)
    app.register_blueprint(manajemen)
    