from pathlib import Path

from flask import Flask

from .config import Config
from .models import db
from .routes import register_blueprints


def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config_class)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["PROCESSED_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["MODEL_DIR"]).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    register_blueprints(app)

    with app.app_context():
        db.create_all()

    return app
