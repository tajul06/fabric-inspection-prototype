from flask import Flask

from .adapt import adapt_bp
from .main import main_bp
from .profiles import profiles_bp


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(main_bp)
    app.register_blueprint(adapt_bp)
    app.register_blueprint(profiles_bp)
