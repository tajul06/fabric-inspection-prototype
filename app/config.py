import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent

    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR / 'fabric_inspection.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024

    UPLOAD_FOLDER = BASE_DIR / "uploads"
    PROCESSED_FOLDER = BASE_DIR / "processed"
    MODEL_DIR = BASE_DIR / "models"

    ALLOWED_EXTENSIONS = {
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff",
        "webp",
    }
