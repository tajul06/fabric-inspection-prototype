from datetime import datetime

from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class InspectionRecord(db.Model):
    __tablename__ = "inspection_records"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    source_filename = db.Column(db.String(255), nullable=False)
    result_id = db.Column(db.String(64), unique=True, nullable=False)

    fabric_type = db.Column(db.String(32), nullable=False)
    fabric_confidence = db.Column(db.Float, nullable=False, default=0.0)

    anomaly_score = db.Column(db.Float, nullable=False, default=0.0)
    anomaly_class = db.Column(db.String(64), nullable=True)
    anomaly_confidence = db.Column(db.Float, nullable=False, default=0.0)

    four_point_score = db.Column(db.Integer, nullable=False, default=0)
    decision = db.Column(db.String(16), nullable=False, default="Hold")


class FabricProfile(db.Model):
    __tablename__ = "fabric_profiles"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    description = db.Column(db.String(255), nullable=True)

    image_threshold = db.Column(db.Float, nullable=False, default=0.995)
    pixel_threshold = db.Column(db.Float, nullable=False, default=0.996)
    min_component_area = db.Column(db.Float, nullable=False, default=0.001)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    active = db.Column(db.Boolean, nullable=False, default=True)


class AdaptationRun(db.Model):
    __tablename__ = "adaptation_runs"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    shots = db.Column(db.Integer, nullable=False)
    fabric_type = db.Column(db.String(32), nullable=False)
    status = db.Column(db.String(16), nullable=False, default="completed")
    support_samples = db.Column(db.Integer, nullable=False, default=0)
    notes = db.Column(db.String(255), nullable=True)
