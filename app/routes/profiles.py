from sqlalchemy import func
from flask import Blueprint, flash, redirect, render_template, request, url_for

from ..models import FabricProfile, InspectionRecord, db


profiles_bp = Blueprint("profiles", __name__, url_prefix="/profiles")


@profiles_bp.route("/", methods=["GET", "POST"])
def profiles_home():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Profile name is required.", "warning")
            return redirect(url_for("profiles.profiles_home"))

        description = request.form.get("description", "").strip()
        image_threshold = float(request.form.get("image_threshold", "0.995"))
        pixel_threshold = float(request.form.get("pixel_threshold", "0.996"))
        min_component_area = float(request.form.get("min_component_area", "0.001"))

        existing = FabricProfile.query.filter(
            func.lower(FabricProfile.name) == name.lower()
        ).first()
        if existing:
            flash("Profile name already exists.", "warning")
            return redirect(url_for("profiles.profiles_home"))

        profile = FabricProfile(
            name=name,
            description=description,
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
            min_component_area=min_component_area,
            active=True,
        )
        db.session.add(profile)
        db.session.commit()

        flash("Profile created.", "success")
        return redirect(url_for("profiles.profiles_home"))

    profiles = FabricProfile.query.order_by(FabricProfile.id.desc()).all()
    return render_template("profiles.html", profiles=profiles)


@profiles_bp.route("/dashboard")
def dashboard():
    total_inspections = db.session.query(func.count(InspectionRecord.id)).scalar() or 0
    avg_anomaly_score = db.session.query(func.avg(InspectionRecord.anomaly_score)).scalar() or 0

    decision_counts = (
        db.session.query(
            InspectionRecord.decision,
            func.count(InspectionRecord.id),
        )
        .group_by(InspectionRecord.decision)
        .all()
    )

    fabric_counts = (
        db.session.query(
            InspectionRecord.fabric_type,
            func.count(InspectionRecord.id),
        )
        .group_by(InspectionRecord.fabric_type)
        .all()
    )

    recent = InspectionRecord.query.order_by(InspectionRecord.id.desc()).limit(10).all()

    return render_template(
        "dashboard.html",
        total_inspections=total_inspections,
        avg_anomaly_score=float(avg_anomaly_score),
        decision_counts=decision_counts,
        fabric_counts=fabric_counts,
        recent=recent,
    )
