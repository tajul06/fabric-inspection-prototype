from pathlib import Path
import uuid

import numpy as np
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from ..ml_models import MLModelManager
from ..models import InspectionRecord, db
from ..utils import (
    allowed_file,
    apply_clahe_only,
    build_anomaly_mask,
    blend_heatmap_on_original,
    compute_four_point_score,
    draw_defect_bounding_boxes,
    read_bgr_image,
    safe_result_id,
    save_bgr_image,
    suggest_defect_type,
)


main_bp = Blueprint("main", __name__)


def parse_positive_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None

    value = raw_value.strip()
    if value == "":
        return None

    parsed = float(value)
    if parsed <= 0:
        raise ValueError("Value must be greater than zero.")
    return parsed


def parse_support_shots(raw_value: str | None) -> int:
    if raw_value is None:
        raise ValueError("Support shots must be 3, 5, or 10.")

    value = raw_value.strip()
    if value == "":
        raise ValueError("Support shots must be 3, 5, or 10.")

    shots = int(value)
    if shots not in {3, 5, 10}:
        raise ValueError("Support shots must be 3, 5, or 10.")
    return shots


def get_model_manager() -> MLModelManager:
    manager = current_app.extensions.get("ml_model_manager")
    if manager is None:
        manager = MLModelManager(Path(current_app.config["MODEL_DIR"]))
        current_app.extensions["ml_model_manager"] = manager
    return manager


@main_bp.route("/results/<path:filename>")
def result_file(filename: str):
    return send_from_directory(current_app.config["PROCESSED_FOLDER"], filename)


@main_bp.route("/", methods=["GET", "POST"])
def inspection():
    result = None

    if request.method == "POST":
        upload = request.files.get("image")

        measurement_mode = (request.form.get("measurement_mode", "auto_ratio") or "auto_ratio").strip().lower()
        if measurement_mode not in {"auto_ratio", "manual_area"}:
            measurement_mode = "auto_ratio"

        pattern_mode = (request.form.get("pattern_mode", "auto") or "auto").strip().lower()
        if pattern_mode not in {"auto", "unknown"}:
            pattern_mode = "auto"

        fabric_override = (request.form.get("fabric_override", "auto") or "auto").strip().lower()
        if fabric_override not in {"auto", "knitted", "woven"}:
            fabric_override = "auto"

        support_shots = 5
        if pattern_mode == "unknown":
            try:
                support_shots = parse_support_shots(request.form.get("support_shots", "5"))
            except ValueError as exc:
                flash(str(exc), "warning")
                support_shots = 5

        width_raw = request.form.get("fabric_width_cm", "")
        height_raw = request.form.get("fabric_height_cm", "")
        fabric_width_cm: float | None = None
        fabric_height_cm: float | None = None

        if measurement_mode == "manual_area":
            try:
                fabric_width_cm = parse_positive_float(width_raw)
                fabric_height_cm = parse_positive_float(height_raw)
            except ValueError:
                flash("Fabric width/height must be positive numbers. Switched to Auto Ratio.", "warning")
                measurement_mode = "auto_ratio"
                fabric_width_cm = None
                fabric_height_cm = None

            if (fabric_width_cm is None) != (fabric_height_cm is None):
                flash(
                    "Manual Area mode needs both width and height. Switched to Auto Ratio.",
                    "warning",
                )
                measurement_mode = "auto_ratio"
                fabric_width_cm = None
                fabric_height_cm = None

        if upload is None or upload.filename == "":
            flash("Please upload an image first.", "warning")
        elif not allowed_file(upload.filename, current_app.config["ALLOWED_EXTENSIONS"]):
            flash("Unsupported file type.", "danger")
        else:
            source_filename = secure_filename(upload.filename)
            stored_filename = f"{uuid.uuid4().hex}_{source_filename}"
            upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / stored_filename
            upload.save(upload_path)

            try:
                original = read_bgr_image(upload_path)
                preprocessed = apply_clahe_only(original)

                winclip_support: list[np.ndarray] = []
                if pattern_mode == "unknown":
                    support_files = request.files.getlist("support_images")
                    if not support_files or not any(file and file.filename for file in support_files):
                        raise ValueError("WinCLIP requires support images.")

                    for support_file in support_files:
                        if support_file is None or support_file.filename == "":
                            continue
                        if not allowed_file(support_file.filename, current_app.config["ALLOWED_EXTENSIONS"]):
                            continue

                        support_name = f"{uuid.uuid4().hex}_{secure_filename(support_file.filename)}"
                        support_path = Path(current_app.config["UPLOAD_FOLDER"]) / support_name
                        support_file.save(support_path)
                        winclip_support.append(apply_clahe_only(read_bgr_image(support_path)))

                    if len(winclip_support) < support_shots:
                        raise ValueError(f"WinCLIP requires {support_shots} support images.")

                manager = get_model_manager()
                inference = manager.inspect(
                    original,
                    preprocessed,
                    pattern_mode=pattern_mode,
                    fabric_override=fabric_override,
                    winclip_support=winclip_support,
                    winclip_shots=support_shots,
                )

                mask_threshold = 0.60

                result_id = uuid.uuid4().hex
                processed_dir = Path(current_app.config["PROCESSED_FOLDER"])

                original_name = f"{result_id}_original.png"
                preprocessed_name = f"{result_id}_preprocessed.png"
                heatmap_name = f"{result_id}_heatmap.png"
                bbox_name = f"{result_id}_bbox.png"
                blended_name = f"{result_id}_blend_45.png"
                anomaly_map_name = f"{result_id}_anomaly.npy"

                save_bgr_image(processed_dir / original_name, original)
                save_bgr_image(processed_dir / preprocessed_name, preprocessed)

                save_bgr_image(processed_dir / heatmap_name, inference["heatmap"])

                anomaly_mask = build_anomaly_mask(
                    inference["anomaly_map"],
                    threshold=mask_threshold,
                    apply_morphology=True,
                    as_uint8=True,
                )

                blended = blend_heatmap_on_original(
                    original,
                    inference["heatmap"],
                    alpha=0.45,
                    anomaly_map=inference["anomaly_map"],
                )
                save_bgr_image(processed_dir / blended_name, blended)

                np.save(processed_dir / anomaly_map_name, inference["anomaly_map"])

                four_point_score, decision, defects = compute_four_point_score(
                    inference["anomaly_map"],
                    fabric_width_cm=fabric_width_cm,
                    fabric_height_cm=fabric_height_cm,
                    anomaly_mask=anomaly_mask,
                )

                bbox_visual = draw_defect_bounding_boxes(original, defects)
                save_bgr_image(processed_dir / bbox_name, bbox_visual)

                total_area_cm2 = (
                    float(fabric_width_cm * fabric_height_cm)
                    if fabric_width_cm is not None and fabric_height_cm is not None
                    else None
                )

                record = InspectionRecord(
                    source_filename=source_filename,
                    result_id=result_id,
                    fabric_type=inference["fabric_type"],
                    fabric_confidence=inference["fabric_confidence"],
                    anomaly_score=inference["anomaly_score"],
                    anomaly_class=inference["anomaly_class"],
                    anomaly_confidence=inference["anomaly_confidence"],
                    four_point_score=four_point_score,
                    decision=decision,
                )
                db.session.add(record)
                db.session.commit()

                result = {
                    "result_id": result_id,
                    "source_filename": source_filename,
                    "measurement_mode": measurement_mode,
                    "fabric_type": inference["fabric_type"],
                    "fabric_confidence": inference["fabric_confidence"],
                    "pattern_type": inference.get("pattern_type", "plain"),
                    "pattern_confidence": inference.get("pattern_confidence", 0.5),
                    "pattern_mode": pattern_mode,
                    "fabric_override": fabric_override,
                    "support_shots": support_shots if pattern_mode == "unknown" else None,
                    "anomaly_route": inference.get("anomaly_route", f"{inference['fabric_type']}_fallback"),
                    "anomaly_score": inference["anomaly_score"],
                    "anomaly_class": inference["anomaly_class"],
                    "anomaly_confidence": inference["anomaly_confidence"],
                    "four_point_score": four_point_score,
                    "decision": decision,
                    "defects": defects,
                    "fabric_width_cm": fabric_width_cm,
                    "fabric_height_cm": fabric_height_cm,
                    "fabric_total_area_cm2": total_area_cm2,
                    "default_alpha": 0.45,
                    "heatmap_shape": list(inference["anomaly_map"].shape),
                    "urls": {
                        "original": url_for("main.result_file", filename=original_name),
                        "preprocessed": url_for(
                            "main.result_file", filename=preprocessed_name
                        ),
                        "heatmap": url_for("main.result_file", filename=heatmap_name),
                        "bbox": url_for("main.result_file", filename=bbox_name),
                        "blended": url_for("main.result_file", filename=blended_name),
                    },
                }
                flash("Inspection complete.", "success")
            except Exception as exc:
                current_app.logger.exception("Inspection failed: %s", exc)
                flash(f"Inspection failed: {exc}", "danger")

    recent_runs = InspectionRecord.query.order_by(InspectionRecord.id.desc()).limit(8).all()

    return render_template(
        "inspection.html",
        result=result,
        recent_runs=recent_runs,
    )


@main_bp.route("/api/blend", methods=["POST"])
def blend_api():
    payload = request.get_json(silent=True) or {}

    try:
        result_id = safe_result_id(str(payload.get("result_id", "")))
        alpha = float(payload.get("alpha", 0.45))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    alpha = max(0.0, min(alpha, 1.0))
    processed_dir = Path(current_app.config["PROCESSED_FOLDER"])

    original_path = processed_dir / f"{result_id}_original.png"
    heatmap_path = processed_dir / f"{result_id}_heatmap.png"
    anomaly_map_path = processed_dir / f"{result_id}_anomaly.npy"
    blend_name = f"{result_id}_blend_{int(alpha * 100):02d}.png"
    blend_path = processed_dir / blend_name

    if not original_path.exists() or not heatmap_path.exists() or not anomaly_map_path.exists():
        return jsonify({"error": "Missing result assets."}), 404

    original = read_bgr_image(original_path)
    heatmap = read_bgr_image(heatmap_path)
    anomaly_map = np.load(anomaly_map_path)
    blended = blend_heatmap_on_original(
        original,
        heatmap,
        alpha,
        anomaly_map=anomaly_map,
    )
    save_bgr_image(blend_path, blended)

    return jsonify(
        {
            "alpha": alpha,
            "url": url_for("main.result_file", filename=blend_name),
        }
    )


@main_bp.route("/api/local-score", methods=["POST"])
def local_score_api():
    payload = request.get_json(silent=True) or {}

    try:
        result_id = safe_result_id(str(payload.get("result_id", "")))
        x = float(payload.get("x", 0.0))
        y = float(payload.get("y", 0.0))
        view_width = max(1.0, float(payload.get("viewWidth", 1.0)))
        view_height = max(1.0, float(payload.get("viewHeight", 1.0)))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    anomaly_map_path = Path(current_app.config["PROCESSED_FOLDER"]) / f"{result_id}_anomaly.npy"
    if not anomaly_map_path.exists():
        return jsonify({"error": "Anomaly map not found."}), 404

    anomaly_map = np.load(anomaly_map_path)
    h, w = anomaly_map.shape[:2]

    map_x = int(np.clip((x / view_width) * (w - 1), 0, w - 1))
    map_y = int(np.clip((y / view_height) * (h - 1), 0, h - 1))

    local_score = float(np.clip(anomaly_map[map_y, map_x], 0.0, 1.0))

    record = InspectionRecord.query.filter_by(result_id=result_id).first()
    fabric_type = record.fabric_type if record else "woven"
    suggested = (
        record.anomaly_class
        if record and record.anomaly_class
        else suggest_defect_type(local_score, fabric_type)
    )

    return jsonify(
        {
            "x": map_x,
            "y": map_y,
            "local_score": local_score,
            "suggested_defect": suggested,
        }
    )


@main_bp.route("/api/pipeline-preview", methods=["POST"])
def pipeline_preview_api():
    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return jsonify({"error": "Please upload an image first."}), 400

    if not allowed_file(upload.filename, current_app.config["ALLOWED_EXTENSIONS"]):
        return jsonify({"error": "Unsupported file type."}), 400

    source_filename = secure_filename(upload.filename)
    stored_filename = f"{uuid.uuid4().hex}_{source_filename}"
    upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / stored_filename
    upload.save(upload_path)

    try:
        original = read_bgr_image(upload_path)
        preprocessed = apply_clahe_only(original)

        manager = get_model_manager()
        inference = manager.inspect(original, preprocessed)

        return jsonify(
            {
                "fabric_type": inference["fabric_type"],
                "fabric_confidence": float(inference["fabric_confidence"]),
                "pattern_type": inference.get("pattern_type", "plain"),
                "pattern_confidence": float(inference.get("pattern_confidence", 0.0)),
                "anomaly_route": inference.get("anomaly_route", "fallback"),
                "anomaly_score": float(inference["anomaly_score"]),
                "anomaly_class": inference["anomaly_class"],
                "anomaly_confidence": float(inference["anomaly_confidence"]),
            }
        )
    except Exception as exc:
        current_app.logger.exception("Pipeline preview failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            current_app.logger.debug("Could not clean temporary preview upload: %s", upload_path)
