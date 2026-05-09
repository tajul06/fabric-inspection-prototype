from pathlib import Path
import uuid

import numpy as np
from flask import Blueprint, current_app, flash, render_template, request
from werkzeug.utils import secure_filename

from ..models import AdaptationRun, db
from ..ml_models import MLModelManager
from ..utils import allowed_file, apply_clahe_only, read_bgr_image


adapt_bp = Blueprint("adapt", __name__, url_prefix="/adapt")


def _normalize_support_fabric(raw_value: str) -> str:
    normalized = raw_value.strip().lower()
    if normalized in {"woven", "weave"}:
        return "woven"
    if normalized in {"knit", "knitted", "knitten"}:
        return "knitted"
    return "knitted"


def _resolve_support_root(
    manager: MLModelManager,
    upload_root: Path,
    fabric_type: str,
) -> Path:
    normalized = _normalize_support_fabric(fabric_type)

    if normalized == "woven":
        if getattr(manager, "woven_proto_classifier", None) is not None:
            return manager.woven_proto_classifier.support_root

        candidates = [
            manager.project_root / "woven  anomaly classifier" / "support",
            manager.project_root / "woven anomaly classifier" / "support",
            manager.project_root / "woven_anomaly_classifier" / "support",
            manager.project_root.parent / "woven  anomaly classifier" / "support",
            manager.project_root.parent / "woven anomaly classifier" / "support",
            manager.project_root.parent / "woven_anomaly_classifier" / "support",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        return upload_root / "woven  anomaly classifier" / "support"

    if manager.knitted_proto_classifier is not None:
        return manager.knitted_proto_classifier.support_root

    candidates = [
        manager.project_root / "knitten anomaly classifier" / "support",
        manager.project_root / "knitted anomaly classifier" / "support",
        manager.project_root.parent / "knitten anomaly classifier" / "support",
        manager.project_root.parent / "knitted anomaly classifier" / "support",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return upload_root / "knitten anomaly classifier" / "support"


def get_model_manager() -> MLModelManager:
    manager = current_app.extensions.get("ml_model_manager")
    if manager is None:
        manager = MLModelManager(Path(current_app.config["MODEL_DIR"]))
        current_app.extensions["ml_model_manager"] = manager
    return manager


@adapt_bp.route("/", methods=["GET", "POST"])
def adapt_home():
    result = None
    if request.method == "POST":
        action = (request.form.get("adapt_action", "detector") or "detector").strip().lower()
        upload = request.files.get("image")
        normal_sample_uploads = request.files.getlist("normal_samples")
        support_set_uploads = request.files.getlist("support_set")
        support_class_raw = request.form.get("support_class", "").strip()
        support_fabric_raw = request.form.get("support_fabric", "")
        notes = request.form.get("notes", "").strip()
        manager = get_model_manager()

        support_class = secure_filename(support_class_raw).strip().lower()
        support_fabric = _normalize_support_fabric(support_fabric_raw)
        if action == "classifier":
            support_set_added = 0
            if support_set_uploads and any(file and file.filename for file in support_set_uploads):
                if not support_class:
                    flash("Support class name is required when uploading support set images.", "warning")
                else:
                    support_root = _resolve_support_root(
                        manager,
                        Path(current_app.config["UPLOAD_FOLDER"]).parent,
                        support_fabric,
                    )
                    class_dir = support_root / support_class
                    class_dir.mkdir(parents=True, exist_ok=True)

                    for support_file in support_set_uploads:
                        if support_file is None or support_file.filename == "":
                            continue
                        if not allowed_file(support_file.filename, current_app.config["ALLOWED_EXTENSIONS"]):
                            continue

                        support_name = f"{uuid.uuid4().hex}_{secure_filename(support_file.filename)}"
                        support_file.save(class_dir / support_name)
                        support_set_added += 1

                    if support_set_added > 0:
                        flash(
                            f"Added {support_set_added} support images to {support_fabric} class '{support_class}'.",
                            "success",
                        )

            if upload is None or upload.filename == "":
                if support_set_added > 0:
                    run = AdaptationRun(
                        shots=5,
                        fabric_type=support_fabric,
                        support_samples=support_set_added,
                        status="support_updated",
                        notes=notes or f"{support_fabric} support class '{support_class}' updated",
                    )
                    db.session.add(run)
                    db.session.commit()
                else:
                    flash("Please upload a query image or support set images.", "warning")
            elif not allowed_file(upload.filename, current_app.config["ALLOWED_EXTENSIONS"]):
                flash("Unsupported query image type.", "danger")
            else:
                query_name = f"{uuid.uuid4().hex}_{secure_filename(upload.filename)}"
                query_path = Path(current_app.config["UPLOAD_FOLDER"]) / query_name
                upload.save(query_path)

                sample_paths: list[Path] = []
                normal_samples_bgr: list[np.ndarray] = []

                try:
                    original = read_bgr_image(query_path)
                    preprocessed = apply_clahe_only(original)

                    for sample in normal_sample_uploads:
                        if sample is None or sample.filename == "":
                            continue
                        if not allowed_file(sample.filename, current_app.config["ALLOWED_EXTENSIONS"]):
                            continue

                        sample_name = f"{uuid.uuid4().hex}_{secure_filename(sample.filename)}"
                        sample_path = Path(current_app.config["UPLOAD_FOLDER"]) / sample_name
                        sample.save(sample_path)
                        sample_paths.append(sample_path)
                        normal_samples_bgr.append(apply_clahe_only(read_bgr_image(sample_path)))

                    result = manager.inspect(original, preprocessed, normal_samples=normal_samples_bgr)
                    result["fewshot_samples_used"] = min(5, len(normal_samples_bgr))
                    result["support_set_added"] = support_set_added
                    result["support_class"] = support_class if support_set_added > 0 else None
                    result["support_fabric"] = support_fabric

                    run = AdaptationRun(
                        shots=5,
                        fabric_type=result["fabric_type"],
                        support_samples=max(min(5, len(normal_samples_bgr)), support_set_added),
                        status="completed",
                        notes=notes or f"{result['pattern_type']} via {result['anomaly_route']}",
                    )
                    db.session.add(run)
                    db.session.commit()

                    flash("Anomaly classifier adaptation completed.", "success")
                except ValueError as exc:
                    flash(str(exc), "warning")
                except Exception as exc:
                    current_app.logger.exception("Adaptation run failed: %s", exc)
                    flash(f"Adaptation run failed: {exc}", "danger")
                finally:
                    try:
                        query_path.unlink(missing_ok=True)
                    except Exception:
                        current_app.logger.debug("Could not clean adapt query upload: %s", query_path)

                    for sample_path in sample_paths:
                        try:
                            sample_path.unlink(missing_ok=True)
                        except Exception:
                            current_app.logger.debug("Could not clean adapt support upload: %s", sample_path)
        else:
            if upload is None or upload.filename == "":
                flash("Please upload a query fabric image.", "warning")
            elif not allowed_file(upload.filename, current_app.config["ALLOWED_EXTENSIONS"]):
                flash("Unsupported query image type.", "danger")
            else:
                query_name = f"{uuid.uuid4().hex}_{secure_filename(upload.filename)}"
                query_path = Path(current_app.config["UPLOAD_FOLDER"]) / query_name
                upload.save(query_path)

                sample_paths: list[Path] = []
                normal_samples_bgr: list[np.ndarray] = []

                try:
                    original = read_bgr_image(query_path)
                    preprocessed = apply_clahe_only(original)

                    for sample in normal_sample_uploads:
                        if sample is None or sample.filename == "":
                            continue
                        if not allowed_file(sample.filename, current_app.config["ALLOWED_EXTENSIONS"]):
                            continue

                        sample_name = f"{uuid.uuid4().hex}_{secure_filename(sample.filename)}"
                        sample_path = Path(current_app.config["UPLOAD_FOLDER"]) / sample_name
                        sample.save(sample_path)
                        sample_paths.append(sample_path)
                        normal_samples_bgr.append(apply_clahe_only(read_bgr_image(sample_path)))

                    result = manager.inspect(original, preprocessed, normal_samples=normal_samples_bgr)
                    result["fewshot_samples_used"] = min(5, len(normal_samples_bgr))

                    run = AdaptationRun(
                        shots=5,
                        fabric_type=result["fabric_type"],
                        support_samples=min(5, len(normal_samples_bgr)),
                        status="completed",
                        notes=notes or f"{result['pattern_type']} via {result['anomaly_route']}",
                    )
                    db.session.add(run)
                    db.session.commit()

                    flash("WinCLIP anomaly detector run completed.", "success")
                except ValueError as exc:
                    flash(str(exc), "warning")
                except Exception as exc:
                    current_app.logger.exception("Adaptation run failed: %s", exc)
                    flash(f"Adaptation run failed: {exc}", "danger")
                finally:
                    try:
                        query_path.unlink(missing_ok=True)
                    except Exception:
                        current_app.logger.debug("Could not clean adapt query upload: %s", query_path)

                    for sample_path in sample_paths:
                        try:
                            sample_path.unlink(missing_ok=True)
                        except Exception:
                            current_app.logger.debug("Could not clean adapt support upload: %s", sample_path)

    runs = AdaptationRun.query.order_by(AdaptationRun.id.desc()).limit(20).all()
    return render_template("adapt.html", runs=runs, result=result)
