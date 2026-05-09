import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RESULT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions


def read_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def save_bgr_image(path: Path, image_bgr: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise ValueError(f"Unable to save image: {path}")


def apply_clahe_only(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    anomaly_map = anomaly_map.astype(np.float32)
    min_v = float(np.min(anomaly_map))
    max_v = float(np.max(anomaly_map))
    if max_v - min_v < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return (anomaly_map - min_v) / (max_v - min_v)


def build_heatmap(anomaly_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    norm = normalize_map(anomaly_map)
    uint8_map = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(uint8_map, colormap)


def build_anomaly_mask(
    anomaly_map: np.ndarray,
    threshold: float = 0.60,
    apply_morphology: bool = True,
    as_uint8: bool = True,
) -> np.ndarray:
    norm_map = normalize_map(anomaly_map)
    threshold = float(np.clip(threshold, 0.05, 0.98))
    defect_mask = (norm_map >= threshold).astype(np.uint8)

    if apply_morphology:
        # Reduce isolated speckles and fill tiny holes so the mask is stable.
        kernel = np.ones((3, 3), dtype=np.uint8)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)

    if as_uint8:
        return (defect_mask * 255).astype(np.uint8)
    return defect_mask


def blend_heatmap_on_original(
    original_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    alpha: float,
    anomaly_map: np.ndarray | None = None,
    defect_threshold: float = 0.60,
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))

    target_h, target_w = original_bgr.shape[:2]
    if heatmap_bgr.shape[:2] != (target_h, target_w):
        heatmap_bgr = cv2.resize(
            heatmap_bgr,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )

    if anomaly_map is None:
        return cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)

    if anomaly_map.shape[:2] != (target_h, target_w):
        anomaly_map = cv2.resize(
            anomaly_map,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        anomaly_map = normalize_map(anomaly_map)

    norm_map = normalize_map(anomaly_map)
    threshold = float(np.clip(defect_threshold, 0.05, 0.98))
    defect_mask = build_anomaly_mask(
        anomaly_map,
        threshold=threshold,
        apply_morphology=True,
        as_uint8=False,
    )

    strength = np.clip((norm_map - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    local_alpha = (alpha * np.power(strength, 0.65) * defect_mask).astype(np.float32)

    original_f = original_bgr.astype(np.float32)
    heatmap_f = heatmap_bgr.astype(np.float32)
    local_alpha = local_alpha[..., None]

    blended = original_f * (1.0 - local_alpha) + heatmap_f * local_alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_defect_bounding_boxes(
    image_bgr: np.ndarray,
    defects: list[dict[str, Any]],
) -> np.ndarray:
    canvas = image_bgr.copy()
    for defect in defects:
        bbox = defect.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x, y, width, height = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 255, 255), 2)

        severity = str(defect.get("severity", "defect")).upper()
        points = int(defect.get("points", 0))
        label = f"{severity} P{points}"
        label_y = max(16, y - 8)
        cv2.putText(
            canvas,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def compute_four_point_score(
    anomaly_map: np.ndarray,
    fabric_width_cm: float | None = None,
    fabric_height_cm: float | None = None,
    anomaly_mask: np.ndarray | None = None,
) -> tuple[int, str, list[dict[str, Any]]]:
    h, w = anomaly_map.shape[:2]
    if anomaly_mask is None:
        binary = build_anomaly_mask(
            anomaly_map,
            threshold=0.60,
            apply_morphology=True,
            as_uint8=False,
        )
    else:
        binary = (anomaly_mask > 0).astype(np.uint8)

    comp_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    total_area_cm2: float | None = None
    if (
        fabric_width_cm is not None
        and fabric_height_cm is not None
        and fabric_width_cm > 0
        and fabric_height_cm > 0
    ):
        total_area_cm2 = float(fabric_width_cm * fabric_height_cm)

    defects: list[dict[str, Any]] = []
    total_points = 0

    for component_id in range(1, comp_count):
        x, y, width, height, area = stats[component_id]
        area_ratio = float(area) / float(h * w)

        if area_ratio < 0.001:
            points = 1
            severity = "minor"
        elif area_ratio < 0.005:
            points = 2
            severity = "moderate"
        elif area_ratio < 0.015:
            points = 3
            severity = "major"
        else:
            points = 4
            severity = "critical"

        total_points += points
        local_mask = labels[y : y + height, x : x + width] == component_id
        local_scores = anomaly_map[y : y + height, x : x + width][local_mask]
        local_score = float(local_scores.mean()) if local_scores.size else 0.0
        area_cm2 = area_ratio * total_area_cm2 if total_area_cm2 is not None else None

        defects.append(
            {
                "bbox": [int(x), int(y), int(width), int(height)],
                "area": int(area),
                "area_ratio": area_ratio,
                "area_cm2": area_cm2,
                "score": local_score,
                "severity": severity,
                "points": points,
            }
        )

    if total_points <= 4:
        decision = "Accept"
    elif total_points <= 8:
        decision = "Hold"
    else:
        decision = "Reject"

    return total_points, decision, defects


def safe_result_id(candidate: str) -> str:
    candidate = (candidate or "").strip().lower()
    if not RESULT_ID_PATTERN.match(candidate):
        raise ValueError("Invalid result_id")
    return candidate


def suggest_defect_type(local_score: float, fabric_type: str) -> str:
    if local_score >= 0.85:
        return "hole_or_tear" if fabric_type == "woven" else "snag_or_run"
    if local_score >= 0.70:
        return "stain_or_contamination"
    if local_score >= 0.55:
        return "texture_irregularity"
    return "low_risk_variation"
