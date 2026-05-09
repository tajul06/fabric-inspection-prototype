import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, resnet50

try:
    import timm

    HAS_TIMM_RUNTIME = True
except Exception:  # pragma: no cover
    timm = None  # type: ignore
    HAS_TIMM_RUNTIME = False

from .utils import build_heatmap, normalize_map

try:
    import anomalib
    from anomalib.data.predict import PredictDataset  # type: ignore
    from anomalib.engine import Engine as AnomalibEngine  # type: ignore
    from anomalib.models import Patchcore as AnomalibPatchcore  # type: ignore
    from anomalib.models.image.winclip import WinClipModel  # type: ignore

    HAS_ANOMALIB_RUNTIME = True
    HAS_WINCLIP_RUNTIME = True
except Exception:  # pragma: no cover
    anomalib = None  # type: ignore
    PredictDataset = None  # type: ignore
    AnomalibEngine = None  # type: ignore
    AnomalibPatchcore = None  # type: ignore
    WinClipModel = None  # type: ignore
    HAS_ANOMALIB_RUNTIME = False
    HAS_WINCLIP_RUNTIME = False


LOGGER = logging.getLogger(__name__)


class KnittedProtoAnomalyClassifier:
    def __init__(
        self,
        ckpt_path: Path,
        support_root: Path,
        device: torch.device,
        class_order: list[str] | None = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.support_root = Path(support_root)
        self.device = device
        self.class_order = [name.strip().lower() for name in (class_order or []) if name.strip()]
        self.model = self._build_model()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.class_names: list[str] = []
        self.prototype_tensor: torch.Tensor | None = None
        self._support_signature: tuple[int, int] = (0, 0)
        self._load_support_prototypes()

    def _build_model(self) -> nn.Module:
        raw_state = torch.load(self.ckpt_path, map_location="cpu")
        if not isinstance(raw_state, dict):
            raise ValueError(f"Unexpected knitted proto checkpoint format: {self.ckpt_path}")

        if isinstance(raw_state.get("state_dict"), dict):
            state_dict = raw_state["state_dict"]
        elif isinstance(raw_state.get("model_state_dict"), dict):
            state_dict = raw_state["model_state_dict"]
        else:
            state_dict = raw_state

        model = resnet18(weights=None)
        model.fc = nn.Identity()

        cleaned_state = {}
        for key, value in state_dict.items():
            clean_key = key.replace("module.", "")
            if clean_key.startswith("backbone."):
                clean_key = clean_key[len("backbone.") :]
            cleaned_state[clean_key] = value

        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        if missing:
            LOGGER.warning("Knitted proto model missing keys (%d): %s", len(missing), missing[:5])
        if unexpected:
            LOGGER.warning("Knitted proto model unexpected keys (%d): %s", len(unexpected), unexpected[:5])

        model.to(self.device)
        model.eval()
        return model

    def _embed(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            embedding = self.model(self.transform(rgb).unsqueeze(0).to(self.device))
        return embedding.squeeze(0)

    def _load_support_prototypes(self) -> None:
        if not self.support_root.exists():
            LOGGER.warning("Knitted support root not found for proto classifier: %s", self.support_root)
            return

        valid_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        class_names: list[str] = []
        prototypes: list[torch.Tensor] = []

        class_dirs: list[Path] = []
        used_names: set[str] = set()

        if self.class_order:
            for class_name in self.class_order:
                candidate = self.support_root / class_name
                if candidate.is_dir():
                    class_dirs.append(candidate)
                    used_names.add(candidate.name)

        for class_dir in sorted(self.support_root.iterdir()):
            if not class_dir.is_dir():
                continue
            if class_dir.name in used_names:
                continue
            class_dirs.append(class_dir)

        for class_dir in class_dirs:
            image_paths = [
                path for path in sorted(class_dir.iterdir()) if path.is_file() and path.suffix.lower() in valid_suffixes
            ]
            if not image_paths:
                continue

            embeddings: list[torch.Tensor] = []
            for image_path in image_paths:
                try:
                    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image_bgr is None:
                        continue
                    embeddings.append(self._embed(image_bgr))
                except Exception as exc:  # pragma: no cover
                    LOGGER.debug("Could not embed knitted support image %s: %s", image_path, exc)

            if not embeddings:
                continue

            class_names.append(class_dir.name)
            prototypes.append(torch.stack(embeddings).mean(dim=0))

        if not prototypes:
            self.class_names = []
            self.prototype_tensor = None
            LOGGER.warning("No valid knitted support prototypes built from %s", self.support_root)
            return

        self.class_names = class_names
        self.prototype_tensor = torch.stack(prototypes).to(self.device)
        LOGGER.info(
            "Loaded knitted proto support with %d classes from %s",
            len(self.class_names),
            self.support_root,
        )

    def _compute_support_signature(self) -> tuple[int, int]:
        if not self.support_root.exists():
            return (0, 0)

        valid_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        files = [
            p
            for p in self.support_root.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ]
        if not files:
            return (0, 0)

        latest_mtime_ns = max(int(p.stat().st_mtime_ns) for p in files)
        return (len(files), latest_mtime_ns)

    def reload_support_if_needed(self) -> None:
        signature = self._compute_support_signature()
        if signature == self._support_signature:
            return

        self._support_signature = signature
        self._load_support_prototypes()

    def predict(self, image_bgr: np.ndarray) -> tuple[str, float] | None:
        self.reload_support_if_needed()

        if self.prototype_tensor is None or not self.class_names:
            return None

        query_embedding = self._embed(image_bgr)
        distances = torch.cdist(query_embedding.unsqueeze(0), self.prototype_tensor, p=2.0).squeeze(0)
        logits = -distances
        probabilities = torch.softmax(logits, dim=0)

        best_idx = int(torch.argmax(probabilities).item())
        return self.class_names[best_idx], float(probabilities[best_idx].item())


class PatchcoreCkptRunner:
    def __init__(self, ckpt_path: Path, temp_dir: Path, device: torch.device) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.temp_dir = Path(temp_dir)
        self.device = device

        self.image_size: tuple[int, int] = (256, 256)
        self.model: Any = None
        self.engine: Any = None

        self._build_runtime()

    def _build_runtime(self) -> None:
        _inject_precision_type_compatibility()

        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        hp = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}

        self.image_size = _extract_image_size(hp.get("pre_processor"))
        self.model = AnomalibPatchcore(
            backbone=hp.get("backbone", "wide_resnet50_2"),
            layers=hp.get("layers", ["layer2", "layer3"]),
            pre_trained=False,
            coreset_sampling_ratio=float(hp.get("coreset_sampling_ratio", 0.1)),
            num_neighbors=int(hp.get("num_neighbors", 9)),
            pre_processor=True,
            post_processor=True,
            evaluator=False,
            visualizer=False,
        )

        state_dict = checkpoint.get("state_dict", {}) if isinstance(checkpoint, dict) else {}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            LOGGER.warning("PatchCore checkpoint missing keys (%d): %s", len(missing), missing[:5])
        if unexpected:
            LOGGER.warning(
                "PatchCore checkpoint unexpected keys (%d): %s",
                len(unexpected),
                unexpected[:5],
            )

        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

        self.engine = AnomalibEngine(
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(self.temp_dir),
        )
        logging.getLogger("anomalib.engine.engine").setLevel(logging.ERROR)

    def predict(self, image_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.temp_dir / f"_predict_{uuid.uuid4().hex}.png"

        try:
            cv2.imwrite(str(temp_path), image_bgr)
            dataset = PredictDataset(path=temp_path, image_size=self.image_size)
            preds = self.engine.predict(
                model=self.model,
                dataset=dataset,
                return_predictions=True,
            )
            if not preds:
                raise RuntimeError("Anomalib Engine returned no predictions")

            item = preds[0]
            score = _extract_pred_score(item)
            anomaly_map = _extract_pred_anomaly_map(item, image_bgr.shape[:2])
            return score, anomaly_map
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                LOGGER.debug("Temporary prediction file cleanup failed: %s", temp_path)


class WinClipFewShotRunner:
    def __init__(self, device: torch.device, class_name: str | None = None) -> None:
        if WinClipModel is None:
            raise RuntimeError("WinCLIP runtime is unavailable.")

        default_name = (class_name or "fabric").strip()
        if not default_name:
            default_name = "fabric"

        self.default_class_name = default_name
        self.device = device
        self.model = WinClipModel(class_name=self.default_class_name, apply_transform=False)
        self.model.to(self.device)
        self.model.eval()
        self.transform = self.model.transform

    def _prepare_batch(self, images_bgr: list[np.ndarray]) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        for image_bgr in images_bgr:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb)
            if tensor.ndim != 3:
                raise ValueError("WinCLIP transform produced unexpected shape.")
            tensors.append(tensor)

        if not tensors:
            raise ValueError("WinCLIP requires at least one support image.")

        return torch.stack(tensors).to(self.device)

    def predict(
        self,
        image_bgr: np.ndarray,
        support_bgr: list[np.ndarray],
        k_shot: int,
        class_name: str | None = None,
    ) -> tuple[float, np.ndarray]:
        if k_shot <= 0:
            raise ValueError("WinCLIP k_shot must be greater than zero.")

        if len(support_bgr) < k_shot:
            raise ValueError(f"WinCLIP requires {k_shot} support images.")

        support_batch = self._prepare_batch(support_bgr[:k_shot])
        target_name = (class_name or self.default_class_name).strip()
        if not target_name:
            target_name = "fabric"
        self.model.setup(class_name=target_name, reference_images=support_batch)

        query_batch = self._prepare_batch([image_bgr])
        with torch.inference_mode():
            prediction = self.model(query_batch)

        pred_score = float(prediction.pred_score.squeeze().item())
        anomaly_map = prediction.anomaly_map.squeeze(0).detach().cpu().numpy()
        return float(np.clip(pred_score, 0.0, 1.0)), normalize_map(anomaly_map)


def _inject_precision_type_compatibility() -> None:
    if anomalib is None or hasattr(anomalib, "PrecisionType"):
        return

    class PrecisionType(Enum):
        FLOAT16 = "float16"
        FLOAT32 = "float32"
        BFLOAT16 = "bfloat16"
        MIXED = "mixed"

    anomalib.PrecisionType = PrecisionType  # type: ignore[attr-defined]


def _extract_image_size(pre_processor: Any) -> tuple[int, int]:
    if pre_processor is None:
        return (256, 256)

    image_size: Any = None
    if isinstance(pre_processor, dict):
        image_size = pre_processor.get("image_size")
    else:
        image_size = getattr(pre_processor, "image_size", None)

    if isinstance(image_size, int):
        return (int(image_size), int(image_size))

    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))

    return (256, 256)


def _extract_pred_score(item: Any) -> float:
    if hasattr(item, "pred_score"):
        pred_score = np.asarray(getattr(item, "pred_score"))
        if pred_score.size:
            return float(np.clip(float(pred_score.squeeze()), 0.0, 1.0))
    return 0.0


def _extract_pred_anomaly_map(item: Any, target_shape: tuple[int, int]) -> np.ndarray:
    anomaly_map = None

    if hasattr(item, "anomaly_map"):
        anomaly_map = getattr(item, "anomaly_map")
    elif isinstance(item, dict):
        anomaly_map = item.get("anomaly_map") or item.get("pred_mask")

    if anomaly_map is None:
        return np.zeros(target_shape, dtype=np.float32)

    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().cpu().numpy()
    else:
        anomaly_map = np.asarray(anomaly_map)

    anomaly_map = np.squeeze(anomaly_map).astype(np.float32)
    anomaly_map = cv2.resize(
        anomaly_map,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return normalize_map(anomaly_map)


def _canonical_fabric_label(raw_label: str) -> str:
    normalized = raw_label.strip().lower()
    if normalized in {"knit", "knitted", "knitten"}:
        return "knitted"
    if normalized in {"woven", "weave"}:
        return "woven"
    return normalized


def _canonical_pattern_label(raw_label: str) -> str:
    normalized = raw_label.strip().lower()
    if normalized in {"stripe", "striped"}:
        return "stripe"
    if normalized in {"plain", "solid"}:
        return "plain"
    if normalized in {"floral", "flower"}:
        return "floral"
    return normalized


def _fabric_to_subset_name(fabric_type: str) -> str:
    normalized = _canonical_fabric_label(fabric_type)
    if normalized == "knitted":
        return "knitten"
    return normalized


class MLModelManager:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.project_root = self.model_dir.parent
        self.workspace_roots = [self.project_root]
        parent_root = self.project_root.parent
        if parent_root != self.project_root:
            self.workspace_roots.append(parent_root)

        self.processed_dir = self.project_root / "processed"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fabric_labels = ["woven", "knitted"]
        self.fabric_classifier = self._load_fabric_classifier()
        self.patchcore_models = {
            "woven": self._load_patchcore_model("woven"),
            "knitted": self._load_patchcore_model("knitted"),
        }
        self.winclip_models: dict[str, WinClipFewShotRunner | None] = {
            "woven": None,
            "knitted": None,
        }
        self.pattern_labels: dict[str, list[str]] = {
            "woven": ["plain", "stripe", "floral"],
            "knitted": ["plain", "stripe"],
        }
        self.pattern_image_size: dict[str, int] = {
            "woven": 224,
            "knitted": 224,
        }
        self.pattern_classifiers = {
            "woven": self._load_pattern_classifier("woven"),
            "knitted": self._load_pattern_classifier("knitted"),
        }
        self.pattern_anomaly_models = self._load_pattern_anomaly_models()
        self.woven_fault_class_order = [
            "foreign_thread_or_lint",
            "hole",
            "stain_or_color_stripe",
            "tear",
            "thread_defect",
        ]
        self.knitted_proto_classifier = self._load_knitted_proto_classifier()
        self.woven_proto_classifier = self._load_woven_proto_classifier()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def requires_winclip_fewshot(self, fabric_type: str, pattern_type: str) -> bool:
        pattern_key = self._build_pattern_key(fabric_type, pattern_type)
        return self.pattern_anomaly_models.get(pattern_key) is None

    def _find_first_existing(self, candidates: list[Path]) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_fabric_classifier(self) -> nn.Module:
        classifier_candidates = [
            self.model_dir / "fabric_classifier_resnet50.pth",
            self.model_dir / "fabric_classifier_resnet50.pt",
        ]
        for root in self.workspace_roots:
            classifier_candidates.extend(
                [
                    root / "Fabric_classifier" / "restnet50" / "resnet50_run" / "best_model.pt",
                    root / "Fabric_classifier" / "restnet50" / "resnet50_run" / "last_model.pt",
                ]
            )
        weights_path = self._find_first_existing(classifier_candidates)

        class_to_idx: dict[str, int] = {
            "knitted": 0,
            "woven": 1,
        }
        state_dict: dict[str, Any] = {}

        if weights_path is not None:
            try:
                loaded = torch.load(weights_path, map_location=self.device)
                if isinstance(loaded, dict):
                    if isinstance(loaded.get("class_to_idx"), dict):
                        raw_mapping = loaded["class_to_idx"]
                        class_to_idx = {
                            _canonical_fabric_label(str(name)): int(index)
                            for name, index in raw_mapping.items()
                        }

                    if isinstance(loaded.get("model_state_dict"), dict):
                        state_dict = loaded["model_state_dict"]
                    elif isinstance(loaded.get("state_dict"), dict):
                        state_dict = loaded["state_dict"]
                    else:
                        state_dict = loaded
                else:
                    state_dict = loaded
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to load fabric classifier checkpoint %s: %s", weights_path, exc)
        else:
            LOGGER.warning("No fabric classifier checkpoint found in known locations")

        idx_to_class: dict[int, str] = {
            int(index): _canonical_fabric_label(str(name)) for name, index in class_to_idx.items()
        }
        if idx_to_class:
            self.fabric_labels = [idx_to_class[index] for index in sorted(idx_to_class)]

        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.fabric_labels))

        if state_dict:
            cleaned = {
                key.replace("module.", ""): value for key, value in state_dict.items()
            }
            try:
                model.load_state_dict(cleaned, strict=False)
                LOGGER.info("Loaded fabric classifier weights from %s", weights_path)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Classifier state dict load failed: %s", exc)

        model.to(self.device)
        model.eval()
        return model

    def _resolve_patchcore_ckpt(self, fabric_type: str) -> Path | None:
        external_candidates: list[Path] = []
        for root in self.workspace_roots:
            if fabric_type == "knitted":
                external_candidates.append(
                    root
                    / "Anomaly_detector_knitten"
                    / "anomalib_knitten"
                    / "anomalib_patchcore_knitten"
                    / "Patchcore"
                    / "knitten_patchcore"
                    / "v0"
                    / "weights"
                    / "lightning"
                    / "model.ckpt"
                )
            else:
                external_candidates.append(
                    root
                    / "Anomaly_detector_woven"
                    / "anomalib_patchcore_woven"
                    / "anomalib_patchcore_woven"
                    / "Patchcore"
                    / "woven_patchcore"
                    / "v0"
                    / "weights"
                    / "lightning"
                    / "model.ckpt"
                )

        if fabric_type == "knitted":
            candidates = [
                self.model_dir / "knitted_patchcore.ckpt",
                self.model_dir / "knitted_patchcore" / "model.ckpt",
            ]
        else:
            candidates = [
                self.model_dir / "woven_patchcore.ckpt",
                self.model_dir / "woven_patchcore" / "model.ckpt",
            ]
        candidates.extend(external_candidates)
        return self._find_first_existing(candidates)

    def _load_patchcore_model(self, fabric_type: str) -> Any:
        ckpt_path = self._resolve_patchcore_ckpt(fabric_type)
        if ckpt_path is None:
            LOGGER.warning("No PatchCore checkpoint found for %s", fabric_type)
            return None

        if not HAS_ANOMALIB_RUNTIME:
            LOGGER.warning(
                "Anomalib runtime import failed; cannot load PatchCore ckpt for %s",
                fabric_type,
            )
            return None

        try:
            runner = PatchcoreCkptRunner(
                ckpt_path=ckpt_path,
                temp_dir=self.processed_dir,
                device=self.device,
            )
            LOGGER.info("Loaded PatchCore ckpt for %s from %s", fabric_type, ckpt_path)
            return runner
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to load PatchCore ckpt %s: %s", ckpt_path, exc)
            return None

    def _resolve_pattern_classifier_ckpt(self, fabric_type: str) -> Path | None:
        subset_name = _fabric_to_subset_name(fabric_type)
        candidates: list[Path] = []
        for root in self.workspace_roots:
            candidates.append(
                root
                / "Fabric_pattern_classifier"
                / f"best_efficientnet_fabric_print_{subset_name}.pth"
            )
        return self._find_first_existing(candidates)

    def _load_pattern_classifier(self, fabric_type: str) -> nn.Module | None:
        ckpt_path = self._resolve_pattern_classifier_ckpt(fabric_type)
        if ckpt_path is None:
            LOGGER.warning("No pattern classifier checkpoint found for %s", fabric_type)
            return None

        if not HAS_TIMM_RUNTIME:
            LOGGER.warning("timm runtime import failed; cannot load pattern classifier for %s", fabric_type)
            return None

        try:
            loaded = torch.load(ckpt_path, map_location=self.device)
            if not isinstance(loaded, dict):
                LOGGER.warning("Unexpected pattern classifier format for %s", ckpt_path)
                return None

            raw_labels = loaded.get("class_names")
            if not isinstance(raw_labels, list) or not raw_labels:
                LOGGER.warning("Pattern classifier %s missing class_names", ckpt_path)
                return None

            pattern_labels = [_canonical_pattern_label(str(label)) for label in raw_labels]
            model_name = str(loaded.get("model_name", "efficientnet_b0"))
            image_size = int(loaded.get("image_size", 224))
            state_dict = loaded.get("model_state_dict")
            if not isinstance(state_dict, dict):
                LOGGER.warning("Pattern classifier %s missing model_state_dict", ckpt_path)
                return None

            model = timm.create_model(model_name, pretrained=False, num_classes=len(pattern_labels))
            cleaned_state_dict = {
                key.replace("module.", ""): value for key, value in state_dict.items()
            }
            model.load_state_dict(cleaned_state_dict, strict=False)
            model.to(self.device)
            model.eval()

            normalized_fabric = _canonical_fabric_label(fabric_type)
            self.pattern_labels[normalized_fabric] = pattern_labels
            self.pattern_image_size[normalized_fabric] = max(32, image_size)

            LOGGER.info("Loaded pattern classifier for %s from %s", normalized_fabric, ckpt_path)
            return model
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to load pattern classifier for %s: %s", fabric_type, exc)
            return None

    def _pattern_detector_roots(self) -> list[Path]:
        detector_roots: list[Path] = []
        seen: set[Path] = set()
        for root in self.workspace_roots:
            # prefer user-provided 'indiv anomaly detector' folders before the older 'Pattern_based_Anomaly_Detect'
            for name in ("indiv anomaly detector", "Pattern_based_Anomaly_Detect"):
                candidate = root / name
                if candidate.exists() and candidate not in seen:
                    detector_roots.append(candidate)
                    seen.add(candidate)
        return detector_roots

    def _pattern_detector_keys(self) -> list[str]:
        detected_keys: set[str] = set()
        for detector_root in self._pattern_detector_roots():
            for child in detector_root.iterdir():
                if child.is_dir():
                    detected_keys.add(child.name.strip().lower())
        return sorted(detected_keys)

    def _resolve_pattern_anomaly_ckpt(self, pattern_key: str) -> Path | None:
        candidates: list[Path] = []
        for detector_root in self._pattern_detector_roots():
            # Only use checkpoints from the 'indiv anomaly detector' folder for woven patterns.
            # This prevents a user-provided indiv detector for knitted patterns from being used.
            try:
                root_name = detector_root.name.lower()
            except Exception:
                root_name = ""

            if root_name == "indiv anomaly detector" and not pattern_key.startswith("woven_"):
                continue

            candidates.extend(
                [
                    detector_root / pattern_key / f"{pattern_key}_best.ckpt",
                    detector_root / pattern_key / "last.ckpt",
                ]
            )
        found = self._find_first_existing(candidates)
        if found is not None:
            return found

        # Fallback: some detector folders use a slightly different ckpt naming
        # (e.g., folder 'woven_gray plain' contains 'woven_plain_best.ckpt').
        # Search the pattern folder for any '*best*.ckpt' or any '.ckpt' file.
        for detector_root in self._pattern_detector_roots():
            pattern_dir = detector_root / pattern_key
            if not pattern_dir.exists() or not pattern_dir.is_dir():
                continue
            try:
                # prefer files containing 'best'
                candidates_files = sorted(pattern_dir.glob('*best*.ckpt'))
                if not candidates_files:
                    candidates_files = sorted(pattern_dir.glob('*.ckpt'))
                if candidates_files:
                    return candidates_files[0]
            except Exception:
                continue

        return None

    def _load_pattern_anomaly_models(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        if not HAS_ANOMALIB_RUNTIME:
            return models

        for key in self._pattern_detector_keys():
            ckpt_path = self._resolve_pattern_anomaly_ckpt(key)
            if ckpt_path is None:
                continue
            try:
                models[key] = PatchcoreCkptRunner(
                    ckpt_path=ckpt_path,
                    temp_dir=self.processed_dir,
                    device=self.device,
                )
                LOGGER.info("Loaded pattern-based PatchCore ckpt for %s from %s", key, ckpt_path)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to load pattern-based ckpt %s: %s", ckpt_path, exc)
        return models

    def _normalize_winclip_shots(self, raw_value: int | None) -> int:
        try:
            value = int(raw_value) if raw_value is not None else 0
        except (TypeError, ValueError):
            value = 0

        if value in {3, 5, 10}:
            return value
        return 5

    def _normalize_fabric_override(self, raw_value: str | None) -> str | None:
        if raw_value is None:
            return None

        value = raw_value.strip().lower()
        if value in {"", "auto"}:
            return None

        normalized = _canonical_fabric_label(value)
        if normalized in {"woven", "knitted"}:
            return normalized
        return None

    def _load_winclip_model(self, fabric_type: str) -> WinClipFewShotRunner | None:
        if not HAS_WINCLIP_RUNTIME:
            LOGGER.warning("WinCLIP runtime import failed; cannot load WinCLIP for %s", fabric_type)
            return None

        for cached_runner in self.winclip_models.values():
            if cached_runner is not None:
                return cached_runner

        try:
            runner = WinClipFewShotRunner(device=self.device, class_name="fabric")
            LOGGER.info("Loaded WinCLIP model (shared) for %s", fabric_type)
            return runner
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to load WinCLIP model for %s: %s", fabric_type, exc)
            return None

    def _get_winclip_model(self, fabric_type: str) -> WinClipFewShotRunner | None:
        normalized = _canonical_fabric_label(fabric_type)
        cached = self.winclip_models.get(normalized)
        if cached is not None:
            return cached

        runner = self._load_winclip_model(normalized)
        if runner is not None:
            self.winclip_models[normalized] = runner
        return runner

    def detect_winclip(
        self,
        image_bgr: np.ndarray,
        fabric_type: str,
        support_images: list[np.ndarray],
        k_shot: int,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        runner = self._get_winclip_model(fabric_type)
        if runner is None:
            raise ValueError(f"WinCLIP model not available for {fabric_type}.")

        class_name = "woven fabric" if fabric_type == "woven" else "knitted fabric"
        score, anomaly_map = runner.predict(image_bgr, support_images, k_shot, class_name=class_name)
        target_h, target_w = image_bgr.shape[:2]
        if anomaly_map.shape[:2] != (target_h, target_w):
            anomaly_map = cv2.resize(
                anomaly_map,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
            anomaly_map = normalize_map(anomaly_map)

        heatmap = build_heatmap(anomaly_map)
        return score, anomaly_map, heatmap

    def _resolve_knitted_proto_ckpt(self) -> Path | None:
        candidates: list[Path] = []
        for root in self.workspace_roots:
            candidates.extend(
                [
                    root / "knitten anomaly classifier" / "best_proto_fabric_model.pth",
                    root / "knitted anomaly classifier" / "best_proto_fabric_model.pth",
                ]
            )
        return self._find_first_existing(candidates)

    def _resolve_knitted_proto_support_root(self) -> Path | None:
        candidates: list[Path] = []
        for root in self.workspace_roots:
            candidates.extend(
                [
                    root / "knitten anomaly classifier" / "support",
                    root / "knitted anomaly classifier" / "support",
                    root / "Novel Fabric" / "knitted",
                    root / "Novel Fabric" / "knitten",
                ]
            )
        return self._find_first_existing(candidates)

    def _resolve_woven_proto_ckpt(self) -> Path | None:
        candidates: list[Path] = []
        for root in self.workspace_roots:
            candidates.extend(
                [
                    root / "woven  anomaly classifier" / "best_proto_fabric_model.pth",
                    root / "woven anomaly classifier" / "best_proto_fabric_model.pth",
                    root / "woven_anomaly_classifier" / "best_proto_fabric_model.pth",
                ]
            )
        return self._find_first_existing(candidates)

    def _resolve_woven_proto_support_root(self) -> Path | None:
        candidates: list[Path] = []
        for root in self.workspace_roots:
            candidates.extend(
                [
                    root / "woven  anomaly classifier" / "support",
                    root / "woven anomaly classifier" / "support",
                    root / "woven_anomaly_classifier" / "support",
                    root / "Novel Fabric" / "woven",
                ]
            )
        return self._find_first_existing(candidates)

    def _load_knitted_proto_classifier(self) -> KnittedProtoAnomalyClassifier | None:
        ckpt_path = self._resolve_knitted_proto_ckpt()
        if ckpt_path is None:
            return None

        support_root = self._resolve_knitted_proto_support_root()
        if support_root is None:
            return None

        try:
            classifier = KnittedProtoAnomalyClassifier(
                ckpt_path=ckpt_path,
                support_root=support_root,
                device=self.device,
            )
            LOGGER.info("Loaded knitted proto anomaly classifier from %s", ckpt_path)
            return classifier
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed loading knitted proto anomaly classifier: %s", exc)
            return None

    def _load_woven_proto_classifier(self) -> KnittedProtoAnomalyClassifier | None:
        ckpt_path = self._resolve_woven_proto_ckpt()
        if ckpt_path is None:
            return None

        support_root = self._resolve_woven_proto_support_root()
        if support_root is None:
            return None

        try:
            classifier = KnittedProtoAnomalyClassifier(
                ckpt_path=ckpt_path,
                support_root=support_root,
                device=self.device,
                class_order=self.woven_fault_class_order,
            )
            LOGGER.info("Loaded woven proto anomaly classifier from %s", ckpt_path)
            return classifier
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed loading woven proto anomaly classifier: %s", exc)
            return None

    def classify_fabric(self, image_bgr: np.ndarray) -> tuple[str, float]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            logits = self.fabric_classifier(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        if float(np.max(probs)) < 0.51:
            return self._heuristic_fabric_guess(image_bgr)

        idx = int(np.argmax(probs))
        return self.fabric_labels[idx], float(probs[idx])

    def classify_pattern(self, image_bgr: np.ndarray, fabric_type: str) -> tuple[str, float]:
        normalized_fabric = _canonical_fabric_label(fabric_type)
        classifier = self.pattern_classifiers.get(normalized_fabric)
        labels = self.pattern_labels.get(normalized_fabric, ["plain"])

        if classifier is None:
            return (labels[0], 0.5)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_size = int(self.pattern_image_size.get(normalized_fabric, 224))
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        with torch.inference_mode():
            tensor = transform(rgb).unsqueeze(0).to(self.device)
            logits = classifier(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx = int(np.argmax(probs))
        idx = int(np.clip(idx, 0, max(0, len(labels) - 1)))
        return labels[idx], float(probs[idx])

    def _build_pattern_key(self, fabric_type: str, pattern_type: str) -> str:
        subset_fabric = _fabric_to_subset_name(fabric_type)
        normalized_pattern = _canonical_pattern_label(pattern_type)
        return f"{subset_fabric}_{normalized_pattern}"

    def detect_pattern_anomaly(
        self,
        image_bgr: np.ndarray,
        fabric_type: str,
        pattern_type: str,
    ) -> tuple[float, np.ndarray, np.ndarray, str]:
        pattern_key = self._build_pattern_key(fabric_type, pattern_type)
        inferencer = self.pattern_anomaly_models.get(pattern_key)

        if inferencer is not None:
            try:
                score, anomaly_map = inferencer.predict(image_bgr)
                heatmap = build_heatmap(anomaly_map)
                return score, anomaly_map, heatmap, pattern_key
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Pattern-routed PatchCore inference failed for %s: %s", pattern_key, exc)

        fallback_score, fallback_map, fallback_heatmap = self.detect_anomaly(image_bgr, fabric_type)
        return fallback_score, fallback_map, fallback_heatmap, f"{pattern_key}:fallback"

    def _heuristic_fabric_guess(self, image_bgr: np.ndarray) -> tuple[str, float]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        orientation_ratio = float(np.mean(np.abs(grad_x)) / (np.mean(np.abs(grad_y)) + 1e-6))

        if orientation_ratio > 1.1:
            return "woven", 0.58
        return "knitted", 0.58

    def detect_anomaly(self, image_bgr: np.ndarray, fabric_type: str) -> tuple[float, np.ndarray, np.ndarray]:
        inferencer = self.patchcore_models.get(fabric_type)
        if inferencer is not None:
            try:
                score, anomaly_map = inferencer.predict(image_bgr)
                heatmap = build_heatmap(anomaly_map)
                return score, anomaly_map, heatmap
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("PatchCore inference failed, using fallback: %s", exc)

        fallback_map = self._fallback_anomaly_map(image_bgr)
        score = float(np.quantile(fallback_map, 0.99))
        heatmap = build_heatmap(fallback_map)
        return score, fallback_map, heatmap

    def _fallback_anomaly_map(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), 2.5)
        high_freq = cv2.absdiff(gray, blur)

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(sobel_x, sobel_y)

        combined = 0.65 * high_freq + 0.35 * gradient
        return normalize_map(combined)

    def _extract_proto_query_crop(
        self,
        image_bgr: np.ndarray,
        anomaly_map: np.ndarray,
        threshold: float = 0.60,
    ) -> tuple[np.ndarray, bool]:
        norm_map = normalize_map(anomaly_map)
        threshold = float(np.clip(threshold, 0.05, 0.98))
        mask = (norm_map >= threshold).astype(np.uint8)

        if int(mask.sum()) == 0:
            return image_bgr, False

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_u8 = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image_bgr, False

        best = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best)

        if w <= 2 or h <= 2:
            return image_bgr, False

        pad_x = max(2, int(round(w * 0.10)))
        pad_y = max(2, int(round(h * 0.10)))

        height, width = image_bgr.shape[:2]
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(width, x + w + pad_x)
        y1 = min(height, y + h + pad_y)

        if x1 <= x0 or y1 <= y0:
            return image_bgr, False

        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return image_bgr, False

        return crop, True

    def classify_anomaly(self, anomaly_map: np.ndarray, fabric_type: str) -> tuple[str, float]:
        peak = float(np.max(anomaly_map))
        mean_val = float(np.mean(anomaly_map))

        if peak >= 0.85:
            return ("hole_or_tear", min(0.95, 0.65 + peak * 0.35))
        if mean_val >= 0.55:
            label = "thread_inconsistency" if fabric_type == "woven" else "knit_structure_shift"
            return (label, min(0.90, 0.55 + mean_val * 0.40))
        if peak >= 0.65:
            return ("surface_stain", min(0.88, 0.45 + peak * 0.45))

        return ("background_variation", 0.52)

    def inspect(
        self,
        original_bgr: np.ndarray,
        preprocessed_bgr: np.ndarray,
        normal_samples: list[np.ndarray] | None = None,
        pattern_mode: str = "auto",
        fabric_override: str | None = None,
        winclip_support: list[np.ndarray] | None = None,
        winclip_shots: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = (pattern_mode or "auto").strip().lower()
        if normalized_mode not in {"auto", "unknown"}:
            normalized_mode = "auto"

        normalized_override = self._normalize_fabric_override(fabric_override)
        if normalized_override:
            fabric_type = normalized_override
            fabric_conf = 1.0
        else:
            fabric_type, fabric_conf = self.classify_fabric(preprocessed_bgr)

        support_images = winclip_support or normal_samples or []
        support_shots = self._normalize_winclip_shots(winclip_shots)

        if normalized_mode == "unknown":
            if len(support_images) < support_shots:
                raise ValueError(f"WinCLIP requires {support_shots} support images.")
            pattern_type = "unknown"
            pattern_conf = 0.0
            anomaly_score, anomaly_map, heatmap = self.detect_winclip(
                preprocessed_bgr,
                fabric_type,
                support_images,
                support_shots,
            )
            route_key = f"winclip_{fabric_type}_k{support_shots}"
        else:
            pattern_type, pattern_conf = self.classify_pattern(preprocessed_bgr, fabric_type)
            anomaly_score, anomaly_map, heatmap, route_key = self.detect_pattern_anomaly(
                preprocessed_bgr,
                fabric_type,
                pattern_type,
            )
        anomaly_class, anomaly_conf = self.classify_anomaly(anomaly_map, fabric_type)

        if fabric_type == "knitted" and self.knitted_proto_classifier is not None:
            proto_query_bgr, used_crop = self._extract_proto_query_crop(preprocessed_bgr, anomaly_map)
            proto_prediction = self.knitted_proto_classifier.predict(proto_query_bgr)
            if proto_prediction is not None:
                anomaly_class, anomaly_conf = proto_prediction
                route_key = (
                    f"{route_key}:knitted_proto_crop"
                    if used_crop
                    else f"{route_key}:knitted_proto_full"
                )

        if fabric_type == "woven" and self.woven_proto_classifier is not None:
            proto_query_bgr, used_crop = self._extract_proto_query_crop(preprocessed_bgr, anomaly_map)
            proto_prediction = self.woven_proto_classifier.predict(proto_query_bgr)
            if proto_prediction is not None:
                anomaly_class, anomaly_conf = proto_prediction
                route_key = (
                    f"{route_key}:woven_proto_crop"
                    if used_crop
                    else f"{route_key}:woven_proto_full"
                )

        return {
            "fabric_type": fabric_type,
            "fabric_confidence": fabric_conf,
            "pattern_type": pattern_type,
            "pattern_confidence": float(np.clip(pattern_conf, 0.0, 1.0)),
            "anomaly_route": route_key,
            "anomaly_score": float(np.clip(anomaly_score, 0.0, 1.0)),
            "anomaly_map": anomaly_map.astype(np.float32),
            "heatmap": heatmap,
            "anomaly_class": anomaly_class,
            "anomaly_confidence": float(np.clip(anomaly_conf, 0.0, 1.0)),
        }
