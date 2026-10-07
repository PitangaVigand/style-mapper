from __future__ import annotations

import importlib.util
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import pytest
from PIL import ExifTags, Image

from common import geojson_to_html_map
from common.protocol import ClassificationResult, Coordinates


LOGGER = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = CURRENT_DIR / "samples"
OUTPUT_DIR = CURRENT_DIR / "output"
AV_JULIO_DE_MESQUITA_DIR = (
    CURRENT_DIR
    / "gen_street_view_images_by_street"
    / "streetview_images"
    / "avenida_julio_de_mesquita"
)
GEOJSON_OUTPUT_PATH = OUTPUT_DIR / "neo_detections.geojson"
MAP_OUTPUT_PATH = OUTPUT_DIR / "neo_map.html"
CLASSIFIER_MODULE_PATH = CURRENT_DIR / "image_classifier" / "test_classify_image.py"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.93
TARGET_LABEL = "neo"
GPS_INFO_TAG = next(
    tag_id for tag_id, tag_name in ExifTags.TAGS.items() if tag_name == "GPSInfo"
)
FILENAME_GPS_PATTERN = re.compile(
    r"lat_(?P<latitude>-?\d+(?:\.\d+)?)_lon_(?P<longitude>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GeolocatedClassification:
    """Classification result enriched with geographic coordinates."""

    image_name: str
    image_path: str
    absolute_image_path: str
    predicted_label: str
    confidence_score: float
    latitude: float
    longitude: float


def run_geolocation_pipeline(
    samples_dir: str | Path = SAMPLES_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    target_label: str = TARGET_LABEL,
) -> Path:
    """Classify sample images and export a GeoJSON dataset.

    Parameters
    ----------
    samples_dir
        Directory containing the input sample images.
    output_dir
        Directory where the GeoJSON output will be saved.
    target_label
        Label retained in the GeoJSON output.

    Returns
    -------
    Path
        Saved GeoJSON path.
    """
    samples_root = Path(samples_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = collect_sample_images(samples_root)
    if not image_paths:
        raise ValueError(f"No supported images were found in '{samples_root}'.")

    classify_images = _load_classify_images()
    detections: list[dict[str, Any]] = []

    for image_path in _progress(image_paths):
        LOGGER.info("Processing image %s", image_path.name)

        try:
            classification_batch = classify_images(image_path, debug=True)
            if not classification_batch:
                LOGGER.warning("No classification result returned for %s", image_path.name)
                continue

            result = classification_batch[0]
            gps = extract_gps_from_exif(image_path)
            if gps is None:
                LOGGER.warning("Skipping %s because no GPS data was found.", image_path.name)
                continue

            enriched = _build_geolocated_result(result, gps)
            if enriched.predicted_label == target_label:
                detections.append(build_feature(enriched))
        except Exception as exc:  # pragma: no cover - defensive pipeline logging
            LOGGER.exception("Failed to process %s: %s", image_path.name, exc)

    geojson_path = output_root / GEOJSON_OUTPUT_PATH.name
    save_geojson(detections, geojson_path)

    LOGGER.info(
        "Saved %d detections to %s",
        len(detections),
        geojson_path,
    )
    return geojson_path


def collect_sample_images(samples_dir: str | Path) -> list[Path]:
    """Collect supported sample images in deterministic order."""
    samples_root = Path(samples_dir)
    return sorted(
        path
        for path in samples_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def extract_gps_from_exif(image_path: str | Path) -> Coordinates | None:
    """Extract GPS coordinates from EXIF metadata or the image filename."""
    path = Path(image_path)
    exif_coordinates = _extract_gps_from_exif_payload(path)
    if exif_coordinates is not None:
        return exif_coordinates

    return _extract_gps_from_filename(path)


def save_geojson(features: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save a GeoJSON FeatureCollection to disk."""
    geojson_path = Path(output_path)
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"type": "FeatureCollection", "features": features}
    geojson_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return geojson_path


def build_feature(result: GeolocatedClassification) -> dict[str, Any]:
    """Convert a geolocated classification result into a GeoJSON feature."""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [result.longitude, result.latitude],
        },
        "properties": {
            "image_name": result.image_name,
            "confidence": result.confidence_score,
            "label": result.predicted_label,
            "image_path": result.image_path,
            "absolute_image_path": result.absolute_image_path,
        },
    }


def _build_geolocated_result(
    result: ClassificationResult,
    coordinates: Coordinates,
) -> GeolocatedClassification:
    image_path = Path(result.image_path).resolve()
    return GeolocatedClassification(
        image_name=image_path.name,
        image_path=_to_project_relative_path(image_path),
        absolute_image_path=str(image_path),
        predicted_label=result.predicted_label,
        confidence_score=result.confidence_score,
        latitude=coordinates.latitude,
        longitude=coordinates.longitude,
    )


def _to_project_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(CURRENT_DIR))
    except ValueError:
        return str(path)


def _extract_gps_from_exif_payload(image_path: Path) -> Coordinates | None:
    try:
        with Image.open(image_path) as image:
            exif_payload = image.getexif()
    except OSError:
        return None

    gps_payload = exif_payload.get(GPS_INFO_TAG)
    if not gps_payload:
        return None

    gps_data = {
        ExifTags.GPSTAGS.get(tag_id, tag_id): value for tag_id, value in gps_payload.items()
    }

    latitude = _gps_to_decimal(
        gps_data.get("GPSLatitude"),
        gps_data.get("GPSLatitudeRef"),
    )
    longitude = _gps_to_decimal(
        gps_data.get("GPSLongitude"),
        gps_data.get("GPSLongitudeRef"),
    )

    if latitude is None or longitude is None:
        return None

    return Coordinates(latitude=latitude, longitude=longitude)


def _extract_gps_from_filename(image_path: Path) -> Coordinates | None:
    match = FILENAME_GPS_PATTERN.search(image_path.stem)
    if match is None:
        return None

    return Coordinates(
        latitude=float(match.group("latitude")),
        longitude=float(match.group("longitude")),
    )


def _gps_to_decimal(values: Any, reference: Any) -> float | None:
    if not values or reference is None:
        return None

    degrees, minutes, seconds = values
    decimal = (
        _rational_to_float(degrees)
        + _rational_to_float(minutes) / 60.0
        + _rational_to_float(seconds) / 3600.0
    )

    reference_text = str(reference)
    if reference_text.upper() in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _rational_to_float(value: Any) -> float:
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator:
        return float(numerator) / float(denominator)

    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])

    return float(value)


def _progress(paths: Iterable[Path]) -> Iterable[Path]:
    try:
        from tqdm import tqdm
    except ImportError:
        return paths

    return tqdm(paths, desc="Classifying samples")


@lru_cache(maxsize=1)
def _load_classifier_module() -> ModuleType:
    if not CLASSIFIER_MODULE_PATH.exists():
        raise FileNotFoundError(f"Classifier module was not found: {CLASSIFIER_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "style_mapper_classifier",
        CLASSIFIER_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load classifier module from {CLASSIFIER_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_classify_images():
    module = _load_classifier_module()
    classify_images = getattr(module, "classify_images", None)
    if classify_images is None:
        raise AttributeError("The classifier module does not expose classify_images().")
    return classify_images


def test_extract_gps_from_filename_parses_generated_streetview_names() -> None:
    path = Path("streetview_0007_lat_-22.9049200_lon_-47.0611300.jpg")
    coordinates = extract_gps_from_exif(path)

    assert coordinates is not None
    assert coordinates.latitude == pytest.approx(-22.90492)
    assert coordinates.longitude == pytest.approx(-47.06113)


def test_save_geojson_writes_feature_collection(tmp_path: Path) -> None:
    output_path = tmp_path / "detections.geojson"
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-47.06113, -22.90492]},
        "properties": {"image_name": "sample.jpg", "confidence": 0.91, "label": "neo"},
    }

    saved_path = save_geojson([feature], output_path)
    payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path == output_path
    assert payload["type"] == "FeatureCollection"
    assert payload["features"] == [feature]


def test_geolocate_samples_pipeline() -> None:
    if not collect_sample_images(SAMPLES_DIR):
        pytest.skip("No sample images were found in Projects/style mapper/samples.")

    geojson_path = run_geolocation_pipeline()
    map_path = geojson_to_html_map(
        geojson_path,
        MAP_OUTPUT_PATH,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )

    assert geojson_path.exists()
    assert map_path.exists()


def test_geolocate_streetview_images_av_julio_de_mesquita() -> None:
    if not collect_sample_images(AV_JULIO_DE_MESQUITA_DIR):
        pytest.skip(
            "No Street View images were found in "
            "Projects/style mapper/gen_street_view_images_by_street/streetview_images/avenida_julio_de_mesquita."
        )

    geojson_path = run_geolocation_pipeline(
        samples_dir=AV_JULIO_DE_MESQUITA_DIR,
        output_dir=OUTPUT_DIR,
    )
    detections = json.loads(geojson_path.read_text(encoding="utf-8"))["features"]
    map_path = geojson_to_html_map(
        geojson_path,
        MAP_OUTPUT_PATH,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    assert geojson_path.exists()
    assert map_path.exists()
    assert detections


def test_geojson_to_map_av_julio_de_mesquita() -> None:
    output_path = geojson_to_html_map(
        GEOJSON_OUTPUT_PATH,
        MAP_OUTPUT_PATH,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )

    assert output_path == MAP_OUTPUT_PATH
    assert output_path.exists()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_geolocation_pipeline()
