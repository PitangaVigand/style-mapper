from __future__ import annotations

import math
import os
import unicodedata
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import requests
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union

from common.protocol import Coordinates, Street

DEFAULT_STREET_NAME = "Avenida Julio de Mesquita"
DEFAULT_CITY_NAME = "Cambui, Campinas, Sao Paulo, Brazil"
DEFAULT_DISTANCE_BETWEEN_IMAGES = 10.0
STREETVIEW_SIDE_OFFSETS = {
    "left": -90.0,
    "right": 90.0,
}

DEFAULT_SEARCH_RADIUS_METERS = 500
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 640
GOOGLE_METADATA_RADIUS_METERS = 50
REQUEST_TIMEOUT_SECONDS = 30
GOOGLE_STREETVIEW_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
GOOGLE_STREETVIEW_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

CURRENT_DIR = Path(__file__).resolve().parent
STREETVIEW_OUTPUT_DIR = CURRENT_DIR / "streetview_images"


def test_gen_streetview_images():
    street = _get_street_from_open_street_map(DEFAULT_STREET_NAME, DEFAULT_CITY_NAME)
    coords = _gen_street_coordinatinates(street, DEFAULT_DISTANCE_BETWEEN_IMAGES)

    assert coords

    first_image_paths = gen_streetview_images(
        coords[:2],
        street_name=street.name,
        max_images=1,
    )
    assert len(first_image_paths) == 2
    assert all(path.exists() and path.stat().st_size > 0 for path in first_image_paths)
    assert all(_filename_contains_coordinates(path) for path in first_image_paths)
    assert {path.stem.split("_")[2] for path in first_image_paths} == {"left", "right"}

    all_image_paths = gen_streetview_images(coords, street_name=street.name)
    assert len(all_image_paths) == len(coords) * len(STREETVIEW_SIDE_OFFSETS)
    assert all(path.exists() and path.stat().st_size > 0 for path in all_image_paths)
    assert all(_filename_contains_coordinates(path) for path in all_image_paths)


def gen_streetview_images(
    coords: list[Coordinates],
    street_name: str = DEFAULT_STREET_NAME,
    max_images: int | None = None,
) -> list[Path]:
    # Para cada coordenada, baixa uma imagem do Street View e salva em output/streetview_images.
    if not coords:
        raise ValueError("At least one coordinate is required to generate Street View images.")

    api_key = _load_google_api_key()
    output_dir = STREETVIEW_OUTPUT_DIR / _slugify(street_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    coords_to_process = coords[:max_images] if max_images is not None else coords
    saved_paths: list[Path] = []

    for index, coord in enumerate(coords_to_process):
        heading = _compute_heading(coords_to_process, index)
        metadata = _get_streetview_metadata(coord, api_key)
        capture_coord = _resolve_capture_coordinates(coord, metadata)

        for side_name, side_heading in _compute_side_headings(heading).items():
            image_path = output_dir / _build_image_filename(index, side_name, capture_coord)
            params = {
                "size": f"{DEFAULT_IMAGE_WIDTH}x{DEFAULT_IMAGE_HEIGHT}",
                "location": f"{coord.latitude:.7f},{coord.longitude:.7f}",
                "heading": f"{side_heading:.2f}",
                "pitch": "0",
                "fov": "90",
                "radius": str(GOOGLE_METADATA_RADIUS_METERS),
                "return_error_code": "true",
                "source": "outdoor",
                "key": api_key,
            }

            response = requests.get(
                GOOGLE_STREETVIEW_IMAGE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            image_path.write_bytes(response.content)
            _attach_gps_metadata(image_path, capture_coord)
            saved_paths.append(image_path)

    return saved_paths


def _get_street_from_open_street_map(name: str, city: str) -> Street:
    # Dado o nome da rua e a cidade, encontra os segmentos no OSM e monta um eixo unico.
    city_coords = ox.geocode(f"{name}, {city}")
    graph = ox.graph_from_point(
        city_coords,
        dist=DEFAULT_SEARCH_RADIUS_METERS,
        network_type="drive",
        simplify=True,
    )
    _, edges = ox.graph_to_gdfs(graph)

    normalized_target = _normalize_text(name)
    street_edges = edges.loc[
        edges["name"].fillna("").astype(str).map(
            lambda value: normalized_target in _normalize_text(value)
        )
    ]

    if street_edges.empty:
        raise ValueError(f"Street '{name}' was not found around '{city}'.")

    axis_of_the_road = _merge_street_geometry(street_edges.geometry.tolist())
    return Street(name=name, axis=axis_of_the_road)


def _gen_street_coordinatinates(
    street: Street,
    distance_between_images: float,
) -> list[Coordinates]:
    # Divide a rua de distance_between_images em distance_between_images metros.
    if distance_between_images <= 0:
        raise ValueError("distance_between_images must be greater than zero.")

    metric_crs = gpd.GeoSeries([street.axis], crs="EPSG:4326").estimate_utm_crs()
    metric_series = gpd.GeoSeries([street.axis], crs="EPSG:4326").to_crs(metric_crs)
    metric_axis = metric_series.iloc[0]

    distances = list(np.arange(0, metric_axis.length, distance_between_images))
    if not distances or distances[-1] != metric_axis.length:
        distances.append(metric_axis.length)

    projected_points = [metric_axis.interpolate(distance, normalized=False) for distance in distances]
    geographic_points = gpd.GeoSeries(projected_points, crs=metric_crs).to_crs("EPSG:4326")

    coords: list[Coordinates] = []
    for point in geographic_points:
        coords.append(Coordinates(latitude=float(point.y), longitude=float(point.x)))
    return coords


def plot_selected_road(street: Street) -> Path:
    # Plota o eixo da rua e salva em output/streetview_images.
    output_dir = STREETVIEW_OUTPUT_DIR / _slugify(street.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_slugify(street.name)}_axis.png"

    x_coords, y_coords = street.axis.xy
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_coords, y_coords, color="red", linewidth=3)
    ax.set_title(street.name)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def _compute_heading(coords: list[Coordinates], index: int) -> float:
    if len(coords) == 1:
        return 0.0

    current = coords[index]
    if index < len(coords) - 1:
        target = coords[index + 1]
    else:
        target = coords[index - 1]
        current, target = target, current

    return _bearing_between(current, target)


def _bearing_between(start: Coordinates, end: Coordinates) -> float:
    lat1 = math.radians(start.latitude)
    lon1 = math.radians(start.longitude)
    lat2 = math.radians(end.latitude)
    lon2 = math.radians(end.longitude)

    delta_lon = lon2 - lon1
    x_value = math.sin(delta_lon) * math.cos(lat2)
    y_value = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    return (math.degrees(math.atan2(x_value, y_value)) + 360.0) % 360.0


def _get_streetview_metadata(coord: Coordinates, api_key: str) -> dict:
    response = requests.get(
        GOOGLE_STREETVIEW_METADATA_URL,
        params={
            "location": f"{coord.latitude:.7f},{coord.longitude:.7f}",
            "radius": str(GOOGLE_METADATA_RADIUS_METERS),
            "source": "outdoor",
            "key": api_key,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") != "OK":
        raise ValueError(
            "Google Street View metadata request failed for "
            f"{coord.latitude:.7f},{coord.longitude:.7f}: {payload.get('status')}"
        )
    return payload


def _compute_side_headings(forward_heading: float) -> dict[str, float]:
    return {
        side_name: (forward_heading + offset) % 360.0
        for side_name, offset in STREETVIEW_SIDE_OFFSETS.items()
    }


def _resolve_capture_coordinates(request_coord: Coordinates, metadata: dict) -> Coordinates:
    location = metadata.get("location", {})
    latitude = location.get("lat")
    longitude = location.get("lng")

    if latitude is None or longitude is None:
        return request_coord

    return Coordinates(latitude=float(latitude), longitude=float(longitude))


def _build_image_filename(index: int, side_name: str, coord: Coordinates) -> str:
    return (
        f"streetview_{index:04d}_{side_name}"
        f"_lat_{coord.latitude:.7f}"
        f"_lon_{coord.longitude:.7f}.jpg"
    )


def _filename_contains_coordinates(image_path: Path) -> bool:
    stem = image_path.stem.lower()
    return "_lat_" in stem and "_lon_" in stem


def _attach_gps_metadata(image_path: Path, coord: Coordinates) -> None:
    try:
        import piexif
    except ImportError:
        return

    exif_payload = {
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: "N" if coord.latitude >= 0 else "S",
            piexif.GPSIFD.GPSLatitude: _decimal_to_dms_rational(coord.latitude),
            piexif.GPSIFD.GPSLongitudeRef: "E" if coord.longitude >= 0 else "W",
            piexif.GPSIFD.GPSLongitude: _decimal_to_dms_rational(coord.longitude),
        }
    }
    piexif.insert(piexif.dump(exif_payload), str(image_path))


def _decimal_to_dms_rational(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    absolute_value = abs(value)
    degrees = int(absolute_value)
    minutes_float = (absolute_value - degrees) * 60.0
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60.0 * 10_000)

    return ((degrees, 1), (minutes, 1), (int(seconds), 10_000))


def _load_google_api_key() -> str:
    
    api_key = os.getenv("STREETVIEW_API_KEY")
    if api_key:
        return api_key

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        named_keys: dict[str, str] = {}
        fallback_values: list[str] = []

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not value or value == "[ENCRYPTION_KEY]":
                continue

            if key in {"STREETVIEW_API_KEY", "GOOGLE_API_KEY", "YOUR_API_KEY"}:
                named_keys[key] = value
                continue

            if not key and value.startswith("AIza"):
                fallback_values.append(value)

        for key_name in ("STREETVIEW_API_KEY", "GOOGLE_API_KEY", "YOUR_API_KEY"):
            if key_name in named_keys:
                return named_keys[key_name]

        if fallback_values:
            return fallback_values[-1]

    raise RuntimeError(
        "STREETVIEW_API_KEY or GOOGLE_API_KEY was not found. Set the environment variable or fix the .env file."
    )


def _merge_street_geometry(geometries: list[LineString | MultiLineString]) -> LineString:
    merged = linemerge(unary_union(geometries))
    if isinstance(merged, LineString):
        return merged

    if isinstance(merged, MultiLineString):
        return max(merged.geoms, key=lambda geometry: geometry.length)

    raise TypeError(f"Unsupported geometry returned for street axis: {type(merged)!r}")


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _slugify(value: str) -> str:
    normalized = _normalize_text(value)
    safe = "".join(char if char.isalnum() else "_" for char in normalized)
    return "_".join(part for part in safe.split("_") if part)
