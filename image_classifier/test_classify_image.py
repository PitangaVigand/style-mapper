from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

from common.protocol import ClassificationResult


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "gen_street_view_images_by_street" / "streetview_images"

IMAGE_SIZE = (224, 224)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def test_classify_image() -> None:
    example_image = PROJECT_ROOT / "examples" / "building (2).jpeg"
    result = classify_image(example_image)

    assert result.image_path.endswith("building (2).jpeg")
    assert result.predicted_label == "neo"
    assert 0.0 <= result.confidence_score <= 1.0


def test_classify_images_av_Julio_de_mesquita() -> None:
    images_root = PROJECT_ROOT / "gen_street_view_images_by_street/streetview_images/avenida_julio_de_mesquita"
    results = classify_images(images_root, debug=True, log=True)
    
    assert results

def test_classify_images_examples() -> None:
    images_root = PROJECT_ROOT / "examples"
    results = classify_images(images_root)

    assert results
  

def classify_image(image_path: str | Path) -> ClassificationResult:
    model, class_names = _load_classifier_assets()
    image = _load_image(image_path)
    prediction = model.predict(image, verbose=0)
    index = int(np.argmax(prediction[0]))

    return ClassificationResult(
        image_path=str(Path(image_path).resolve()),
        predicted_label=class_names[index],
        confidence_score=float(prediction[0][index]),
    )
