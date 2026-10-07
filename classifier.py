
def classify_images(
    image_dir: str | Path | None = None,
    debug: bool = False,
    log:bool = False
) -> list[ClassificationResult]:
    """Classify all supported images in a directory or a single image path."""
    model, class_names = _load_classifier_assets()
    images_root = _resolve_images_directory(image_dir)
    image_paths = _collect_image_paths(images_root)

    if not image_paths:
        raise ValueError(f"No images were found in '{images_root}'.")

    results: list[ClassificationResult] = []
    for image_path in image_paths:
        image = _load_image(image_path)
        prediction = model.predict(image, verbose=0)
        index = int(np.argmax(prediction[0]))
        if debug:
            print(class_names[index],image_path )
        results.append(
            ClassificationResult(
                image_path=str(image_path.resolve()),
                predicted_label=class_names[index],
                confidence_score=float(prediction[0][index]),
            )
        )

    if log:
        source_name = images_root.stem if images_root.is_file() else images_root.name
        _write_results(results, source_name)

    return results


@lru_cache(maxsize=1)
def _load_classifier_assets():
    np.set_printoptions(suppress=True)
    model_path = CURRENT_DIR / "keras_model.h5"
    labels_path = CURRENT_DIR / "labels.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file was not found: {labels_path}")

    patched_model_path = _prepare_legacy_model(model_path)
    model = load_model(patched_model_path, compile=False)
    class_names = [
        line.strip().split(" ", 1)[1] if " " in line.strip() else line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return model, class_names


def _load_image(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data


def _resolve_images_directory(image_dir: str | Path | None) -> Path:
    if image_dir is not None:
        images_root = Path(image_dir)
    else:
        images_root = DEFAULT_IMAGES_DIR

    if not images_root.exists():
        raise FileNotFoundError(f"Images directory was not found: {images_root}")
    return images_root


def _collect_image_paths(images_root: Path) -> list[Path]:
    if images_root.is_file():
        if images_root.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {images_root.suffix}")
        return [images_root]

    return sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _write_results(results: list[ClassificationResult], source_name: str) -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{source_name}_classification.csv"
    json_path = output_dir / f"{source_name}_classification.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_path", "predicted_label", "confidence_score"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    json_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _prepare_legacy_model(model_path: Path) -> Path:
    """Patch legacy H5 config fields that newer Keras rejects."""
    model_bytes = model_path.read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()[:12]
    patched_path = Path(tempfile.gettempdir()) / f"{model_path.stem}_{digest}_patched.h5"

    if patched_path.exists():
        return patched_path

    with h5py.File(model_path, "r") as source_file:
        model_config = source_file.attrs.get("model_config")
        if model_config is None:
            return model_path

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        config_payload = json.loads(model_config)
        _strip_legacy_depthwise_groups(config_payload)

    patched_path.write_bytes(model_bytes)

    with h5py.File(patched_path, "r+") as patched_file:
        patched_file.attrs["model_config"] = json.dumps(config_payload).encode("utf-8")

    return patched_path


def _strip_legacy_depthwise_groups(node: object) -> None:
    if isinstance(node, dict):
        class_name = node.get("class_name")
        config = node.get("config")
        if class_name == "DepthwiseConv2D" and isinstance(config, dict):
            config.pop("groups", None)

        for value in node.values():
            _strip_legacy_depthwise_groups(value)
    elif isinstance(node, list):
        for item in node:
            _strip_legacy_depthwise_groups(item)


if __name__ == "__main__":
    classify_images()
