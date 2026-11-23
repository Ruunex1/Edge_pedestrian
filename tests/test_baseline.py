import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from edgeped.cli import main
from edgeped.config import DetectorConfig
from edgeped.detector import Detection, Image


@pytest.mark.parametrize(
    "override", ["confidence: true", "image_size: 33", "device: cuda"]
)
def test_invalid_config(tmp_path: Path, override: str) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(override, encoding="utf-8")
    with pytest.raises(ValueError):
        DetectorConfig.load(path)


@pytest.mark.parametrize("payload", [b"", b"invalid image"])
def test_invalid_image(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "bad.png"
    path.write_bytes(payload)
    assert main(["--input", str(path), "--output-dir", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("detections", [[], [Detection((2, 3, 30, 32), 0.9)]])
def test_cli_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detections: list[Detection]
) -> None:
    class StubBackend:
        def __init__(self, config: DetectorConfig) -> None:
            assert config.weights == "yolo11n.pt"

        def predict(self, image: Image) -> list[Detection]:
            assert image.shape == (40, 50, 3)
            return detections

    monkeypatch.setattr("edgeped.pytorch_backend.PyTorchBackend", StubBackend)
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)
    path = tmp_path / "scène.png"
    path.write_bytes(encoded.tobytes())
    output = tmp_path / "out"

    assert main(["--input", str(path), "--output-dir", str(output)]) == 0
    report = json.loads((output / "scène.detections.json").read_text(encoding="utf-8"))
    assert report["prediction_ms"] >= 0
    assert (report["width"], report["height"]) == (50, 40)
    assert report["detections"] == [
        {"xyxy": list(item.xyxy), "confidence": item.confidence, "label": "person"}
        for item in detections
    ]
    annotated = cv2.imdecode(
        np.frombuffer((output / "scène.annotated.png").read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert bool(np.any(annotated)) == bool(detections)
