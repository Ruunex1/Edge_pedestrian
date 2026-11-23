from pathlib import Path

import cv2
import numpy as np

from edgeped.detector import Detection, Image


def load_image(path: Path) -> Image:
    data = path.read_bytes()
    if not data:
        raise ValueError(f"Image is empty: {path}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def annotate(image: Image, detections: list[Detection]) -> Image:
    annotated = image.copy()
    color = (0, 220, 0)
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def save_png(path: Path, image: Image) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())
