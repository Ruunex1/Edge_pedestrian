from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    label: str = "person"


class InferenceBackend(Protocol):
    def predict(self, image: Image) -> list[Detection]:
        """Return detections in original-image pixel coordinates."""


@dataclass(frozen=True, slots=True)
class DetectionResult:
    detections: list[Detection]
    prediction_ms: float


def detect(image: Image, backend: InferenceBackend) -> DetectionResult:
    start = perf_counter()
    detections = backend.predict(image)
    return DetectionResult(detections, (perf_counter() - start) * 1000)
