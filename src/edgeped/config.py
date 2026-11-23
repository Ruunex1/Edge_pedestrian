from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    weights: str
    confidence: float
    image_size: int

    @classmethod
    def load(cls, path: Path | None = None) -> "DetectorConfig":
        baseline = files("edgeped").joinpath("baseline.yaml")
        settings = yaml.safe_load(baseline.read_text(encoding="utf-8"))
        if path is not None:
            overrides = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(overrides, dict) or overrides.keys() - settings.keys():
                raise ValueError(
                    "Expected a YAML mapping with weights, confidence, or image_size"
                )
            settings.update(overrides)
        weights = settings["weights"]
        confidence = settings["confidence"]
        image_size = settings["image_size"]
        if not isinstance(weights, str) or not weights.strip():
            raise ValueError("weights must be a nonempty string")
        if type(confidence) not in (int, float) or not 0 < confidence <= 1:
            raise ValueError("confidence must be a number in (0, 1]")
        if type(image_size) is not int or image_size <= 0 or image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        return cls(weights, float(confidence), image_size)
