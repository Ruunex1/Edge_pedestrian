import json
import math
from dataclasses import dataclass
from pathlib import Path

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GroundTruth:
    box: Box
    occluded: bool | None
    ignored: bool = False
    crowd: bool = False


@dataclass(frozen=True, slots=True)
class Frame:
    name: str
    timeofday: str
    objects: tuple[GroundTruth, ...]


def load_annotations(path: Path) -> list[Frame]:
    """Read a BDD100K detection split in Scalabel JSON format."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        raw = raw.get("frames")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Expected a nonempty list of BDD100K frames")

    frames = []
    seen: set[str] = set()
    for entry in raw:
        try:
            name = entry["name"]
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}
                or name.casefold() in seen
            ):
                raise ValueError("Expected a unique image filename")
            timeofday = (entry.get("attributes") or {}).get("timeofday", "undefined")
            if not isinstance(timeofday, str):
                raise ValueError("Expected a time-of-day string")
            timeofday = timeofday.lower().strip().replace("dawn-dusk", "dawn/dusk")
            if timeofday not in {"daytime", "night", "dawn/dusk"}:
                timeofday = "undefined"
            labels = entry.get("labels")
            if "frames" in entry:
                legacy_frames = entry["frames"]
                if not isinstance(legacy_frames, list) or len(legacy_frames) != 1:
                    raise ValueError(
                        "Expected exactly one legacy detection frame per image"
                    )
                labels = legacy_frames[0]["objects"]
            if labels is not None and not isinstance(labels, list):
                raise ValueError("Expected a list of object labels")
            objects = tuple(
                _parse_object(label)
                for label in labels or []
                if label["category"] in {"pedestrian", "person", "rider"}
            )
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ValueError(
                f"Invalid BDD100K frame at index {len(frames)}: {exc}"
            ) from exc
        frames.append(Frame(name, timeofday, objects))
        seen.add(name.casefold())
    return sorted(frames, key=lambda frame: frame.name)


def _parse_object(label: dict) -> GroundTruth:
    coordinates = label["box2d"]
    x1, y1, x2, y2 = (float(coordinates[key]) for key in ("x1", "y1", "x2", "y2"))
    if not all(map(math.isfinite, (x1, y1, x2, y2))) or x1 >= x2 or y1 >= y2:
        raise ValueError("Expected a finite box with positive area")
    attrs = label.get("attributes") or {}
    occluded = attrs.get("occluded")
    crowd = attrs.get("crowd") is True or attrs.get("iscrowd") is True
    return GroundTruth(
        (x1, y1, x2, y2),
        occluded if isinstance(occluded, bool) else None,
        label["category"] == "rider" or crowd or attrs.get("ignore") is True,
        crowd,
    )
