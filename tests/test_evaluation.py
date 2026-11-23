import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from edgeped.config import DetectorConfig
from edgeped.datasets.bdd100k import Frame, GroundTruth, load_annotations
from edgeped.detector import Detection, Image
from edgeped.evaluate_cli import main, select_frames
from edgeped.evaluation import (
    analyze_errors,
    average_precision,
    evaluate,
    match,
    overlap,
)

BOX = (10.0, 10.0, 30.0, 30.0)
OTHER = (40.0, 40.0, 60.0, 60.0)


def test_annotations(tmp_path: Path) -> None:
    labels = tmp_path / "val.json"
    box = dict(zip(("x1", "y1", "x2", "y2"), BOX, strict=True))
    labels.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "name": "a.jpg",
                        "attributes": {"timeofday": "night"},
                        "labels": [
                            {"category": "person", "box2d": box},
                            {
                                "category": "pedestrian",
                                "box2d": box,
                                "attributes": {"occluded": True},
                            },
                            {"category": "rider", "box2d": box},
                            {"category": "car"},
                        ],
                    },
                    {"name": "b.jpg", "labels": None},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_annotations(labels) == [
        Frame(
            "a.jpg",
            "night",
            (
                GroundTruth(BOX, None),
                GroundTruth(BOX, True),
                GroundTruth(BOX, None, True),
            ),
        ),
        Frame("b.jpg", "undefined", ()),
    ]


def test_one_to_one_and_ignored_crowd() -> None:
    frame = Frame(
        "a.jpg",
        "daytime",
        (GroundTruth(BOX, False), GroundTruth((35, 35, 100, 100), None, True, True)),
    )
    result = match(
        frame,
        [Detection(BOX, 0.7), Detection(OTHER, 0.6), Detection(BOX, 0.9)],
        0.5,
    )
    assert [(p.target, p.ignored) for p in result.predictions] == [
        (0, False),
        (None, False),
        (None, True),
    ]
    assert result.missed == set()
    assert overlap(BOX, (10, 10, 20, 30)) == 0.5


def test_legacy_nested_objects(tmp_path: Path) -> None:
    path = tmp_path / "val.json"
    frame = {
        "name": "old.jpg",
        "frames": [
            {
                "objects": [
                    {
                        "category": "person",
                        "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
                    }
                ]
            }
        ],
    }
    path.write_text(json.dumps([frame]), encoding="utf-8")
    assert load_annotations(path) == [
        Frame("old.jpg", "undefined", (GroundTruth(BOX, None),))
    ]


def test_ap_and_operating_threshold() -> None:
    day = Frame("day.jpg", "daytime", (GroundTruth(BOX, True),))
    night = Frame("night.jpg", "night", (GroundTruth(OTHER, False),))
    predictions = {
        day.name: [Detection(OTHER, 0.9), Detection(BOX, 0.1)],
        night.name: [],
    }
    metrics, matches = evaluate([day, night], predictions, 0.25)
    assert metrics["overall"]["ap_50"] == pytest.approx(0.5 * 51 / 101)
    assert metrics["daytime"]["ap_50_95"] == pytest.approx(0.5)
    assert metrics["night"]["ap_50"] == 0
    assert metrics["overall"]["recall"] == 0
    assert metrics["overall"]["precision"] == 0
    assert metrics["night"]["precision"] is None
    assert average_precision([], 0) is None
    errors = analyze_errors([day, night], matches)
    assert errors["by_size"]["small"]["fn"] == 2
    assert errors["by_occlusion"]["occluded"]["fn"] == 1
    assert len(errors["images"][0]["false_positives"]) == 1


def test_seeded_sampling() -> None:
    frames = [Frame(f"{i}.jpg", "daytime" if i < 4 else "night", ()) for i in range(8)]
    selected = select_frames(frames, 2, 42)
    assert selected == select_frames(frames, 2, 42)
    assert [frame.timeofday for frame in selected] == ["daytime"] * 2 + ["night"] * 2


def test_evaluation_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class StubBackend:
        def __init__(self, config: DetectorConfig) -> None:
            assert config.confidence == 0.001

        def predict(self, image: Image) -> list[Detection]:
            return [Detection(BOX, 0.9)]

    monkeypatch.setattr("edgeped.pytorch_backend.PyTorchBackend", StubBackend)
    labels = tmp_path / "val.json"
    labels.write_text('[{"name":"a.jpg","attributes":{"timeofday":"daytime"}}]')
    output = tmp_path / "results"
    argv = [
        "--images",
        str(tmp_path),
        "--labels",
        str(labels),
        "--output-dir",
        str(output),
    ]
    assert main(argv) == 1
    assert not output.exists()
    assert cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((80, 80, 3), dtype=np.uint8))
    assert main(argv) == 0
    payload = json.loads((output / "metrics.json").read_text())
    assert payload["run"]["images"] == ["a.jpg"]
    assert payload["metrics"]["daytime"]["fp"] == 1
    assert payload["metrics"]["daytime"]["ap_50"] is None
    assert (output / "examples" / "daytime-fp-a.jpg.png").is_file()
    assert (output / "report.md").is_file()
    assert main(argv) == 1
