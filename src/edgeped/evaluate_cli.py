import argparse
import hashlib
import json
import logging
import platform
import random
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import cv2
import yaml

from edgeped.config import DetectorConfig
from edgeped.datasets.bdd100k import Frame, load_annotations
from edgeped.detector import Detection, InferenceBackend
from edgeped.evaluation import ImageMatches, analyze_errors, evaluate
from edgeped.image_io import load_image, save_png

logger = logging.getLogger(__name__)


def select_frames(
    frames: list[Frame], per_condition: int | None, seed: int
) -> list[Frame]:
    if per_condition is None:
        return frames
    rng = random.Random(seed)
    groups: dict[str, list[Frame]] = {}
    for frame in frames:
        groups.setdefault(frame.timeofday, []).append(frame)
    selected = [
        frame
        for condition in sorted(groups)
        for frame in rng.sample(
            groups[condition], min(per_condition, len(groups[condition]))
        )
    ]
    return sorted(selected, key=lambda frame: frame.name)


def run_predictions(
    frames: list[Frame], image_dir: Path, backend: InferenceBackend
) -> dict[str, list[Detection]]:
    predictions = {}
    for index, frame in enumerate(frames, start=1):
        predictions[frame.name] = backend.predict(load_image(image_dir / frame.name))
        if index % 100 == 0 or index == len(frames):
            logger.info("Processed %d/%d images", index, len(frames))
    return predictions


def write_examples(
    directory: Path,
    image_dir: Path,
    frames: list[Frame],
    operating: dict[str, ImageMatches],
) -> None:
    for condition in sorted({frame.timeofday for frame in frames}):
        subset = [frame for frame in frames if frame.timeofday == condition]
        for error in ("fp", "fn"):

            def count(frame: Frame) -> int:
                matches = operating[frame.name]
                if error == "fn":
                    return len(matches.missed)
                return sum(
                    p.target is None and not p.ignored for p in matches.predictions
                )

            frame = max(subset, key=count)
            if count(frame) == 0:
                continue
            directory.mkdir(exist_ok=True)
            image = load_image(image_dir / frame.name)
            matches = operating[frame.name]
            boxes = [
                (p.prediction.xyxy, "FP", (0, 0, 255))
                for p in matches.predictions
                if p.target is None and not p.ignored
            ]
            boxes.extend(
                (frame.objects[i].box, "FN", (0, 190, 255))
                for i in sorted(matches.missed)
            )
            for box, label, color in boxes:
                x1, y1, x2, y2 = map(round, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    label,
                    (x1, max(14, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            filename = f"{condition.replace('/', '-')}-{error}-{frame.name}.png"
            save_png(directory / filename, image)


def markdown_report(
    metrics: dict, errors: dict, floor: float, confidence: float
) -> str:
    def percentage(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        "# EdgePed BDD100K evaluation",
        "",
        f"Precision, recall, and counts use IoU 0.50 and confidence {confidence:g}.",
        f"AP uses detections returned above the prediction floor of {floor:g}.",
        "",
        "| Time of day | Images | Pedestrians | TP | FP | FN |"
        " Precision | Recall | AP50 | AP50–95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        counts = [str(row[key]) for key in ("images", "ground_truth", "tp", "fp", "fn")]
        scores = [
            percentage(row[key]) for key in ("precision", "recall", "ap_50", "ap_50_95")
        ]
        lines.append("| " + " | ".join([name, *counts, *scores]) + " |")
    for key, caption in (("by_size", "Box area"), ("by_occlusion", "Occlusion")):
        lines.extend(
            [
                "",
                f"## Recall by {caption.lower()}",
                "",
                "| Group | TP | FN | Recall |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, row in errors[key].items():
            lines.append(
                f"| {name} | {row['tp']} | {row['fn']} | {percentage(row['recall'])} |"
            )
    lines.extend(
        [
            "",
            "See `errors.json` for per-image boxes and `examples/` for error overlays.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11n on BDD100K detection labels."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prediction-floor", type=float, default=0.001)
    parser.add_argument(
        "--per-condition", type=int, help="Sample up to N images per time of day"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = DetectorConfig.load(args.config)
        if not 0 < args.prediction_floor <= config.confidence:
            raise ValueError(
                "Prediction floor must be positive and at most the operating confidence"
            )
        if args.per_condition is not None and args.per_condition < 1:
            raise ValueError("--per-condition must be positive")
        if args.output_dir.exists():
            raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
        frames = select_frames(
            load_annotations(args.labels), args.per_condition, args.seed
        )
        missing = [
            frame.name for frame in frames if not (args.images / frame.name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} selected images missing, first: {missing[0]}"
            )

        from edgeped.pytorch_backend import PyTorchBackend

        predictions = run_predictions(
            frames,
            args.images,
            PyTorchBackend(replace(config, confidence=args.prediction_floor)),
        )
        metrics, operating = evaluate(frames, predictions, config.confidence)
        errors = analyze_errors(frames, operating)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_examples(args.output_dir / "examples", args.images, frames, operating)
        metadata = {
            "dataset": "BDD100K detection",
            "labels": str(args.labels),
            "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
            "device": "cpu",
            "backend": "pytorch",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": version("torch"),
            "ultralytics": version("ultralytics"),
            "detector": asdict(config),
            "prediction_floor": args.prediction_floor,
            "per_condition": args.per_condition,
            "seed": args.seed,
            "images": [frame.name for frame in frames],
        }
        for name, payload in (
            ("metrics", {"run": metadata, "metrics": metrics}),
            ("errors", errors),
        ):
            (args.output_dir / f"{name}.json").write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        (args.output_dir / "report.md").write_text(
            markdown_report(metrics, errors, args.prediction_floor, config.confidence),
            encoding="utf-8",
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        cv2.error,
        yaml.YAMLError,
    ) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Evaluation saved to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
