import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import yaml

from edgeped.config import DetectorConfig
from edgeped.detector import detect
from edgeped.image_io import annotate, load_image, save_png

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect pedestrians in one image on CPU."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input image path")
    parser.add_argument("--config", type=Path, help="YAML overrides for the baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = DetectorConfig.load(args.config)
        image = load_image(args.input)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        from edgeped.pytorch_backend import PyTorchBackend

        result = detect(image, PyTorchBackend(config))
        payload = {
            "input": str(args.input),
            "width": image.shape[1],
            "height": image.shape[0],
            "backend": "pytorch",
            "device": "cpu",
            "config": asdict(config),
            **asdict(result),
        }
        with TemporaryDirectory(dir=args.output_dir, prefix=".edgeped-") as staging:
            png = Path(staging) / f"{args.input.stem}.annotated.png"
            report = Path(staging) / f"{args.input.stem}.detections.json"
            save_png(png, annotate(image, result.detections))
            report.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n", "utf-8"
            )
            png.replace(args.output_dir / png.name)
            report.replace(args.output_dir / report.name)
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

    logger.info(
        "%d person detections in %.1f ms. Results saved to %s",
        len(result.detections),
        result.prediction_ms,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
