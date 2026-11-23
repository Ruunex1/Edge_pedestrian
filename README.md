# EdgePed

EdgePed is a CPU-first pedestrian detection research and portfolio project.
The Week 1 baseline uses pretrained YOLO11n and the COCO `person` class.

## Windows setup

Run these commands from the repository root in PowerShell on 64-bit Windows.

```powershell
winget install --id astral-sh.uv --exact
```

Open a new PowerShell session so it picks up uv on PATH.

```powershell
uv python install 3.11
uv sync --locked
```

The lockfile targets Windows x64 and resolves PyTorch and torchvision from
the [PyTorch CPU index](https://docs.astral.sh/uv/guides/integration/pytorch/).
The first prediction downloads the default YOLO11n weights. Use images and
model weights you have permission to use. See the
[Ultralytics licensing terms](https://www.ultralytics.com/license) before redistribution.

## Detect pedestrians

Place a street photo you can use at `data/samples/street.jpg` or pass another
image path.

```powershell
uv run --locked python -m edgeped.cli --input data/samples/street.jpg
```

The CLI writes `outputs/street.annotated.png` and
`outputs/street.detections.json`. JSON boxes use original-image pixel
coordinates in `[x1, y1, x2, y2]` order. A result with no people has an
empty detection list. Use a distinct `--output-dir` for concurrent runs
with the same input stem.

Pass `--config path/to/settings.yaml` to override any of the packaged
defaults in `src/edgeped/baseline.yaml`. Relative weight paths resolve
from the current working directory. Use trusted weights.

`prediction_ms` covers the backend prediction call, including Ultralytics
preprocessing, postprocessing, and any first-call initialization. It
excludes weight loading, image decoding, annotation, and output writes.
This is a single-image measurement. EdgePed is intended for research and
portfolio demonstration.

## Development checks

```powershell
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Unit tests run with local inputs and a stub backend, without downloading weights.
