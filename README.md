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

## Evaluate on BDD100K

Download the 100K images and detection annotations from the
[BDD100K download page](https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst).
Keep them under `data/` and follow the
[BDD100K dataset license](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst).
The evaluator reads split-level Scalabel JSON, either a frame list or an
object with a `frames` list. It also reads a JSON list of legacy annotations
with one nested detection frame per image. Point `--images` at the
directory containing the validation JPEGs.

```powershell
uv run --locked python -m edgeped.evaluate_cli --images data/bdd100k/images/100k/val --labels data/bdd100k/labels/det_20/det_val.json --output-dir outputs/bdd100k-val
```

For a shorter run, append `--per-condition 25 --seed 42`. The sampler
selects up to 25 frames from each time-of-day group, including frames with
no pedestrian labels. Choose a fresh output directory for each run.
`metrics.json` records the model settings, package versions, annotation
checksum, and selected image names. `report.md` summarizes the results,
`errors.json` records per-image false positives and misses, and
`examples/` contains overlays of error-heavy frames.
The [Week 2 pilot report](reports/week2-bdd100k.md) records measured
results from a seeded subset of 64 validation images.

### Evaluation rules

- BDD100K `pedestrian` and the legacy `person` label map to COCO `person`.
  Riders and pedestrian annotations flagged as ignored or crowd are ignore
  regions. Other categories contribute no ground-truth boxes.
- Predictions are sorted by confidence and greedily matched to the
  highest-IoU unmatched pedestrian in the same image. Unmatched predictions
  overlapping an ignore region at the threshold are omitted. Crowd regions
  use intersection over prediction area and can absorb multiple detections.
- Precision, recall, and error counts use IoU 0.50 and the configured
  operating confidence (default 0.25). AP uses a prediction floor of 0.001
  by default and 101-point interpolated precision. AP50–95 averages ten
  IoU thresholds from 0.50 through 0.95. Empty denominators are reported as
  `null`. These metrics follow the rules above for this experiment.
- Recall breakdowns use original box area in pixels squared, with boundaries
  at 32² and 96², and the annotated occlusion flag where available. Day,
  night, dawn/dusk, and undefined conditions are reported separately when
  present. Overall metrics pool all selected frames, including negatives.

## Development checks

```powershell
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Unit tests run with local inputs and a stub backend, without downloading weights.
