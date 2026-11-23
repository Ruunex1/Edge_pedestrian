# Week 2 BDD100K pilot

Pretrained YOLO11n ran on the Windows CPU against 64 images sampled without
replacement from the 10,000 BDD100K validation frames. The run used the
legacy detection annotations, inference size 640, a 0.001 prediction floor
for AP, and confidence 0.25 for the following precision and recall counts.
See the [evaluation rules](../README.md#evaluation-rules) for matching details.

| Time of day | Images | Pedestrians | TP | FP | FN | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 64 | 118 | 36 | 8 | 82 | 81.8% | 30.5% | 42.9% | 19.2% |
| Daytime | 40 | 97 | 32 | 7 | 65 | 82.1% | 33.0% | 45.2% | 20.1% |
| Night | 19 | 11 | 4 | 1 | 7 | 80.0% | 36.4% | 54.5% | 30.2% |
| Dawn/dusk | 5 | 10 | 0 | 0 | 10 | n/a | 0.0% | 8.3% | 3.1% |

The night subset contains only 11 labeled pedestrians, so its higher AP50
is too uncertain to infer a nighttime advantage. At the operating threshold,
52 of the 82 misses were small pedestrians. Recall was 5.5% for small boxes
(3/55), 50.0% for medium boxes (28/56), and 71.4% for large boxes (5/7).
Recall was 16.2% for occluded pedestrians (13/80) and 60.5% for visible
pedestrians (23/38). These size and occlusion counts can overlap.

The error overlays include a daytime false-positive example
(`c76fbaa1-cd6c2860.jpg`) and a nighttime miss example
(`ca28f275-cac4f254.jpg`). The eight false positives are unmatched
detections under the documented IoU and ignore rules.

## Reproduce the pilot

Use the [pinned validation mirror](https://huggingface.co/datasets/dronefreak/BDD100K/tree/fea86dd9b24ea425d66b70e0fcb6ae0023b5dca5),
which redistributes legacy annotations under the
[BDD100K dataset license](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst).
List the JSON paths in both shards under `data/labels/100k/val` and sample
with `sorted(random.Random(42).sample(sorted(paths), 64))` in Python 3.11.
Download the corresponding JPEGs from the same shards under
`data/images_100k/100k/val`. Set each annotation's `name` to its JPEG
basename and combine the records in sampled order with
`json.dumps(frames, sort_keys=True)` in UTF-8. The resulting annotation
SHA-256 was `b1c047459aa4d799b5ab1544080118a5e6ccf7453320f1041d7404907ddd5d99`.

```powershell
uv run --locked python -m edgeped.evaluate_cli --images data/bdd100k-pilot/images --labels data/bdd100k-pilot/val.json --output-dir outputs/bdd100k-week2
```

This run used Python 3.11.18, torch 2.15.0+cpu, and Ultralytics 8.4.216.
The generated metrics JSON records all 64 image names. The full-split
evaluation command is in the [README](../README.md#evaluate-on-bdd100k).
