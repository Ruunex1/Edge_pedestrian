from collections import Counter
from dataclasses import asdict, dataclass

from edgeped.datasets.bdd100k import Box, Frame
from edgeped.detector import Detection

IOU_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


@dataclass(frozen=True, slots=True)
class PredictionMatch:
    prediction: Detection
    target: int | None
    ignored: bool = False


@dataclass(slots=True)
class ImageMatches:
    predictions: list[PredictionMatch]
    missed: set[int]


def overlap(a: Box, b: Box, *, crowd: bool = False) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denominator = area_a if crowd else area_a + area_b - intersection
    return intersection / denominator if denominator > 0 else 0.0


def match(frame: Frame, predictions: list[Detection], threshold: float) -> ImageMatches:
    unmatched = {i for i, target in enumerate(frame.objects) if not target.ignored}
    ignored = [target for target in frame.objects if target.ignored]
    outcomes = []
    for prediction in sorted(predictions, key=lambda item: -item.confidence):
        score, index = max(
            ((overlap(prediction.xyxy, frame.objects[i].box), -i) for i in unmatched),
            default=(0.0, 0),
        )
        if score >= threshold:
            unmatched.remove(-index)
            outcomes.append(PredictionMatch(prediction, -index))
            continue
        suppress = any(
            overlap(prediction.xyxy, target.box, crowd=target.crowd) >= threshold
            for target in ignored
        )
        outcomes.append(PredictionMatch(prediction, None, suppress))
    return ImageMatches(outcomes, unmatched)


def average_precision(outcomes: list[PredictionMatch], positives: int) -> float | None:
    if not positives:
        return None
    tp = fp = 0
    points = [(0.0, 0.0)]
    for outcome in sorted(outcomes, key=lambda item: -item.prediction.confidence):
        if outcome.ignored:
            continue
        tp += outcome.target is not None
        fp += outcome.target is None
        points.append((tp / positives, tp / (tp + fp)))
    # Sample the right-to-left precision envelope on 101 recall levels.
    total = 0.0
    best = 0.0
    index = len(points) - 1
    for level in range(100, -1, -1):
        while index >= 0 and points[index][0] >= level / 100:
            best = max(best, points[index][1])
            index -= 1
        total += best
    return total / 101


def evaluate(
    frames: list[Frame],
    predictions: dict[str, list[Detection]],
    confidence: float,
) -> tuple[dict, dict[str, ImageMatches]]:
    groups = {"overall": frames}
    groups.update(
        (condition, [frame for frame in frames if frame.timeofday == condition])
        for condition in sorted(
            {"daytime", "night", *(frame.timeofday for frame in frames)}
        )
    )
    operating = {
        frame.name: match(
            frame,
            [p for p in predictions[frame.name] if p.confidence >= confidence],
            0.50,
        )
        for frame in frames
    }
    metrics = {
        condition: _operating_metrics(subset, operating)
        for condition, subset in groups.items()
    }
    for threshold in IOU_THRESHOLDS:
        matched = {
            frame.name: match(frame, predictions[frame.name], threshold)
            for frame in frames
        }
        for condition, subset in groups.items():
            summary = metrics[condition]
            outcomes = [p for frame in subset for p in matched[frame.name].predictions]
            summary["ap_by_iou"][f"{threshold:.2f}"] = average_precision(
                outcomes, summary["ground_truth"]
            )
    for summary in metrics.values():
        ap_values = [ap for ap in summary["ap_by_iou"].values() if ap is not None]
        summary["ap_50"] = summary["ap_by_iou"]["0.50"]
        summary["ap_50_95"] = sum(ap_values) / len(ap_values) if ap_values else None
    return metrics, operating


def _operating_metrics(frames: list[Frame], operating: dict[str, ImageMatches]) -> dict:
    tp = fp = fn = ignored = 0
    for frame in frames:
        image = operating[frame.name]
        counts = Counter(
            "ignored" if p.ignored else "tp" if p.target is not None else "fp"
            for p in image.predictions
        )
        tp += counts["tp"]
        fp += counts["fp"]
        ignored += counts["ignored"]
        fn += len(image.missed)
    return {
        "images": len(frames),
        "ground_truth": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": ignored,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "ap_by_iou": {},
    }


def analyze_errors(frames: list[Frame], operating: dict[str, ImageMatches]) -> dict:
    size: dict[str, Counter] = {key: Counter() for key in ("small", "medium", "large")}
    occlusion: dict[str, Counter] = {
        key: Counter() for key in ("occluded", "visible", "unknown")
    }
    images = []
    for frame in frames:
        outcomes = operating[frame.name]
        for index, target in enumerate(frame.objects):
            if target.ignored:
                continue
            x1, y1, x2, y2 = target.box
            area = (x2 - x1) * (y2 - y1)
            size_key = (
                "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            )
            visibility = {True: "occluded", False: "visible", None: "unknown"}
            status = "fn" if index in outcomes.missed else "tp"
            size[size_key][status] += 1
            occlusion[visibility[target.occluded]][status] += 1
        false_positives = [
            asdict(p.prediction)
            for p in outcomes.predictions
            if p.target is None and not p.ignored
        ]
        missed = [asdict(frame.objects[i]) for i in sorted(outcomes.missed)]
        if false_positives or missed:
            images.append(
                {
                    "image": frame.name,
                    "timeofday": frame.timeofday,
                    "false_positives": false_positives,
                    "false_negatives": missed,
                }
            )
    return {
        "by_size": _recalls(size),
        "by_occlusion": _recalls(occlusion),
        "images": images,
    }


def _recalls(buckets: dict[str, Counter]) -> dict:
    return {
        key: {
            "tp": counts["tp"],
            "fn": counts["fn"],
            "recall": counts["tp"] / counts.total() if counts.total() else None,
        }
        for key, counts in buckets.items()
    }
