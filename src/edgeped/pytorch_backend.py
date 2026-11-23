from ultralytics import YOLO

from edgeped.config import DetectorConfig
from edgeped.detector import Detection, Image

COCO_PERSON_ID = 0


class PyTorchBackend:
    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._model = YOLO(config.weights, task="detect")
        if self._model.names.get(COCO_PERSON_ID) != "person":
            raise ValueError("Model must map COCO class 0 to person")

    def predict(self, image: Image) -> list[Detection]:
        result = self._model.predict(
            source=image,
            device="cpu",
            classes=[COCO_PERSON_ID],
            conf=self._config.confidence,
            imgsz=self._config.image_size,
            verbose=False,
            save=False,
        )[0]
        if result.boxes is None:
            return []
        detections = []
        for box in result.boxes.cpu().data.tolist():
            x1, y1, x2, y2, confidence, class_id = box
            if class_id == COCO_PERSON_ID:
                detections.append(Detection((x1, y1, x2, y2), confidence))
        return detections
