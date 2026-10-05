from ultralytics import YOLO


class Detector:
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(self, model_name: str = "yolo11n.pt", device: str = "cpu"):
        self.model = YOLO(model_name)
        self.device = device
        self.class_names = self.model.names if self.model.names else {
            i: name for i, name in enumerate(self.COCO_CLASSES)
        }

    def detect(self, frame, conf: float = 0.5):
        results = self.model(frame, device=self.device, conf=conf, verbose=False)
        return results[0]

    def draw(self, frame, result):
        annotated = result.plot()
        return annotated

    @staticmethod
    def get_results_data(result):
        detections = []
        if result.boxes is not None:
            for box, cls_id, conf_val in zip(
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf,
            ):
                detections.append({
                    "bbox": box.tolist(),
                    "class_id": int(cls_id),
                    "confidence": float(conf_val),
                })
        return detections
