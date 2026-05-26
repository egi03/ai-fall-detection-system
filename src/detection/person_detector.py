"""
Person detection wrapper using YOLOv8 (Ultralytics).

Detects persons in video frames and returns bounding boxes with
confidence scores. Uses YOLOv8 nano by default for optimal
speed/accuracy trade-off on CPU.

DECISION: Using YOLOv8n per research/2.1. It achieves ~56ms ONNX CPU
latency (~17.8 FPS), satisfying the 15 FPS minimum for fall detection.
The anchor-free design and OBB support improve detection of horizontal
(lying-down) subjects.

Reference: research/2.1 - YOLOv8 provides best speed/accuracy for
real-time CPU inference and reliable detection in non-standard poses.
Reference: research/2.2 - Bounding box aspect ratio is a primary
geometric feature for fall detection.
"""

from typing import List, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# COCO class ID for 'person'
_PERSON_CLASS_ID = 0

# Minimum bounding box dimension to avoid degenerate detections
_MIN_BBOX_DIM = 5


class PersonDetector:
    """
    Wrapper around YOLOv8 for person detection in video frames.

    Filters detections to only the 'person' class and provides
    bounding boxes with confidence scores and aspect ratios.

    Parameters
    ----------
    model_name : str
        YOLOv8 model variant: 'yolov8n', 'yolov8s', 'yolov8m'.
    confidence_threshold : float
        Minimum confidence score to accept a detection.
    iou_threshold : float
        IoU threshold for non-maximum suppression.
    device : str
        Inference device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package is required for person detection. "
                "Install with: pip install ultralytics"
            ) from e

        self._conf_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._device = device

        # Load pretrained model (downloads automatically if not cached)
        model_path = f"{model_name}.pt"
        logger.info(
            "Loading YOLOv8 model",
            extra={"model": model_name, "device": device},
        )
        self._model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Detect persons in a single video frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array of shape (H, W, 3).

        Returns
        -------
        list of dict
            Each dict contains:
            - 'bbox': (x1, y1, x2, y2) bounding box in pixel coordinates
            - 'confidence': float detection confidence
            - 'aspect_ratio': float width/height ratio
        """
        results = self._model(
            frame,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            classes=[_PERSON_CLASS_ID],
            device=self._device,
            verbose=False,
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id != _PERSON_CLASS_ID:
                continue

            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            # Skip degenerate boxes
            if (x2 - x1) < _MIN_BBOX_DIM or (y2 - y1) < _MIN_BBOX_DIM:
                continue

            bbox = (int(x1), int(y1), int(x2), int(y2))

            detections.append({
                "bbox": bbox,
                "confidence": confidence,
                "aspect_ratio": self.compute_aspect_ratio(bbox),
            })

        return detections

    @staticmethod
    def compute_aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
        """
        Compute width/height aspect ratio of a bounding box.

        DECISION: Using width/height ratio per research/2.2.
        AR > 1.0 indicates horizontal posture (potential fall).
        AR < 1.0 indicates upright posture (normal activity).

        Parameters
        ----------
        bbox : tuple
            (x1, y1, x2, y2) bounding box coordinates.

        Returns
        -------
        float
            Aspect ratio (width / height). Returns NaN if height is zero.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if height <= 0:
            return float("nan")

        return width / height

    @staticmethod
    def compute_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Compute the centroid of a bounding box.

        Parameters
        ----------
        bbox : tuple
            (x1, y1, x2, y2) bounding box coordinates.

        Returns
        -------
        tuple of float
            (cx, cy) centroid coordinates.
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def compute_area(bbox: Tuple[int, int, int, int]) -> int:
        """
        Compute the area of a bounding box.

        Parameters
        ----------
        bbox : tuple
            (x1, y1, x2, y2) bounding box coordinates.

        Returns
        -------
        int
            Area in pixels.
        """
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)
