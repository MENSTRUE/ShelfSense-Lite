from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "shelfsense_yolo11s_best.pt"
)

DEFAULT_IMGSZ = 960
DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.60
DEFAULT_MAX_DET = 800


@dataclass
class DetectionResult:
    image: np.ndarray
    annotated_image: np.ndarray

    count: int

    boxes: np.ndarray
    confidences: np.ndarray

    image_width: int
    image_height: int

    mean_confidence: float
    max_confidence: float
    min_confidence: float


class ShelfSenseDetector:
    """
    ShelfSense Lite inference wrapper.

    Scope
    -----
    - Detect visible retail products.
    - Estimate visible-product count from detections.

    This model does NOT:
    - identify individual SKU identities,
    - recognize brands,
    - estimate hidden inventory,
    - determine planogram compliance.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        max_det: int = DEFAULT_MAX_DET,
        device: Optional[Union[str, int]] = None,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        self.device = device

        self.model = YOLO(str(self.model_path))

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> DetectionResult:

        used_conf = (
            self.conf
            if conf is None
            else float(conf)
        )

        used_iou = (
            self.iou
            if iou is None
            else float(iou)
        )

        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=used_conf,
            iou=used_iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

        if not results:
            raise RuntimeError(
                "Model returned no prediction result."
            )

        result = results[0]

        original = result.orig_img.copy()

        if result.boxes is None:
            boxes = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty(
                (0,),
                dtype=np.float32,
            )

        else:

            boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            confidences = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        count = int(len(boxes))

        annotated = self._draw_detections(
            original,
            boxes,
            confidences,
        )

        h, w = original.shape[:2]

        if count > 0:

            mean_conf = float(
                confidences.mean()
            )

            max_conf = float(
                confidences.max()
            )

            min_conf = float(
                confidences.min()
            )

        else:

            mean_conf = 0.0
            max_conf = 0.0
            min_conf = 0.0

        return DetectionResult(
            image=original,
            annotated_image=annotated,
            count=count,
            boxes=boxes,
            confidences=confidences,
            image_width=w,
            image_height=h,
            mean_confidence=mean_conf,
            max_confidence=max_conf,
            min_confidence=min_conf,
        )

    @staticmethod
    def _draw_detections(
        image: np.ndarray,
        boxes: np.ndarray,
        confidences: np.ndarray,
    ) -> np.ndarray:

        output = image.copy()

        for index, (
            box,
            confidence,
        ) in enumerate(
            zip(
                boxes,
                confidences,
            ),
            start=1,
        ):

            x1, y1, x2, y2 = (
                box.astype(int)
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (40, 220, 90),
                2,
            )

            label = (
                f"Product {index} "
                f"{confidence:.2f}"
            )

            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            font_scale = 0.45
            thickness = 1

            (
                text_width,
                text_height,
            ), baseline = (
                cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    thickness,
                )
            )

            text_y = max(
                y1 - 6,
                text_height + 8,
            )

            cv2.rectangle(
                output,
                (
                    x1,
                    text_y
                    - text_height
                    - 6,
                ),
                (
                    x1
                    + text_width
                    + 6,
                    text_y
                    + baseline,
                ),
                (40, 220, 90),
                -1,
            )

            cv2.putText(
                output,
                label,
                (
                    x1 + 3,
                    text_y - 3,
                ),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        count_text = (
            f"Visible products: "
            f"{len(boxes)}"
        )

        cv2.rectangle(
            output,
            (20, 20),
            (310, 65),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            count_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output