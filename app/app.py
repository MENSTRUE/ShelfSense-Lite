from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.inference import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    ShelfSenseDetector,
)


st.set_page_config(
    page_title="ShelfSense Lite",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource
def load_detector():
    return ShelfSenseDetector()


def pil_to_bgr(
    image: Image.Image,
) -> np.ndarray:

    rgb = np.array(
        image.convert("RGB")
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )


def bgr_to_rgb(
    image: np.ndarray,
) -> np.ndarray:

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def main():

    st.title(
        "ShelfSense Lite"
    )

    st.caption(
        "Dense retail-product detection "
        "and visible-product counting."
    )

    st.info(
        "ShelfSense Lite detects visible "
        "retail products in shelf images. "
        "The resulting count represents "
        "visible detections, not total "
        "or hidden inventory."
    )

    try:

        detector = load_detector()

    except Exception as exc:

        st.error(
            "Failed to load the model."
        )

        st.exception(exc)

        st.stop()

    with st.sidebar:

        st.header(
            "Inference Settings"
        )

        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.90,
            value=float(DEFAULT_CONF),
            step=0.05,
            help=(
                "0.30 was selected using "
                "validation counting MAE."
            ),
        )

        iou = st.slider(
            "IoU threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(DEFAULT_IOU),
            step=0.05,
        )

        st.divider()

        st.subheader(
            "Model"
        )

        st.write(
            "**YOLO11s**"
        )

        st.write(
            f"Input size: `{DEFAULT_IMGSZ}`"
        )

        st.write(
            f"Max detections: `{DEFAULT_MAX_DET}`"
        )

        st.write(
            "Task: `single-class detection`"
        )

        st.divider()

        st.caption(
            "Recommended confidence: 0.30"
        )

    uploaded_file = (
        st.file_uploader(
            "Upload a retail shelf image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
        )
    )

    if uploaded_file is None:

        st.markdown(
            """
            ### Try ShelfSense

            Upload an image containing
            visible retail products.

            The model will:

            - localize visible products,
            - display confidence scores,
            - estimate the visible-product count.
            """
        )

        return

    try:

        pil_image = Image.open(
            uploaded_file
        )

        image_bgr = pil_to_bgr(
            pil_image
        )

    except Exception as exc:

        st.error(
            "Unable to read the uploaded image."
        )

        st.exception(exc)

        return

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Original"
        )

        st.image(
            pil_image,
            use_container_width=True,
        )

    with st.spinner(
        "Detecting visible products..."
    ):

        try:

            result = detector.predict(
                image_bgr,
                conf=confidence,
                iou=iou,
            )

        except Exception as exc:

            st.error(
                "Inference failed."
            )

            st.exception(exc)

            return

    with right:

        st.subheader(
            "ShelfSense Detection"
        )

        st.image(
            bgr_to_rgb(
                result.annotated_image
            ),
            use_container_width=True,
        )

    st.divider()

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    metric1.metric(
        "Visible Products",
        result.count,
    )

    metric2.metric(
        "Mean Confidence",
        f"{result.mean_confidence:.1%}",
    )

    metric3.metric(
        "Highest Confidence",
        f"{result.max_confidence:.1%}",
    )

    metric4.metric(
        "Detection Threshold",
        f"{confidence:.2f}",
    )

    st.divider()

    st.subheader(
        "Detection Details"
    )

    if result.count == 0:

        st.warning(
            "No visible product detections "
            "passed the selected threshold."
        )

    else:

        rows = []

        for index, (
            box,
            score,
        ) in enumerate(
            zip(
                result.boxes,
                result.confidences,
            ),
            start=1,
        ):

            x1, y1, x2, y2 = (
                box.tolist()
            )

            rows.append(
                {
                    "Detection": index,
                    "Confidence": round(
                        float(score),
                        4,
                    ),
                    "x1": round(
                        x1,
                        1,
                    ),
                    "y1": round(
                        y1,
                        1,
                    ),
                    "x2": round(
                        x2,
                        1,
                    ),
                    "y2": round(
                        y2,
                        1,
                    ),
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.markdown(
        """
        ### Interpretation

        **Visible Products** represents the number of
        model detections that pass the selected confidence
        threshold.

        This value should **not** be interpreted as:

        - total store inventory,
        - hidden or occluded inventory,
        - individual SKU identity,
        - brand recognition,
        - planogram compliance.

        ShelfSense Lite is currently an MVP for
        dense retail-product detection and
        detection-derived visible-product counting.
        """
    )


if __name__ == "__main__":
    main()