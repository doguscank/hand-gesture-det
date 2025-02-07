import argparse
from pathlib import Path

import cv2
import pandas as pd
from autogluon.tabular import TabularPredictor

from det_keypoints import detect_keypoints, init_detector
from utils import build_features_from_keypoints


def main() -> None:
    """
    Entry point for keypoint detection and gesture prediction.

    This function parses the command-line arguments, reads the input image, performs
    keypoint detection using a YOLO model, builds a feature DataFrame, and predicts
    the gesture class using an AutoGluon model.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Predict hand gesture from image using YOLO and AutoGluon"
    )
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--yolo-model", type=str, help="Path to the YOLO model file")
    parser.add_argument(
        "--autogluon-model", type=str, help="Path to the saved AutoGluon model"
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    # Load image using string path
    image = cv2.imread(str(image_path))
    if image is None:
        print("Error: Unable to read the image.")
        return

    yolo_model_path = Path(args.yolo_model).resolve()
    # Load YOLO model using absolute path string
    yolo = init_detector(str(yolo_model_path))
    detection = detect_keypoints(image, yolo)
    if detection is None or not detection.get("keypoints"):
        print("Error: No keypoints detected.")
        return

    # Build feature DataFrame
    features = build_features_from_keypoints(detection["keypoints"])
    df = pd.DataFrame([features])

    autogluon_model_path = Path(args.autogluon_model).resolve()
    # Load AutoGluon model and predict class
    predictor: TabularPredictor = TabularPredictor.load(str(autogluon_model_path))
    prediction = predictor.predict(df)
    print("Predicted Class:", prediction.iloc[0])


if __name__ == "__main__":
    main()
