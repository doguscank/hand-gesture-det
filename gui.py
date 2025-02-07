import cv2
import gradio as gr
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from pathlib import Path

from det_keypoints import detect_keypoints, init_detector
from utils import build_features_from_keypoints

# Load models using fixed paths
YOLO_WEIGHTS = Path("./weights/yolo-best.pt").resolve()
AUTOGLUON_MODEL_PATH = Path("./weights/autogluon_model").resolve()

yolo_model_instance = init_detector(str(YOLO_WEIGHTS))
predictor = TabularPredictor.load(str(AUTOGLUON_MODEL_PATH))


def predict(image):
    """Predicts hand gesture class from an image."""
    if image is None:
        return "Error: No image provided."

    # Ensure image is a NumPy array
    if not isinstance(image, np.ndarray):
        return "Error: Invalid image format."

    # Ensure image has valid dimensions
    if len(image.shape) != 3 or image.shape[2] != 3:
        return "Error: Image must be a 3-channel color image."

    try:
        # Convert image from RGB to BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detection = detect_keypoints(img_bgr, yolo_model_instance)
        if detection is None or not detection.get("keypoints"):
            return "Error: No keypoints detected."

        features = build_features_from_keypoints(detection["keypoints"])
        df = pd.DataFrame([features])
        pred = predictor.predict(df)
        return f"Predicted Class: {pred.iloc[0]}"

    except Exception as e:
        return f"Error processing image: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("## Hand Gesture Detection")

    with gr.Tabs():
        with gr.Tab("Upload Image"):
            img_upload = gr.Image(
                label="Upload Image",
                type="numpy",
                sources=["upload"],
                format="png",
            )
            btn_upload = gr.Button("Predict")
            out_upload = gr.Textbox(label="Prediction")
            btn_upload.click(predict, inputs=img_upload, outputs=out_upload)

        with gr.Tab("Webcam"):
            img_webcam = gr.Image(
                label="Webcam Feed",
                type="numpy",
                source="webcam",
                streaming=True,
            )
            out_webcam = gr.Textbox(label="Prediction")
            img_webcam.stream(
                predict,
                inputs=img_webcam,
                outputs=out_webcam,
                time_limit=30,
                stream_every=3,
            )

    gr.Markdown("### Ensure your weights are available at the specified paths.")

demo.launch(share=True)
