import argparse
import json
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

from utils import create_train_val_split, load_and_combine, load_model


def main() -> None:
    """
    Load dataset, obtain a validation split, load a saved model, perform predictions,
    and write an image-path-to-prediction mapping.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Validate the AutoGluon model for hand gesture detection."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./weights/autogluon_model",
        help="Directory path of the saved model.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for the dataset JSON files.",
    )
    parser.add_argument(
        "--mapping-out",
        type=str,
        default="./autogluon_mapping.json",
        help="Output path for the validation mapping JSON file.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    data_root = Path(args.data_root).resolve()
    mapping_out = Path(args.mapping_out).resolve()

    json_paths = {
        "five": (data_root / "five" / "bbox_keypoints.json").as_posix(),
        "four": (data_root / "four" / "bbox_keypoints.json").as_posix(),
        "stop": (data_root / "stop" / "bbox_keypoints.json").as_posix(),
        "none": (data_root / "none" / "bbox_keypoints.json").as_posix(),
    }
    df = load_and_combine(json_paths)

    # Obtain validation split
    _, df_val = create_train_val_split(df)
    predictor: TabularPredictor = load_model(model_path)
    df_val_features = df_val.drop(columns=["class", "img_path"])
    val_predictions = predictor.predict(df_val_features)
    mapping = pd.DataFrame(
        {"img_path": df_val["img_path"].values, "prediction": val_predictions.values}
    ).to_dict(orient="index")

    # Calculate validation metrics
    true_labels = df_val["class"]
    acc = accuracy_score(true_labels, val_predictions)
    report = classification_report(true_labels, val_predictions)
    print("Validation Accuracy:", acc)

    cm = confusion_matrix(true_labels, val_predictions)
    precision = precision_score(true_labels, val_predictions, average="macro")
    recall = recall_score(true_labels, val_predictions, average="macro")
    print("Confusion Matrix:")
    print(cm)
    print("Macro Precision:", precision)
    print("Macro Recall:", recall)

    print("Classification Report:")
    print(report)

    with mapping_out.open("w") as f:
        json.dump(mapping, f)
    print("Validation mapping saved at:", mapping_out.as_posix())


if __name__ == "__main__":
    main()
