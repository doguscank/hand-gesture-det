import argparse  # added argparse import
import json

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (  # updated metrics imports
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
    # Argument parser added
    parser = argparse.ArgumentParser(
        description="Validate the AutoGluon model for hand gesture detection."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./weights/autogluon_model",
        help="Directory path of the saved model.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for the dataset JSON files.",
    )
    parser.add_argument(
        "--mapping_out",
        type=str,
        default="./autogluon_mapping.json",
        help="Output path for the validation mapping JSON file.",
    )
    args = parser.parse_args()

    json_paths = {
        "five": f"{args.data_root}/five/bbox_keypoints.json",
        "four": f"{args.data_root}/four/bbox_keypoints.json",
        "stop": f"{args.data_root}/stop/bbox_keypoints.json",
        "none": f"{args.data_root}/none/bbox_keypoints.json",
    }
    df = load_and_combine(json_paths)

    # Obtain validation split
    _, df_val = create_train_val_split(df)
    model_path = args.model_path
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

    mapping_out = args.mapping_out
    with open(mapping_out, "w") as f:
        json.dump(mapping, f)
    print("Validation mapping saved at:", mapping_out)


if __name__ == "__main__":
    main()
