import argparse

import pandas as pd
from autogluon.tabular import TabularPredictor

from utils import create_train_val_split, load_and_combine


def train_model(
    df_train: pd.DataFrame, df_val: pd.DataFrame, model_path: str
) -> TabularPredictor:
    """
    Train the AutoGluon model using provided training and validation splits and save it.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data.
    df_val : pd.DataFrame
        Validation data.
    model_path : str
        Directory path to save the trained model.

    Returns
    -------
    TabularPredictor
        The trained AutoGluon predictor.
    """
    tuning_data = df_val.drop(columns=["img_path"])
    predictor = TabularPredictor(label="class", path=model_path).fit(
        train_data=df_train, tuning_data=tuning_data, presets="medium"
    )
    return predictor


def main() -> None:
    """
    Combine JSON data, split it into training and validation sets, train the model,
    and display the model save path.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Train the AutoGluon model for hand gesture detection."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./weights/autogluon_model",
        help="Directory path to save the trained model.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for the dataset JSON files.",
    )
    args = parser.parse_args()

    json_paths = {
        "five": f"{args.data_root}/five/bbox_keypoints.json",
        "four": f"{args.data_root}/four/bbox_keypoints.json",
        "stop": f"{args.data_root}/stop/bbox_keypoints.json",
        "none": f"{args.data_root}/none/bbox_keypoints.json",
    }
    df = load_and_combine(json_paths)
    df_train, df_val = create_train_val_split(df)
    model_save_path = args.model_path
    predictor: TabularPredictor = train_model(df_train, df_val, model_save_path)
    print("Model trained and saved at:", model_save_path)


if __name__ == "__main__":
    main()
