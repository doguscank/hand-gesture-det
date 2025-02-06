import json
from typing import Dict, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor


def load_and_flatten(json_path: str, label_map: Dict) -> pd.DataFrame:
    """
    Load JSON file and flatten keypoints into a DataFrame.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    label_map : dict
        Mapping from sample keys to labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with flattened keypoints and corresponding labels.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    records = []
    for sample, details in data.items():
        record = {}
        for idx, kp in enumerate(details["keypoints"]):
            record[f"kp{idx}_rel_x"] = kp["rel_x"]
            record[f"kp{idx}_rel_y"] = kp["rel_y"]
        record["class"] = label_map.get(sample, None)
        records.append(record)
    return pd.DataFrame(records)


def load_and_combine(json_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Combine multiple JSON files into a single DataFrame with labels and image paths.

    Parameters
    ----------
    json_paths : dict
        Dictionary mapping class names to JSON file paths.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all samples.
    """
    records = []
    for cls, json_path in json_paths.items():
        with open(json_path, "r") as f:
            data = json.load(f)
        for sample, details in data.items():
            record = {}
            for idx, kp in enumerate(details["keypoints"]):
                record[f"kp{idx}_rel_x"] = kp["rel_x"]
                record[f"kp{idx}_rel_y"] = kp["rel_y"]
            record["class"] = cls
            record["img_path"] = sample
            records.append(record)
    return pd.DataFrame(records)


def create_train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified, shuffled train and validation splits from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the complete dataset.

    Returns
    -------
    tuple
        A tuple (df_train, df_val) of shuffled DataFrames for training and validation.
    """
    train_list = []
    val_list = []
    for cls, group in df.groupby("class"):
        train_grp = group.sample(frac=0.8, random_state=42)
        val_grp = group.drop(train_grp.index)
        train_list.append(train_grp.drop(columns=["img_path"]))
        val_list.append(val_grp)
    df_train = (
        pd.concat(train_list).sample(frac=1, random_state=42).reset_index(drop=True)
    )
    df_val = pd.concat(val_list).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_train, df_val


def train_model(
    df_train: pd.DataFrame, df_val: pd.DataFrame, model_path: str
) -> Tuple[TabularPredictor, pd.DataFrame]:
    """
    Train the AutoGluon model using provided training and validation splits,
    and save the model to the given path.

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
    tuple
        A tuple (predictor, df_val) where predictor is the trained model.
    """
    tuning_data = df_val.drop(columns=["img_path"])
    predictor = TabularPredictor(label="class", path=model_path).fit(
        train_data=df_train, tuning_data=tuning_data, presets="medium"
    )
    return predictor, df_val


def predict(predictor: TabularPredictor, sample_json: str) -> pd.Series:
    """
    Predict classes for samples in a given JSON file.

    Parameters
    ----------
    predictor : TabularPredictor
        Trained AutoGluon predictor.
    sample_json : str
        Path to the JSON file with samples.

    Returns
    -------
    pd.Series
        Predictions for each sample.
    """
    df_sample = load_and_combine({"dummy": sample_json})
    df_sample = df_sample.drop(columns=["class"])
    return predictor.predict(df_sample)


def load_model(model_path: str) -> TabularPredictor:
    """
    Load a saved AutoGluon model.

    Parameters
    ----------
    model_path : str
        Path to the saved model.

    Returns
    -------
    TabularPredictor
        Loaded AutoGluon predictor.
    """
    return TabularPredictor.load(model_path)


if __name__ == "__main__":
    json_paths = {
        "five": "/home/doguscank/holonext_ws/data/self/five/bbox_keypoints.json",
        "four": "/home/doguscank/holonext_ws/data/self/four/bbox_keypoints.json",
        "stop": "/home/doguscank/holonext_ws/data/self/stop/bbox_keypoints.json",
        "none": "/home/doguscank/holonext_ws/data/self/none/bbox_keypoints.json",
    }
    df = load_and_combine(json_paths)
    df_train, df_val = create_train_val_split(df)
    model_save_path = "/home/doguscank/holonext_ws/autogluon_model"
    predictor, df_val = train_model(df_train, df_val, model_save_path)
    df_val_features = df_val.drop(columns=["class", "img_path"])
    val_predictions = predictor.predict(df_val_features)
    mapping = pd.DataFrame(
        {"img_path": df_val["img_path"].values, "prediction": val_predictions.values}
    ).to_dict(orient="index")
    with open("/home/doguscank/holonext_ws/autogluon_mapping.json", "w") as f:
        json.dump(mapping, f)
