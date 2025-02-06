import json
from typing import Any, Dict, List, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor


def load_and_flatten(json_path: str, label_map: Dict[str, str]) -> pd.DataFrame:
    """
    Load a JSON file and flatten keypoints into a DataFrame.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    label_map : dict of {str: str}
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
    json_paths : dict of {str: str}
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
    tuple of (pd.DataFrame, pd.DataFrame)
        A tuple (df_train, df_val) containing the training and validation DataFrames.
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


def load_model(model_path: str) -> TabularPredictor:
    """
    Load a saved AutoGluon model.

    Parameters
    ----------
    model_path : str
        Path to the saved model directory.

    Returns
    -------
    TabularPredictor
        Loaded AutoGluon predictor.
    """
    return TabularPredictor.load(model_path)


def build_features_from_keypoints(keypoints: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Build feature dictionary from keypoints.

    Parameters
    ----------
    keypoints : List[Dict[str, Any]]
        List of keypoint dictionaries. Each dictionary is expected to have keys
        'rel_x' and 'rel_y' representing normalized coordinates.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys formatted as "kp{index}_rel_x" and "kp{index}_rel_y".
    """
    features = {}
    for idx, kp in enumerate(keypoints):
        features[f"kp{idx}_rel_x"] = kp.get("rel_x", 0)
        features[f"kp{idx}_rel_y"] = kp.get("rel_y", 0)
    return features
