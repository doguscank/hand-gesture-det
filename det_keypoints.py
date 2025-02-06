import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO


@dataclass
class Bbox:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


@dataclass
class Keypoint:
    x: float
    y: float
    rel_x: float
    rel_y: float
    conf: float


@dataclass
class Dataset:
    name: str
    folder_path: str


def init_detector(model_path: str) -> YOLO:
    """
    Initialize the YOLO detector.

    Parameters
    ----------
    model_path : str
        Path to the YOLO model file.

    Returns
    -------
    YOLO
        Initialized YOLO model.
    """
    return YOLO(model_path)


def extract_bbox(result: Any) -> Tuple[Bbox, float, float]:
    """
    Extract bounding box from detection result.

    Parameters
    ----------
    result : Any
        Detection result containing bounding box data.

    Returns
    -------
    Tuple[Bbox, float, float]
        A tuple containing the bounding box dataclass and its x1, y1 coordinates.
    """
    bbox_x1 = float(result.boxes.xyxy[0][0])
    bbox_y1 = float(result.boxes.xyxy[0][1])
    bbox_x2 = float(result.boxes.xyxy[0][2])
    bbox_y2 = float(result.boxes.xyxy[0][3])
    bbox_conf = float(result.boxes.conf[0])
    return Bbox(bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_conf), bbox_x1, bbox_y1


def extract_keypoints(result: Any, bbox_x1: float, bbox_y1: float) -> List[Keypoint]:
    """
    Extract keypoints relative to the bounding box.

    Parameters
    ----------
    result : Any
        Detection result containing keypoints data.
    bbox_x1 : float
        The x-coordinate of the bounding box.
    bbox_y1 : float
        The y-coordinate of the bounding box.

    Returns
    -------
    List[Keypoint]
        List of keypoints with computed relative positions.
    """
    keypoints_list = []
    for (x, y), conf in zip(result.keypoints.xy[0], result.keypoints.conf[0]):
        x = float(x)
        y = float(y)
        conf = float(conf)
        kp = Keypoint(x=x, y=y, rel_x=x - bbox_x1, rel_y=y - bbox_y1, conf=conf)
        keypoints_list.append(kp)
    return keypoints_list


def detect_keypoints(image: Any, model: YOLO) -> Optional[Dict[str, Any]]:
    """
    Detect keypoints in an image using the YOLO model.

    Parameters
    ----------
    image : Any
        Input image or image path for detection.
    model : YOLO
        YOLO model instance.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing bounding box and keypoints data, or None if no detection.
    """
    result = model.predict(image, max_det=1)[0].cpu().numpy()
    if len(result.boxes.xyxy) == 0:
        return None
    bbox_result, bbox_x1, bbox_y1 = extract_bbox(result)
    keypoints_list = extract_keypoints(result, bbox_x1, bbox_y1)
    return {
        "bbox": asdict(bbox_result),
        "keypoints": [asdict(kp) for kp in keypoints_list],
    }


def process_folder(name: str, folder_path: str, model: YOLO) -> Dict[str, Any]:
    """
    Process a folder of images to detect keypoints.

    Parameters
    ----------
    name : str
        Name identifier for the dataset.
    folder_path : str
        Path to the folder containing images.
    model : YOLO
        YOLO model instance.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping image identifiers to detection results.
    """
    data = {}
    for file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file)
        result = detect_keypoints(image_path, model)
        if result is None:
            continue
        data[f"{name}-{file}"] = result
    return data


def save_results(name: str, data: Dict[str, Any], output_dir: str) -> None:
    """
    Save detection results to a JSON file.

    Parameters
    ----------
    name : str
        Dataset name.
    data : Dict[str, Any]
        Detection results data.
    output_dir : str
        Directory where the output file will be saved.
    """
    output_file = f"{output_dir}/{name}/bbox_keypoints.json"
    with open(output_file, "w") as f:
        json.dump(data, f)
