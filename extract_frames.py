import os
from typing import Dict, List
from pathlib import Path

import cv2


def process_video(
    video_path: str, save_dir: str, save_each_n_frame: int, start_frame: int = 0
) -> int:
    """
    Process a single video file and save frames.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    save_dir : str
        Directory to save the extracted frames.
    save_each_n_frame : int
        Save every nth frame.
    start_frame : int, optional
        Starting frame count, by default 0.

    Returns
    -------
    int
        The final frame count after processing the video.
    """
    video_path = Path(video_path).resolve()
    cap = cv2.VideoCapture(str(video_path))
    frame_count = start_frame
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % save_each_n_frame == 0:
            cv2.imwrite(str(save_dir / f"{frame_count}.jpg"), frame)
        frame_count += 1
    cap.release()
    return frame_count


def extract_frames_from_videos(
    video_paths: Dict[str, List[str]], save_path_template: str, save_each_n_frame: int
) -> None:
    """
    Extract frames from multiple videos based on given paths.

    Parameters
    ----------
    video_paths : Dict[str, List[str]]
        Dictionary mapping labels to lists of video paths.
    save_path_template : str
        A template string for the save directory with a placeholder for the label.
    save_each_n_frame : int
        Save every nth frame.
    """
    for name, paths in video_paths.items():
        frame_count = 0
        for path in paths:
            # Convert each video path to an absolute path
            video_abs = Path(path).resolve()
            save_dir = Path(save_path_template.format(name=name)).resolve()
            frame_count = process_video(str(video_abs), str(save_dir), save_each_n_frame, start_frame=frame_count)


if __name__ == "__main__":
    video_paths: Dict[str, List[str]] = {
        "stop": [
            "./data/stop/VID_20250206_223146.mp4",
            "./data/stop/WIN_20250206_22_29_49_Pro.mp4",
        ],
        "four": [
            "./data/four/VID_20250206_223207.mp4",
            "./data/four/WIN_20250206_22_30_28_Pro.mp4",
        ],
        "five": [
            "./data/five/VID_20250206_223245.mp4",
            "./data/five/WIN_20250206_22_29_03_Pro.mp4",
        ],
        "none": [
            "./data/none/VID_20250206_230620.mp4",
            "./data/none/VID_20250206_230629.mp4",
            "./data/none/VID_20250206_230637.mp4",
            "./data/none/VID_20250206_230647.mp4",
            "./data/none/VID_20250206_230723.mp4",
        ],
    }

    save_path_template: str = "./data/{name}/frames"
    save_each_n_frame: int = 10

    extract_frames_from_videos(video_paths, save_path_template, save_each_n_frame)
