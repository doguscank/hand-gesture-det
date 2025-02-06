import os
from typing import Dict, List

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
    cap = cv2.VideoCapture(video_path)
    frame_count = start_frame
    os.makedirs(save_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % save_each_n_frame == 0:
            cv2.imwrite(os.path.join(save_dir, f"{frame_count}.jpg"), frame)
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
            # Format the save directory for this label
            save_dir = save_path_template.format(name=name)
            frame_count = process_video(
                path, save_dir, save_each_n_frame, start_frame=frame_count
            )


if __name__ == "__main__":
    video_paths: Dict[str, List[str]] = {
        "stop": [
            "/home/doguscank/holonext_ws/data/self/stop/VID_20250206_223146.mp4",
            "/home/doguscank/holonext_ws/data/self/stop/WIN_20250206_22_29_49_Pro.mp4",
        ],
        "four": [
            "/home/doguscank/holonext_ws/data/self/four/VID_20250206_223207.mp4",
            "/home/doguscank/holonext_ws/data/self/four/WIN_20250206_22_30_28_Pro.mp4",
        ],
        "five": [
            "/home/doguscank/holonext_ws/data/self/five/VID_20250206_223245.mp4",
            "/home/doguscank/holonext_ws/data/self/five/WIN_20250206_22_29_03_Pro.mp4",
        ],
        "none": [
            "/home/doguscank/holonext_ws/data/self/none/VID_20250206_230620.mp4",
            "/home/doguscank/holonext_ws/data/self/none/VID_20250206_230629.mp4",
            "/home/doguscank/holonext_ws/data/self/none/VID_20250206_230637.mp4",
            "/home/doguscank/holonext_ws/data/self/none/VID_20250206_230647.mp4",
            "/home/doguscank/holonext_ws/data/self/none/VID_20250206_230723.mp4",
        ],
    }

    save_path_template: str = "/home/doguscank/holonext_ws/data/self/{name}/frames"
    save_each_n_frame: int = 10

    extract_frames_from_videos(video_paths, save_path_template, save_each_n_frame)
