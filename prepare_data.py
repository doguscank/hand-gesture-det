import argparse  # Added for argument parsing
from pathlib import Path

from det_keypoints import Dataset, init_detector, process_folder, save_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for hand gesture detection."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data",
        help="Directory containing the datasets",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="./weights/yolo-best.pt",
        help="Path to the detector weights file",
    )
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    yolo_weights = Path(args.yolo_weights).resolve()

    datasets = [
        Dataset("five", (dataset_dir / "five" / "frames").as_posix()),
        Dataset("four", (dataset_dir / "four" / "frames").as_posix()),
        Dataset("stop", (dataset_dir / "stop" / "frames").as_posix()),
        Dataset("none", (dataset_dir / "none" / "frames").as_posix()),
    ]

    model = init_detector(str(yolo_weights))

    for ds in datasets:
        results = process_folder(ds.name, ds.folder_path, model)
        save_results(ds.name, results, dataset_dir.as_posix())
