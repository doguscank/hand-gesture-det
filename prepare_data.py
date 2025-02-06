import argparse  # Added for argument parsing

from det_keypoints import (
    Dataset,
    init_detector,
    process_folder,
    save_results,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for hand gesture detection."
    )
    parser.add_argument(
        "--dataset_dir",
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

    datasets = [
        Dataset("five", f"{args.dataset_dir}/five/frames"),
        Dataset("four", f"{args.dataset_dir}/four/frames"),
        Dataset("stop", f"{args.dataset_dir}/stop/frames"),
        Dataset("none", f"{args.dataset_dir}/none/frames"),
    ]

    model = init_detector(args.yolo_weights)

    for ds in datasets:
        results = process_folder(ds.name, ds.folder_path, model)
        save_results(ds.name, results, args.dataset_dir)
