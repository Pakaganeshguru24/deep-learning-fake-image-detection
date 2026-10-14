import argparse
import os
import subprocess
import sys


def run_step(cmd, cwd):
    print("\n>>>", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed with exit code {result.returncode}: {' '.join(cmd)}")


def parse_args():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    dataset_root = os.path.join(project_root, "dataset")

    parser = argparse.ArgumentParser(
        description="Run strict reproducible split -> train -> tune -> eval pipeline."
    )
    parser.add_argument("--dataset-root", default=dataset_root)
    parser.add_argument("--strict-split-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=300)
    parser.add_argument("--epochs-phase1", type=int, default=10)
    parser.add_argument("--epochs-phase2", type=int, default=6)
    parser.add_argument(
        "--source-mode",
        choices=["existing-splits", "raw-classes"],
        default="existing-splits",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
    )
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--model-path", default=os.path.join(app_dir, "models", "best_model.keras"))
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    app_dir = os.path.dirname(os.path.abspath(__file__))
    strict_split_dir = os.path.abspath(
        args.strict_split_dir or os.path.join(args.dataset_root, "strict_split")
    )
    model_path = os.path.abspath(args.model_path)
    threshold_json = os.path.join(os.path.dirname(model_path), "threshold.json")

    py = sys.executable

    if not args.skip_split:
        split_cmd = [
            py,
            os.path.join(app_dir, "build_stratified_splits.py"),
            "--dataset-root",
            os.path.abspath(args.dataset_root),
            "--source-mode",
            args.source_mode,
            "--output-dir",
            strict_split_dir,
            "--seed",
            str(args.seed),
            "--valid-ratio",
            str(args.valid_ratio),
            "--test-ratio",
            str(args.test_ratio),
            "--link-mode",
            args.link_mode,
        ]
        if args.overwrite_split:
            split_cmd.append("--overwrite")
        run_step(split_cmd, cwd=app_dir)

    if not args.skip_train:
        train_cmd = [
            py,
            os.path.join(app_dir, "train.py"),
            "--dataset-root",
            strict_split_dir,
            "--model-path",
            model_path,
            "--seed",
            str(args.seed),
            "--img-size",
            str(args.img_size),
            "--batch-size",
            str(args.batch_size),
            "--epochs-phase1",
            str(args.epochs_phase1),
            "--epochs-phase2",
            str(args.epochs_phase2),
        ]
        run_step(train_cmd, cwd=app_dir)

    if not args.skip_tune:
        tune_cmd = [
            py,
            os.path.join(app_dir, "tune_threshold.py"),
            "--dataset-root",
            strict_split_dir,
            "--model-path",
            model_path,
            "--batch-size",
            str(max(16, args.batch_size)),
            "--output-json",
            threshold_json,
        ]
        run_step(tune_cmd, cwd=app_dir)

    if not args.skip_test_eval:
        eval_cmd = [
            py,
            os.path.join(app_dir, "eval_valid.py"),
            "--dataset-root",
            strict_split_dir,
            "--split",
            "test",
            "--model-path",
            model_path,
            "--batch-size",
            str(max(16, args.batch_size)),
            "--progress-every",
            "100",
            "--threshold-config",
            threshold_json,
        ]
        run_step(eval_cmd, cwd=app_dir)

    print("\nStrict pipeline completed successfully.")
    print("Strict split root:", strict_split_dir)
    print("Model path:", model_path)
    print("Threshold artifact:", threshold_json)


if __name__ == "__main__":
    main()
