import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from datetime import datetime

SUPPORTED_EXTS = (".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp", ".tif", ".tiff")
CLASS_NAMES = ("fake", "real")


def parse_args():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    dataset_root = os.path.join(project_root, "dataset")

    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic stratified train/valid/test folders with manifests. "
            "Default source is existing dataset splits."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=dataset_root,
        help="Dataset root containing source folders.",
    )
    parser.add_argument(
        "--source-mode",
        choices=["existing-splits", "raw-classes"],
        default="existing-splits",
        help=(
            "existing-splits: source from dataset/train|valid|test/{fake,real}; "
            "raw-classes: source from dataset/{fake,real}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output split root. Default: <dataset-root>/strict_split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split.")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Validation ratio in [0,1).")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test ratio in [0,1).")
    parser.add_argument(
        "--valid-count-per-class",
        type=int,
        default=None,
        help="If set, use this exact validation count per class (overrides --valid-ratio).",
    )
    parser.add_argument(
        "--test-count-per-class",
        type=int,
        default=None,
        help="If set, use this exact test count per class (overrides --test-ratio).",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="How to materialize files into split folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory first if it already exists.",
    )
    return parser.parse_args()


def collect_sources(dataset_root, source_mode):
    sources = {"fake": [], "real": []}

    if source_mode == "raw-classes":
        for cls in CLASS_NAMES:
            class_dir = os.path.join(dataset_root, cls)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Expected source class folder: {class_dir}")
            sources[cls].append(class_dir)
    else:
        found_any = False
        for split in ("train", "valid", "test"):
            split_dir = os.path.join(dataset_root, split)
            if not os.path.isdir(split_dir):
                continue
            for cls in CLASS_NAMES:
                class_dir = os.path.join(split_dir, cls)
                if os.path.isdir(class_dir):
                    found_any = True
                    sources[cls].append(class_dir)
        if not found_any:
            raise FileNotFoundError(
                "No existing split source directories found under dataset/{train,valid,test}/{fake,real}."
            )

    return sources


def collect_images(class_dirs):
    paths = []
    for class_dir in class_dirs:
        for root, _, files in os.walk(class_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in SUPPORTED_EXTS:
                    paths.append(os.path.abspath(os.path.join(root, name)))

    deduped = sorted(set(paths))
    return deduped


def stratified_split(
    paths_by_class,
    valid_ratio,
    test_ratio,
    seed,
    valid_count_per_class=None,
    test_count_per_class=None,
):
    if valid_ratio < 0 or test_ratio < 0 or (valid_ratio + test_ratio) >= 1.0:
        raise ValueError("valid_ratio and test_ratio must be >=0 and sum to < 1.0")
    if valid_count_per_class is not None and valid_count_per_class < 0:
        raise ValueError("valid_count_per_class must be >= 0")
    if test_count_per_class is not None and test_count_per_class < 0:
        raise ValueError("test_count_per_class must be >= 0")

    rng = random.Random(seed)
    split_map = {"train": {"fake": [], "real": []}, "valid": {"fake": [], "real": []}, "test": {"fake": [], "real": []}}

    for cls, paths in paths_by_class.items():
        local = list(paths)
        rng.shuffle(local)
        n = len(local)

        if test_count_per_class is None:
            n_test = int(round(n * test_ratio))
        else:
            n_test = int(test_count_per_class)

        if valid_count_per_class is None:
            n_valid = int(round(n * valid_ratio))
        else:
            n_valid = int(valid_count_per_class)

        if n_test + n_valid >= n:
            raise RuntimeError(
                f"Requested split sizes exceed available samples for class '{cls}': "
                f"n={n}, valid={n_valid}, test={n_test}."
            )

        n_train = n - n_valid - n_test

        if n_train <= 0:
            raise RuntimeError(
                f"Not enough samples for class '{cls}' with ratios valid={valid_ratio}, test={test_ratio}."
            )

        split_map["test"][cls] = local[:n_test]
        split_map["valid"][cls] = local[n_test:n_test + n_valid]
        split_map["train"][cls] = local[n_test + n_valid:]

    return split_map


def ensure_clean_dir(path, overwrite):
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def safe_name(src_path, index):
    base = os.path.basename(src_path)
    stem, ext = os.path.splitext(base)
    digest = hashlib.sha1(src_path.encode("utf-8")).hexdigest()[:10]
    return f"{index:06d}_{stem}_{digest}{ext.lower()}"


def materialize(split_map, output_root, link_mode):
    manifests_dir = os.path.join(output_root, "manifests")
    os.makedirs(manifests_dir, exist_ok=True)

    summary = {"train": {}, "valid": {}, "test": {}}

    for split in ("train", "valid", "test"):
        rows = []
        for cls in CLASS_NAMES:
            dst_class_dir = os.path.join(output_root, split, cls)
            os.makedirs(dst_class_dir, exist_ok=True)

            class_paths = split_map[split][cls]
            for idx, src in enumerate(class_paths, start=1):
                dst_name = safe_name(src, idx)
                dst = os.path.join(dst_class_dir, dst_name)
                if link_mode == "hardlink":
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)

                rows.append(
                    {
                        "split": split,
                        "label": 0 if cls == "fake" else 1,
                        "label_str": cls,
                        "src_path": src,
                        "dst_path": dst,
                    }
                )

            summary[split][cls] = len(class_paths)

        manifest_path = os.path.join(manifests_dir, f"{split}.csv")
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split", "label", "label_str", "src_path", "dst_path"],
            )
            writer.writeheader()
            writer.writerows(rows)

    return summary


def main():
    args = parse_args()
    dataset_root = os.path.abspath(args.dataset_root)
    output_root = os.path.abspath(args.output_dir or os.path.join(dataset_root, "strict_split"))

    sources = collect_sources(dataset_root, args.source_mode)
    paths_by_class = {
        cls: collect_images(sources[cls])
        for cls in CLASS_NAMES
    }

    for cls in CLASS_NAMES:
        if len(paths_by_class[cls]) == 0:
            raise RuntimeError(f"No usable images found for class '{cls}'.")

    split_map = stratified_split(
        paths_by_class=paths_by_class,
        valid_ratio=float(args.valid_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        valid_count_per_class=args.valid_count_per_class,
        test_count_per_class=args.test_count_per_class,
    )

    ensure_clean_dir(output_root, overwrite=bool(args.overwrite))
    summary = materialize(split_map, output_root=output_root, link_mode=args.link_mode)

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset_root": dataset_root,
        "output_root": output_root,
        "source_mode": args.source_mode,
        "seed": int(args.seed),
        "valid_ratio": float(args.valid_ratio),
        "test_ratio": float(args.test_ratio),
        "valid_count_per_class": args.valid_count_per_class,
        "test_count_per_class": args.test_count_per_class,
        "link_mode": args.link_mode,
        "source_counts": {cls: len(paths_by_class[cls]) for cls in CLASS_NAMES},
        "split_counts": summary,
    }

    metadata_path = os.path.join(output_root, "manifests", "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Created strict stratified split at:", output_root)
    print("Source counts:", metadata["source_counts"])
    print("Split counts:", summary)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
