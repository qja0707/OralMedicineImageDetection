import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def list_split_images(split_dir):
    split_dir = Path(split_dir)
    return sorted(
        path
        for path in split_dir.glob("*/*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalize_names(names):
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_yolo_classification(
    model_path,
    data_dir,
    split="val",
    output_dir="ml/outputs/classification_eval",
    imgsz=224,
    device=None,
    topk=5,
):
    from ultralytics import YOLO

    model_path = Path(model_path)
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    image_paths = list_split_images(split_dir)
    if not image_paths:
        raise ValueError(f"No images found: {split_dir}")

    model = YOLO(str(model_path))
    names = normalize_names(model.names)

    prediction_rows = []
    confusion = Counter()
    support = Counter()
    predicted_count = Counter()
    correct_count = Counter()
    topk_correct_count = Counter()

    for image_path in image_paths:
        true_class = image_path.parent.name
        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]

        probs = result.probs
        top1_index = int(probs.top1)
        top1_class = names.get(top1_index, str(top1_index))
        top1_conf = float(probs.top1conf)

        top_indices = [int(index) for index in probs.top5[:topk]]
        top_classes = [names.get(index, str(index)) for index in top_indices]
        top_confs = [float(probs.data[index]) for index in top_indices]

        is_correct = true_class == top1_class
        is_topk_correct = true_class in top_classes

        support[true_class] += 1
        predicted_count[top1_class] += 1
        if is_correct:
            correct_count[true_class] += 1
        if is_topk_correct:
            topk_correct_count[true_class] += 1
        confusion[(true_class, top1_class)] += 1

        prediction_rows.append(
            {
                "image_path": str(image_path),
                "true_class": true_class,
                "pred_class": top1_class,
                "top1_conf": f"{top1_conf:.6f}",
                "correct": int(is_correct),
                f"top{topk}_correct": int(is_topk_correct),
                f"top{topk}_classes": "|".join(top_classes),
                f"top{topk}_confs": "|".join(f"{conf:.6f}" for conf in top_confs),
            }
        )

    classes = sorted(set(support) | set(predicted_count))
    per_class_rows = []
    for class_name in classes:
        tp = correct_count[class_name]
        fp = predicted_count[class_name] - tp
        fn = support[class_name] - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        top1_acc = safe_div(tp, support[class_name])
        topk_acc = safe_div(topk_correct_count[class_name], support[class_name])

        per_class_rows.append(
            {
                "class_name": class_name,
                "support": support[class_name],
                "predicted": predicted_count[class_name],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": f"{precision:.6f}",
                "recall": f"{recall:.6f}",
                "f1": f"{f1:.6f}",
                "top1_acc": f"{top1_acc:.6f}",
                f"top{topk}_acc": f"{topk_acc:.6f}",
            }
        )

    worst_class_rows = sorted(
        per_class_rows,
        key=lambda row: (
            float(row["recall"]),
            float(row["f1"]),
            float(row["precision"]),
            -int(row["support"]),
        ),
    )
    misclassified_rows = [
        row for row in prediction_rows if int(row["correct"]) == 0
    ]
    confusion_pair_rows = []
    for (true_class, pred_class), count in confusion.most_common():
        if true_class == pred_class:
            continue
        confusion_pair_rows.append(
            {
                "true_class": true_class,
                "pred_class": pred_class,
                "count": count,
                "true_support": support[true_class],
                "rate_in_true_class": f"{safe_div(count, support[true_class]):.6f}",
            }
        )

    confusion_rows = []
    for true_class in classes:
        row = {"true_class": true_class}
        for pred_class in classes:
            row[pred_class] = confusion[(true_class, pred_class)]
        confusion_rows.append(row)

    total = len(image_paths)
    top1_correct = sum(correct_count.values())
    topk_correct = sum(topk_correct_count.values())
    summary = {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "split": split,
        "num_images": total,
        "num_classes": len(classes),
        "top1_acc": safe_div(top1_correct, total),
        f"top{topk}_acc": safe_div(topk_correct, total),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{split}_predictions.csv"
    misclassified_path = output_dir / f"{split}_misclassified.csv"
    per_class_path = output_dir / f"{split}_per_class_metrics.csv"
    worst_classes_path = output_dir / f"{split}_worst_classes.csv"
    confusion_path = output_dir / f"{split}_confusion_matrix.csv"
    confusion_pairs_path = output_dir / f"{split}_confusion_pairs.csv"
    summary_path = output_dir / f"{split}_summary.json"

    write_csv(
        predictions_path,
        prediction_rows,
        [
            "image_path",
            "true_class",
            "pred_class",
            "top1_conf",
            "correct",
            f"top{topk}_correct",
            f"top{topk}_classes",
            f"top{topk}_confs",
        ],
    )
    write_csv(
        per_class_path,
        per_class_rows,
        [
            "class_name",
            "support",
            "predicted",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "top1_acc",
            f"top{topk}_acc",
        ],
    )
    write_csv(
        worst_classes_path,
        worst_class_rows,
        [
            "class_name",
            "support",
            "predicted",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "top1_acc",
            f"top{topk}_acc",
        ],
    )
    write_csv(
        misclassified_path,
        misclassified_rows,
        [
            "image_path",
            "true_class",
            "pred_class",
            "top1_conf",
            "correct",
            f"top{topk}_correct",
            f"top{topk}_classes",
            f"top{topk}_confs",
        ],
    )
    write_csv(confusion_path, confusion_rows, ["true_class", *classes])
    write_csv(
        confusion_pairs_path,
        confusion_pair_rows,
        [
            "true_class",
            "pred_class",
            "count",
            "true_support",
            "rate_in_true_class",
        ],
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print("YOLO classification 평가 완료")
    print("=" * 60)
    print(f"  Model             : {model_path}")
    print(f"  Dataset split     : {split_dir}")
    print(f"  Images            : {total}")
    print(f"  Classes           : {len(classes)}")
    print(f"  Top-1 acc         : {summary['top1_acc']:.4f}")
    print(f"  Top-{topk} acc     : {summary[f'top{topk}_acc']:.4f}")
    print(f"  Predictions CSV   : {predictions_path}")
    print(f"  Misclassified CSV : {misclassified_path}")
    print(f"  Per-class CSV     : {per_class_path}")
    print(f"  Worst classes CSV : {worst_classes_path}")
    print(f"  Confusion CSV     : {confusion_path}")
    print(f"  Confusion pairs   : {confusion_pairs_path}")
    print(f"  Summary JSON      : {summary_path}")

    return {
        "summary": summary,
        "predictions_path": predictions_path,
        "misclassified_path": misclassified_path,
        "per_class_path": per_class_path,
        "worst_classes_path": worst_classes_path,
        "confusion_path": confusion_path,
        "confusion_pairs_path": confusion_pairs_path,
        "summary_path": summary_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO classification model per class.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="ml/outputs/classification_eval")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    evaluate_yolo_classification(
        model_path=args.model,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        device=args.device,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
