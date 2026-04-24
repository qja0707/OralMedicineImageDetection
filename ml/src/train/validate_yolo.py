from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _safe_getattr(obj, attr_name, default=None):
    return getattr(obj, attr_name, default)


def build_per_class_metrics(metrics, class_names):
    box_metrics = _safe_getattr(metrics, "box")
    if box_metrics is None:
        raise AttributeError("Ultralytics validation result does not contain `box` metrics.")

    maps = _to_list(_safe_getattr(box_metrics, "maps"))
    precision = _to_list(_safe_getattr(box_metrics, "p"))
    recall = _to_list(_safe_getattr(box_metrics, "r"))
    f1_scores = _to_list(_safe_getattr(box_metrics, "f1"))
    ap_class_index = _to_list(_safe_getattr(box_metrics, "ap_class_index"))

    num_classes = len(class_names)
    if not ap_class_index:
        ap_class_index = list(range(num_classes))

    rows = []
    for yolo_class_id in range(num_classes):
        class_name = class_names.get(yolo_class_id, str(yolo_class_id))
        metric_index = (
            ap_class_index.index(yolo_class_id)
            if yolo_class_id in ap_class_index
            else yolo_class_id
        )

        rows.append(
            {
                "yolo_class_id": yolo_class_id,
                "class_name": class_name,
                "precision": precision[metric_index] if metric_index < len(precision) else 0.0,
                "recall": recall[metric_index] if metric_index < len(recall) else 0.0,
                "f1": f1_scores[metric_index] if metric_index < len(f1_scores) else 0.0,
                "map50_95": maps[yolo_class_id] if yolo_class_id < len(maps) else 0.0,
            }
        )

    rows.sort(key=lambda row: (row["map50_95"], row["recall"], row["precision"]))
    return rows


def plot_low_per_class_metrics(weights_path, data_yaml_path, top_k=10, output_path=None):
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    data_yaml_path = Path(data_yaml_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YOLO dataset yaml not found: {data_yaml_path}")

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml_path))

    class_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    rows = build_per_class_metrics(metrics, class_names)
    selected_rows = rows[:top_k]

    if output_path is None:
        output_path = Path("ml/outputs/logs/yolo_validation/low_per_class_metrics.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [row["class_name"] for row in selected_rows]
    precision_values = [row["precision"] for row in selected_rows]
    recall_values = [row["recall"] for row in selected_rows]
    map_values = [row["map50_95"] for row in selected_rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, top_k * 0.45)), sharey=True)

    axes[0].barh(labels, precision_values, color="#4E79A7")
    axes[0].set_title("Precision")
    axes[0].set_xlim(0, 1)

    axes[1].barh(labels, recall_values, color="#F28E2B")
    axes[1].set_title("Recall")
    axes[1].set_xlim(0, 1)

    axes[2].barh(labels, map_values, color="#E15759")
    axes[2].set_title("mAP50-95")
    axes[2].set_xlim(0, 1)

    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
        axis.invert_yaxis()

    fig.suptitle(f"Lowest {len(selected_rows)} Classes by mAP50-95", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("=" * 55)
    print("YOLO validation plot 완료")
    print("=" * 55)
    print(f"  weights            : {weights_path}")
    print(f"  data yaml          : {data_yaml_path}")
    print(f"  top_k              : {top_k}")
    print(f"  output plot        : {output_path}")
    print("  성능이 낮은 클래스:")
    for row in selected_rows:
        print(
            f"    class={row['class_name']:<12} "
            f"map50_95={row['map50_95']:.4f} "
            f"recall={row['recall']:.4f} "
            f"precision={row['precision']:.4f}"
        )

    return {
        "metrics": metrics,
        "rows": rows,
        "selected_rows": selected_rows,
        "output_path": output_path,
    }
