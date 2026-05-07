from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
data_root = project_root / "ml" / "data"

from ml.src.augment.bbox_augment import bbox_augment

img_path = str(data_root / "processed" / "yolo_single_class" / "images" / "train")
ann_path = str(data_root / "processed" / "yolo_single_class" / "labels" / "train")

def main():
    
    bbox_augment(img_path, ann_path)


if __name__ == "__main__":
    main()
