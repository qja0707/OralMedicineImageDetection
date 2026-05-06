from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
data_root = project_root / "ml" / "data"

from ml.src.augment.bbox_augment import bbox_augment

def main():
    
    bbox_augment()


if __name__ == "__main__":
    main()
