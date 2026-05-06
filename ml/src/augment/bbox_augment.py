import albumentations as A
import cv2
import os
import glob

# 1. 원본 YOLO 라벨 읽기
def read_yolo_labels(path):
    with open(path, 'r') as f:
        bboxes = []
        for line in f.readlines():
            items = line.split()
            # albumentations는 [x, y, w, h, class_id] 순서를 선호함
            bboxes.append([float(items[1]), float(items[2]), float(items[3]), float(items[4]), int(items[0])])
    return bboxes

def bbox_augment(train_data_dir:str, train_label_dir:str):
    image_files = []

    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        image_files.extend(glob.glob(f"*{ext}"))


    transforms = [
        A.Affine(
            scale=(0.9, 1.12),
            translate_percent=(-0.04, 0.04),
            rotate=(-180, 180),
            shear=(-3, 3),
            fit_output=False,
            border_mode=cv2.BORDER_REPLICATE,
            p=0.9,
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                A.ImageCompression(quality_range=(78, 96), p=1.0),
            ],
            p=0.35,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ]

    for img_path in image_files:
        image = cv2.imread(os.path.join(train_data_dir, img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = read_yolo_labels(train_label_dir)

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        

                # 3. 증강 반복 및 파일 저장
        
        num_augmentations = 5  # 생성할 증강 이미지 개수
    
        for i in range(num_augmentations):
            # 증강 적용
            transformed = transforms(image=image, bboxes=bboxes)
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
    
            
            # 고유 파일명 생성
            file_name = f"aug_{img_name}_{i}.jpg"
            save_path = os.path.join(train_data_dir, file_name)
            
            # 이미지 저장 (다시 BGR로 변환하여 저장)
            cv2.imwrite(save_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            
            # YOLO txt 라벨 저장
            with open(f"{train_label_dir}/{file_name}.txt", 'w') as f:
                for bbox in aug_bboxes:
                    # bbox: [x_center, y_center, width, height, class_id]
                    f.write(f"{int(bbox[4])} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")    

def main():
    bbox_augment()

if __name__ == "__main__":
    main()

    