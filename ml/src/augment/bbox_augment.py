import albumentations as A
import cv2
import os
import glob
import numpy as np

# 1. 원본 YOLO 라벨 읽기
def read_yolo_labels(path):
    bboxes = []
    if not os.path.exists(path): return bboxes # 파일 없을 경우 대비
    with open(path, 'r') as f:
        for line in f.readlines():
            items = line.split()
            if len(items) < 5: continue
            # [수정] items[1:5]는 x,y,w,h / items[0]은 class_id
            # Albumentations YOLO 형식: [x, y, w, h, class_id]
            bboxes.append([float(items[1]), float(items[2]), float(items[3]), float(items[4]), int(items[0])])
    return bboxes

def bbox_augment(train_data_dir:str, train_label_dir:str):
    image_files = glob.glob(os.path.join(train_data_dir, "*.jpg")) # [수정] 경로 꼬임 방지 및 확장자 명시

    transforms = A.Compose([
        A.Affine(
            scale=(0.9, 1.12),
            translate_percent=(-0.04, 0.04),
            rotate=(-180, 180),
            shear=(-3, 3),
            fit_output=False,
            border_mode=cv2.BORDER_REPLICATE,
            p=0.9,
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.ImageCompression(quality_range=(78, 96), p=1.0),
        ], p=0.35),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ], bbox_params=A.BboxParams(format='yolo')) # [수정] 필수 설정 추가

    for img_path in image_files:
        image = cv2.imread(img_path) # [수정] 중복된 os.path.join 제거
        if image is None: continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        bboxes = read_yolo_labels(os.path.join(train_label_dir, img_name+'.txt'))
        # bboxes = np.array(bboxes, dtype=np.float32) # [수정] TypeError 방지를 위해 삭제
        
        if not bboxes: continue # 라벨이 없는 이미지는 건너뜀

        num_augmentations = 5 
        for i in range(num_augmentations):
            # [중요] check_each_transform을 통해 좌표 오류 지점 확인 가능하지만 일단 실행
            try:
                transformed = transforms(image=image, bboxes=bboxes)
                aug_image = transformed['image']
                aug_bboxes = transformed['bboxes']
        
                file_name = f"aug_{img_name}_{i}.jpg"
                save_path = os.path.join(train_data_dir, file_name)
                cv2.imwrite(save_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                
                with open(os.path.join(train_label_dir, f"aug_{img_name}_{i}.txt"), 'w') as f:
                    for bbox in aug_bboxes:
                        # bbox: [x, y, w, h, class_id]
                        f.write(f"{int(bbox[4])} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            except ValueError as e:
                print(f"Error skipping {img_name}: {e}")
                continue

def main():
    # 경로를 직접 넣어주세요
    bbox_augment("/content/.../images/train", "/content/.../labels/train")

if __name__ == "__main__":
    main()
