"""
데이터 로드 & EDA
------------------------
입력: output/train_coco.json, output/category_mapping.json
출력: output/eda/ 폴더에 시각화 이미지 저장
"""

import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정 
ZIP_PATH     = r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\ai10-level1-project.zip"
OUTPUT_DIR   = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\output")
COCO_JSON    = OUTPUT_DIR / "train_coco.json"
MAPPING_JSON = OUTPUT_DIR / "category_mapping.json"
EDA_DIR      = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("EDA 시작")
print("=" * 60)

# 데이터 로드 
with open(COCO_JSON, encoding="utf-8") as f:
    coco = json.load(f)
with open(MAPPING_JSON, encoding="utf-8") as f:
    mapping = json.load(f)

images      = coco["images"]
annotations = coco["annotations"]
categories  = coco["categories"]
id_to_name  = {int(k): v for k, v in mapping["id_to_name"].items()}

print(f"  이미지 수      : {len(images)}장")
print(f"  어노테이션 수  : {len(annotations)}개")
print(f"  카테고리 수    : {len(categories)}종")

# 집계 
# 클래스별 count
class_count = defaultdict(int)
for ann in annotations:
    class_count[ann["category_id"]] += 1

# 이미지당 알약 수
img_pill_count = defaultdict(int)
for ann in annotations:
    img_pill_count[ann["image_id"]] += 1

# bbox 크기 분포
bbox_areas  = [ann["area"] for ann in annotations]
bbox_widths = [ann["bbox"][2] for ann in annotations]
bbox_heights= [ann["bbox"][3] for ann in annotations]

# 이미지 크기
img_widths  = [img["width"]  for img in images]
img_heights = [img["height"] for img in images]

# 시각화 1: 클래스별 어노테이션 수 
print("\n[1] 클래스별 분포 시각화...")
sorted_classes = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
names  = [id_to_name.get(cid, str(cid))[:10] for cid, _ in sorted_classes]
counts = [cnt for _, cnt in sorted_classes]

fig, ax = plt.subplots(figsize=(18, 6))
colors = ['#E24B4A' if c <= 5 else '#EF9F27' if c <= 12 else '#1D9E75'
          for c in counts]
bars = ax.bar(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_xlabel("약품명")
ax.set_ylabel("어노테이션 수")
ax.set_title("클래스별 어노테이션 수 분포 (빨강=3~5개, 주황=6~12개, 초록=13개+)")
ax.axhline(y=np.mean(counts), color='gray', linestyle='--', alpha=0.7,
           label=f'평균 {np.mean(counts):.1f}개')
ax.legend()

# 막대 위 숫자
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(cnt), ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(EDA_DIR / "01_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  저장: {EDA_DIR / '01_class_distribution.png'}")

# 시각화 2: 이미지당 알약 수 분포 
print("[2] 이미지당 알약 수 분포...")
pill_dist = defaultdict(int)
for cnt in img_pill_count.values():
    pill_dist[cnt] += 1

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 파이 차트
labels = [f"{k}개짜리\n({v}장)" for k, v in sorted(pill_dist.items())]
sizes  = [v for _, v in sorted(pill_dist.items())]
colors_pie = ['#85B7EB', '#1D9E75', '#EF9F27']
axes[0].pie(sizes, labels=labels, colors=colors_pie[:len(sizes)],
            autopct='%1.1f%%', startangle=90)
axes[0].set_title("이미지당 알약 수 분포")

# 막대 차트
keys = sorted(pill_dist.keys())
axes[1].bar([str(k) + "개" for k in keys],
            [pill_dist[k] for k in keys],
            color=colors_pie[:len(keys)], edgecolor='white')
axes[1].set_xlabel("이미지당 알약 수")
axes[1].set_ylabel("이미지 수")
axes[1].set_title("이미지당 알약 수")
for i, (k, v) in enumerate([(k, pill_dist[k]) for k in keys]):
    axes[1].text(i, v + 0.5, str(v), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(EDA_DIR / "02_pills_per_image.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  저장: {EDA_DIR / '02_pills_per_image.png'}")

# 시각화 3: bbox 크기 분포 
print("[3] bbox 크기 분포...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(bbox_widths,  bins=30, color='#378ADD', edgecolor='white', alpha=0.8)
axes[0].set_title("bbox 너비 분포")
axes[0].set_xlabel("픽셀")
axes[0].axvline(np.mean(bbox_widths), color='red', linestyle='--',
                label=f"평균 {np.mean(bbox_widths):.0f}px")
axes[0].legend()

axes[1].hist(bbox_heights, bins=30, color='#1D9E75', edgecolor='white', alpha=0.8)
axes[1].set_title("bbox 높이 분포")
axes[1].set_xlabel("픽셀")
axes[1].axvline(np.mean(bbox_heights), color='red', linestyle='--',
                label=f"평균 {np.mean(bbox_heights):.0f}px")
axes[1].legend()

axes[2].hist(bbox_areas,   bins=30, color='#EF9F27', edgecolor='white', alpha=0.8)
axes[2].set_title("bbox 면적 분포")
axes[2].set_xlabel("픽셀²")
axes[2].axvline(np.mean(bbox_areas), color='red', linestyle='--',
                label=f"평균 {np.mean(bbox_areas):.0f}px²")
axes[2].legend()

plt.tight_layout()
plt.savefig(EDA_DIR / "03_bbox_size_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  저장: {EDA_DIR / '03_bbox_size_distribution.png'}")

# 시각화 4: bbox 위치 히트맵 
print("[4] bbox 위치 히트맵...")
# 정규화된 중심점 수집
cx_list, cy_list = [], []
for ann in annotations:
    img_id = ann["image_id"]
    img_info = next((img for img in images if img["id"] == img_id), None)
    if img_info is None:
        continue
    x, y, w, h = ann["bbox"]
    cx = (x + w/2) / img_info["width"]
    cy = (y + h/2) / img_info["height"]
    cx_list.append(cx)
    cy_list.append(cy)

fig, ax = plt.subplots(figsize=(7, 6))
heatmap, xedges, yedges = np.histogram2d(cx_list, cy_list, bins=20,
                                          range=[[0,1],[0,1]])
im = ax.imshow(heatmap.T, origin='lower', cmap='YlOrRd',
               extent=[0, 1, 0, 1], aspect='auto')
plt.colorbar(im, ax=ax, label='bbox 빈도')
ax.set_xlabel("이미지 너비 방향 (0=왼쪽, 1=오른쪽)")
ax.set_ylabel("이미지 높이 방향 (0=위, 1=아래)")
ax.set_title("bbox 중심점 위치 히트맵\n(알약이 이미지 어디에 주로 위치하는가)")
plt.tight_layout()
plt.savefig(EDA_DIR / "04_bbox_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  저장: {EDA_DIR / '04_bbox_heatmap.png'}")

# 시각화 5: 샘플 이미지 + bbox 시각화
print("[5] 샘플 이미지 bbox 시각화 (3장)...")

# 이미지별 어노테이션 그룹핑
img_anns = defaultdict(list)
for ann in annotations:
    img_anns[ann["image_id"]].append(ann)

# 알약 4개짜리 이미지 우선 선택
target_imgs = [img for img in images
               if len(img_anns.get(img["id"], [])) == 4][:3]
if len(target_imgs) < 3:
    target_imgs = images[:3]

# 색상 팔레트
colors_bbox = ['#E24B4A', '#1D9E75', '#378ADD', '#EF9F27']

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    png_map = {Path(f).name: f for f in z.namelist()
               if f.endswith(".png") and "train_images" in f}

    fig, axes = plt.subplots(1, len(target_imgs),
                              figsize=(6 * len(target_imgs), 6))
    if len(target_imgs) == 1:
        axes = [axes]

    for ax, img_info in zip(axes, target_imgs):
        fname = img_info["file_name"]
        if fname not in png_map:
            continue

        import io
        from PIL import Image
        with z.open(png_map[fname]) as f:
            img_pil = Image.open(io.BytesIO(f.read())).convert("RGB")

        ax.imshow(img_pil)
        ax.axis('off')
        ax.set_title(f"{fname[:30]}\n"
                     f"({img_info['width']}×{img_info['height']})", fontsize=8)

        for i, ann in enumerate(img_anns.get(img_info["id"], [])):
            x, y, w, h = ann["bbox"]
            cat_name = id_to_name.get(ann["category_id"], "?")[:8]
            color = colors_bbox[i % len(colors_bbox)]
            rect = patches.Rectangle((x, y), w, h,
                                      linewidth=2, edgecolor=color,
                                      facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 4, cat_name, color=color, fontsize=7,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.7, edgecolor='none'))

plt.suptitle("샘플 이미지 + bbox 시각화", fontsize=12)
plt.tight_layout()
plt.savefig(EDA_DIR / "05_sample_images.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  저장: {EDA_DIR / '05_sample_images.png'}")

# EDA 요약 출력 
print("\n" + "=" * 60)
print("EDA 요약")
print("=" * 60)
print(f"  이미지 크기 범위  : {min(img_widths)}~{max(img_widths)} × "
      f"{min(img_heights)}~{max(img_heights)}")
print(f"  bbox 너비 평균    : {np.mean(bbox_widths):.1f}px  "
      f"(min {min(bbox_widths):.0f} / max {max(bbox_widths):.0f})")
print(f"  bbox 높이 평균    : {np.mean(bbox_heights):.1f}px  "
      f"(min {min(bbox_heights):.0f} / max {max(bbox_heights):.0f})")
print(f"  이미지당 알약 수  : {dict(sorted(pill_dist.items()))}")
print(f"  클래스 불균형 비율: {max(counts)} / {min(counts)} = "
      f"{max(counts)/min(counts):.1f}배")
print(f"\n  EDA 결과 저장 위치: {EDA_DIR}")
print("=" * 60)

