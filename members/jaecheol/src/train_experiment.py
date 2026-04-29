"""
최종 실험 학습
--------------
실험 구성:
  실험군1: YOLOv11l  (신버전 YOLO)
  실험군2: RT-DETRv2 (Transformer 기반)

주요 설정:
  - 데이터    : 1116장 (186 원본 + 840 소수 클래스 증강 + 90 인터넷 이미지)
  - bbox 이상치 수정 (ann_id=240 x좌표 6567 → 567)
  - imgsz     : 1280
  - epochs    : 150
  - hsv_h/s   : 0.0 (알약 색상 보존)
  - hsv_v     : 0.4 (밝기만 허용)
  - copy_paste: 0.5
  - shear/mixup/erasing: 0.0 제거
"""

import json
from pathlib import Path
from ultralytics import YOLO, RTDETR
import os

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    OUTPUT_DIR = Path(r"C:\Users\Admin\Desktop\AI Engineer 10th\초급 프로젝트\output")
    YAML_PATH  = OUTPUT_DIR / "pill.yaml"
    RUNS_DIR   = OUTPUT_DIR / "runs"

    # ── 공통 설정 ─────────────────────────────────────────────────
    COMMON = dict(
        data          = str(YAML_PATH),
        epochs        = 150,
        imgsz         = 1280,
        batch         = 4,
        device        = 0,
        project       = str(RUNS_DIR),
        exist_ok      = True,
        patience      = 50,
        workers       = 0,
        save          = True,
        save_period   = 10,
        val           = True,
        plots         = True,
        verbose       = True,

        # ── 증강 ─────────────────────────────────────────────────
        mosaic        = 1.0,
        copy_paste    = 0.5,
        degrees       = 180.0,
        flipud        = 0.5,
        fliplr        = 0.5,
        hsv_h         = 0.0,           # 색조 변환 제거 (알약 색상 보존)
        hsv_s         = 0.0,           # 채도 변환 제거 (알약 색상 보존)
        hsv_v         = 0.4,           # 밝기만 유지 (조명 시뮬레이션)
        scale         = 0.5,
        translate     = 0.1,
        close_mosaic  = 20,

        # ── 제거된 증강 ───────────────────────────────────────────
        shear         = 0.0,           # 알약에 부자연스러운 변환
        mixup         = 0.0,           # 색상 특징 파괴 방지
        perspective   = 0.0,           # 효과 미미
        erasing       = 0.0,           # 각인 문자 보호

        # ── 학습률 ────────────────────────────────────────────────
        lr0           = 0.001,
        lrf           = 0.0001,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 5,
        cos_lr        = True,
    )

    # ── 실험군1: YOLOv11l ─────────────────────────────────────────
    print("=" * 60)
    print("실험군1: YOLOv11l (신버전 YOLO)")
    print("=" * 60)

    model_v11 = YOLO("yolo11l.pt")
    res_v11 = model_v11.train(
        **COMMON,
        name = "exp_yolo11l_final",
    )
    map_v11 = res_v11.results_dict.get("metrics/mAP50-95(B)", 0)
    print(f"\n  YOLOv11l mAP@0.5:0.95 = {map_v11:.4f}")

    # ── 실험군2: RT-DETRv2 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("실험군2: RT-DETRv2 (Transformer 기반)")
    print("=" * 60)

    rtdetr_config = {**COMMON}
    rtdetr_config.update(dict(
        lr0           = 0.0001,        # Transformer 적정 학습률
        lrf           = 0.00001,
        mosaic        = 0.5,           # 전역 맥락 학습 보호
        warmup_epochs = 10,            # Transformer 더 긴 워밍업
        name          = "exp_rtdetr_final",
    ))

    model_r = RTDETR("rtdetr-l.pt")
    res_r = model_r.train(**rtdetr_config)
    map_r = res_r.results_dict.get("metrics/mAP50-95(B)", 0)
    print(f"\n  RT-DETRv2 mAP@0.5:0.95 = {map_r:.4f}")

    # ── 최종 결과 비교 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("최종 결과 비교")
    print("=" * 60)
    print(f"  YOLOv11l  : {map_v11:.4f}")
    print(f"  RT-DETRv2 : {map_r:.4f}")

    scores = {"YOLOv11l": map_v11, "RT-DETRv2": map_r}
    winner = max(scores, key=scores.get)
    print(f"\n  최종 승자: {winner} ({scores[winner]:.4f})")

    summary = {
        "exp_yolo11l" : round(map_v11, 4),
        "exp_rtdetr"  : round(map_r,   4),
        "winner"      : winner,
    }
    summary_path = RUNS_DIR / "experiment_summary_final.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  요약 저장: {summary_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
