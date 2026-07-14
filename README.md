# Road Vision — DUET AI Hackathon 2025
**Team Backprop_SUST**- Ranked in the top 15 out of 101 competing teams and were selected to present their solution at the final showcase.

Multi-class vehicle detection from Bangladeshi CCTV footage using YOLOv8x fine-tuned on BNVD weights, with SAHI tiled inference and Weighted Boxes Fusion post-processing.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Analysis](#dataset-analysis)
- [Pipeline Overview](#pipeline-overview)
- [Results](#results)
- [Reproducing the Solution](#reproducing-the-solution)
- [Dependencies](#dependencies)
- [References](#references)

---

## Problem Statement

Detect and classify 13 vehicle types from traffic camera images captured across 4 CCTV sources in Bangladesh. Evaluated on mAP@0.50.

**Classes (13):** Rickshaw, Motorcycle, Tempu, Sedan Car, Pickup, Microbus, Mini Bus, Mini Truck, Agro Use, Medium Truck, Large Bus, Heavy Truck, Trailer

---

## Dataset Analysis

Key findings from EDA that drove all pipeline decisions:

| Metric | Value |
|---|---|
| Total images | ~1,635 |
| Total bounding boxes | ~17,000+ |
| Class imbalance ratio (max/min) | **29.88×** |
| Shannon entropy (class dist.) | 2.71 / 3.70 max |
| Median box area | 0.00421 (normalised) |
| Camera sources | 4 (CCTV10, CCTV11, CCTV12, CCTV13) |

**Rare classes** (high augmentation priority):

| Class | Approx. Count | Strategy |
|---|---|---|
| Agro Use | ~83 | 4× oversample + copy-paste |
| Mini Truck | ~120 | 3× oversample + copy-paste |
| Trailer | ~150 | 2× oversample + copy-paste |
| Pickup, Mini Bus, Medium Truck, Heavy Truck | 200–400 | 1× oversample |

CCTV10 was identified as the primary low-light source, receiving 60% low-light augmentation probability vs. 25% baseline.

Small object density (median box area ~0.004) directly motivated SAHI tiled inference at inference time.

---

## Pipeline Overview

```
BNVD Pretrained Weights (YOLOv8x, 17-class Bangladeshi vehicles)
        │
        ▼
  [Cell 2] CSV → YOLO Label Conversion  (safe_yolo_line validation)
        │
        ▼
  [Cell 3] 80/20 Train/Val Split  (random, seed=42)
        │
        ▼
  [Cell 4] Rare-Class Augmentation
           ├── Oversampling  (1–4× per image based on RARE_REPEAT)
           └── Copy-Paste Synthesis  (200 images, classes 7/8/12 only)
        │
        ▼
  [Cell 5] Low-Light Augmentation
           └── Brightness × Gamma × Gaussian noise  (cam-aware probability)
        │
        ▼
  [Cell 6] 2-Stage YOLOv8x Training
           ├── Stage 1: Frozen backbone, 20 epochs  (last 10 layers active)
           └── Stage 2: Full fine-tune, 40 epochs
        │
        ├──▶ [Cell 7] Standard Inference + WBF → submission_normal.csv
        └──▶ [Cell 8] SAHI Tiled Inference + WBF → submission_sahi.csv  ✅ submitted
```

### Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Image size | 1280 | Large images; small objects |
| Total epochs | 60 (20+40) | 2-stage frozen/unfrozen |
| Batch size | 8 | GPU memory constraint |
| SAHI slice | 640×640, 25% overlap | Dense small-object detection |
| SAHI conf threshold | 0.05 | High recall, filter at WBF stage |
| WBF IoU threshold | 0.55 | |
| Final conf threshold | 0.03 | |

---

## Results

| Submission | Public mAP@0.50 |
|---|---|
| Standard inference + WBF | ~0.59 |
| **SAHI tiled inference + WBF** | **0.622** |

Val mAP@0.50: 0.815 

SAHI provided the largest single improvement by recovering small and partially-occluded vehicles that standard full-image inference missed.

---

## Reproducing the Solution

### 1. Weights

This solution requires two weight files:

**BNVD pretrained weights** (starting point for fine-tuning):
- Repository: [github.com/bipin-saha/BNVD](https://github.com/bipin-saha/BNVD)
- The YOLOv8x checkpoint is in the `Checkpoints/` folder of that repo
- Dataset page: [bipin-saha.github.io/BNVD](https://bipin-saha.github.io/BNVD/)
- Paper: [arxiv.org/abs/2405.12150](https://arxiv.org/abs/2405.12150)

> **Note on class mismatch:** BNVD has 17 classes (Bicycle, Bus, Bhotbhoti, Car, CNG, Easybike, etc.); this competition has 13 different classes. The BNVD backbone provides domain-adapted feature extraction for Bangladeshi traffic — the final detection head is re-initialised for 13 classes during training. Set `START_WEIGHTS` in Cell 1 to your local BNVD checkpoint path.

**Fine-tuned competition weights** (`yolov8_ods_new_100e_best.pt`):
- This file is stored as a private Kaggle dataset under `miftahuladib/finetuned-model`
- Contact the team if you need access for evaluation purposes

### 2. Data

Competition data from the DUET Road Vision challenge. Expected layout:

```
/kaggle/input/competitions/road-vision/RoadVision_DUET/
    train/
        images/      # .jpg files
        train.csv    # image_id, class_id, x_center, y_center, width, height (normalised)
    test/
        images/
        sample_submission.csv
```

### 3. Running

Open `Backprop_SUST.ipynb` on Kaggle (GPU T4×2 recommended) and run cells 0–9 in order. Cell 10 is optional visualisation.

```
Cell 0  — Install dependencies
Cell 1  — Configuration (set BNVD_WEIGHTS path here)
Cell 2  — Convert CSV labels to YOLO format
Cell 3  — Train/val split
Cell 4  — Rare-class oversampling + copy-paste augmentation
Cell 5  — Low-light augmentation
Cell 6  — 2-stage training
Cell 7  — Standard inference → submission_normal_3fold_wbf.csv
Cell 8  — SAHI inference   → submission_sahi_3fold_wbf.csv  ← use this
Cell 9  — Summary
Cell 10 — (Optional) Visualise predictions on test images
```

---


## Dependencies

```
ultralytics>=8.0
sahi>=0.11
ensemble-boxes>=1.0.4
scikit-learn>=1.3
opencv-python
pandas
numpy
```

Install: `pip install -q ultralytics sahi ensemble-boxes scikit-learn`

---



## References

- [BNVD: Bangladeshi Native Vehicle Detection in Wild](https://arxiv.org/abs/2405.12150) — Saha et al., 2024
- [BNVD GitHub + Checkpoints](https://github.com/bipin-saha/BNVD)
- [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
