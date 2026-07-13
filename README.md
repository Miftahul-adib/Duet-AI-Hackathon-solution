# RoadVision DUET — Vehicle Detection Pipeline

**Team:** Backprop SUST &nbsp;·&nbsp; DUET AI Hackathon, CSE Carnival 2026  
**Task:** Multi-class vehicle detection in Bangladesh highway CCTV footage  
**Result:** mAP@0.50 of **0.62279** (public) · **0.59821** (private) — selected as one of **5 finalist teams** to present at the onsite finals out of **101 registered teams**

---

## Overview

Bangladesh road traffic is uniquely challenging for object detection — dense mixed traffic, region-specific vehicle types (Rickshaw, Tempu, Agro Use), low-resolution CCTV cameras, and severe lighting variation. Standard models trained on COCO or Cityscapes do not transfer well.

This solution is a YOLOv8-based pipeline where every architectural decision was directly motivated by EDA findings: class imbalance drove oversampling and copy-paste augmentation, small object geometry drove high-resolution training and SAHI tiled inference, and dense scenes drove mosaic and MixUp augmentation.

---

## Dataset

| Property               | Train  | Test |
|------------------------|--------|------|
| Images                 | 810    | 327  |
| Annotations            | 10,475 | —    |
| Avg objects / image    | 12.93  | —    |
| Median objects / image | 11     | —    |
| Max objects / image    | 38     | —    |
| Camera sources         | 4      | 4    |
| Vehicle classes        | 13     | 13   |

**13 classes:** Rickshaw, Motorcycle, Tempu, Sedan Car, Pickup, Microbus, Mini Bus, Mini Truck, Agro Use, Medium Truck, Large Bus, Heavy Truck, Trailer

---

## Key EDA Findings

**Class imbalance is severe.** Sedan Car has 2,480 annotations (23.68%) while Agro Use has only 83 (0.79%) — a **29.88× imbalance ratio**. Shannon entropy of the class distribution is 3.063 / 3.700 bits (82.8% of maximum).

**Objects are small.** Mean box area is 0.00862 and median is 0.00421 (normalized). Median width is 0.046 with a minimum of 0.006. 120 boxes are micro-tiny (area < 0.0005) and 64 boxes clip image boundaries.

**Scenes are dense.** Average 12.93 objects per image with a maximum of 38, making recall-preserving inference critical.

**Classes overlap geometrically.** Trailer, Mini Truck, Microbus, Sedan Car, and Pickup share similar area, aspect ratio, and diagonal ranges — making careful post-processing essential.

**Data quality is clean.** Zero missing values, zero duplicate annotation rows, zero zero-size boxes.

---

## Pipeline

```
Data Audit → YOLO Conversion → Augmentation → 2-Stage Training → SAHI Inference → WBF → Submission
```

### Training

| Parameter               | Value                                           |
|-------------------------|-------------------------------------------------|
| Model                   | YOLOv8 (BNVD fine-tuned weights)               |
| Image size              | 1280                                            |
| Batch size              | 8                                               |
| Epochs                  | 60 (20 frozen + 40 unfrozen)                    |
| Early stopping patience | 15                                              |
| Train / val split       | 80% / 20% (seed 42)                             |

### Augmentation

| Technique              | Key Parameters                                           |
|------------------------|----------------------------------------------------------|
| Mosaic                 | prob = 1.0, close at epoch 8                             |
| MixUp                  | prob = 0.10                                              |
| HSV jitter             | H=0.015, S=0.7, V=0.4                                   |
| Flip                   | horizontal 0.5, vertical 0.0                             |
| Scale / translate      | 0.5 / 0.1                                               |
| Rare class oversample  | Agro Use 4×, Mini Truck 3×, Trailer 2×, others 1×       |
| Copy-paste synthesis   | 200 images, source: Mini Truck / Agro Use / Trailer      |
| Low-light simulation   | brightness 0.45–0.72, gamma 0.60–0.85, noise std 4–14   |

### Inference & Post-processing

| Stage              | Key Parameters                                    |
|--------------------|---------------------------------------------------|
| Standard inference | conf=0.001, IoU=0.6, imgsz=1280                   |
| SAHI tiled         | 640×640 slices, 25% overlap, conf=0.05            |
| WBF fusion         | IoU thr=0.55, skip thr=0.05, final conf thr=0.03  |

---

## Results

| Metric        | Value |
|---------------|-------|
| Precision     | 0.804 |
| Recall        | 0.742 |
| mAP@0.50      | 0.815 |
| mAP@0.50-0.95 | 0.556 |

*Validation on 20% holdout. Final test submission: 327 rows, 0 empty predictions.*

---

## Setup

```bash
pip install ultralytics sahi ensemble-boxes scikit-learn opencv-python pandas numpy torch
```

Update the three paths at the top of Cell 1 in the pipeline notebook, then run cells 0 → 9 sequentially. Cell 10 is optional visualization.

```python
TRAIN_IMG    = "/path/to/train/images"
TEST_IMG     = "/path/to/test/images"
TRAIN_CSV    = "/path/to/train/train.csv"
BNVD_WEIGHTS = "/path/to/yolov8_ods_new_100e_best.pt"
```

---

## Competition

Hosted by the Dept. of Computer Science and Engineering, Dhaka University of Engineering and Technology (DUET), partially sponsored by ICT Division Bangladesh. Total prize pool: 90,000 BDT. Final standings weighted 70% online leaderboard + 30% onsite presentation.

**Integrity:** No manual test labeling, no leaked labels, fully automated pipeline, OSI-licensed code.
