# RoadVision DUET — Vehicle Detection Pipeline

**Team:** Backprop SUST  
**Competition:** AI Hackathon, DUET CSE Carnival 2026 — hosted by Dept. of CSE, DUET, Gazipur, Bangladesh  
**Kaggle:** https://www.kaggle.com/t/f421b6563ade432a9b68bd19349e1357  
**Metric:** mAP@0.50

---

## Final Scores

| Leaderboard | mAP@0.50 |
|-------------|----------|
| Public      | 0.62279  |
| Private     | 0.59821  |

---

## Dataset at a Glance

| Property                  | Train  | Test |
|---------------------------|--------|------|
| Images                    | 810    | 327  |
| Annotations               | 10,475 | —    |
| Avg objects / image       | 12.93  | —    |
| Median objects / image    | 11     | —    |
| Max objects in one image  | 38     | —    |
| Camera sources            | 4      | 4    |
| Vehicle classes           | 13     | 13   |

---

## Data Quality Audit

| Check                                  | Count | % of annotations |
|----------------------------------------|-------|-----------------|
| Missing values                         | 0     | 0.000           |
| Duplicate annotation rows              | 0     | 0.000           |
| Zero or negative width / height        | 0     | 0.000           |
| Clipped / boundary-exceeding boxes     | 64    | —               |
| Micro-tiny boxes (area < 0.0005)       | 120   | —               |

---

## Class Distribution

| ID | Class        | Annotations | Share (%) |
|----|--------------|-------------|-----------|
| 3  | Sedan Car    | 2,480       | 23.68     |
| 8  | Agro Use     | 83          | 0.79      |

> Full per-class counts are printed at runtime by EDA cell [A4]. Sedan Car is the most frequent (2,480 · 23.68%) and Agro Use is the rarest (83 · 0.79%).

**Imbalance ratio (Sedan Car / Agro Use): 29.88×**  
**Shannon entropy: 3.063 / 3.700 bits (82.8% of maximum)**

---

## Bounding Box Geometry

| Metric      | Value   |
|-------------|---------|
| Mean area   | 0.00862 |
| Median area | 0.00421 |
| Median width| 0.046   |
| Min width   | 0.006   |

Smallest footprint: **Motorcycle**. Largest: **Large Bus** and **Medium Truck**. Trailer, Mini Truck, Microbus, Sedan Car, and Pickup overlap in geometry range.

---

## Rare Class Configuration

| Class ID | Class Name   | Oversample Repeats | Copy-Paste Target |
|----------|--------------|--------------------|-------------------|
| 8        | Agro Use     | 4×                 | Yes               |
| 7        | Mini Truck   | 3×                 | Yes               |
| 12       | Trailer      | 2×                 | Yes               |
| 4        | Pickup       | 1×                 | No                |
| 6        | Mini Bus     | 1×                 | No                |
| 9        | Medium Truck | 1×                 | No                |
| 11       | Heavy Truck  | 1×                 | No                |

---

## EDA → Modeling Decisions

| EDA Signal                       | Key Number                                   | Modeling Decision                                            |
|----------------------------------|----------------------------------------------|--------------------------------------------------------------|
| Class imbalance                  | 29.88× (Sedan Car vs Agro Use)               | Rare class oversampling + class-aware copy-paste augmentation |
| Small object geometry            | Median width 0.046; min width 0.006          | Image size 1280 + SAHI tiled inference (640×640 slices)      |
| Dense traffic scenes             | Avg 12.93 objects / image; max 38            | Mosaic (1.0) + MixUp (0.10)                                  |
| Box overlap and class similarity | Trailer / Mini Truck / Microbus / Sedan / Pickup share similar geometry | WBF post-processing + low confidence preservation |
| Shannon entropy below maximum    | 3.063 / 3.700 bits (82.8%)                   | Rare class weighting and oversampling                        |
| Boundary-exceeding boxes         | 64 boxes                                     | `safe_yolo_line` clipping in label conversion                |

---

## Training Configuration

| Parameter               | Value                                           |
|-------------------------|-------------------------------------------------|
| Framework               | Ultralytics YOLOv8                              |
| Starting weights        | BNVD fine-tuned YOLOv8 (`yolov8x.pt` fallback) |
| Image size              | 1280                                            |
| Batch size              | 8                                               |
| Total epochs            | 60                                              |
| Stage 1 — frozen        | 20 epochs                                       |
| Stage 2 — unfrozen      | 40 epochs                                       |
| Layers frozen (Stage 1) | All except last 10                              |
| Early stopping patience | 15                                              |
| Optimizer               | auto                                            |
| AMP                     | True                                            |
| Seed                    | 42                                              |
| Device                  | GPU [0]                                         |
| Train / val split       | 80% / 20%                                       |

---

## Augmentation Parameters

### YOLO Built-in Augmentation

| Parameter    | Value |
|--------------|-------|
| mosaic       | 1.0   |
| close_mosaic | 8     |
| mixup        | 0.10  |
| copy_paste   | 0.0   |
| hsv_h        | 0.015 |
| hsv_s        | 0.7   |
| hsv_v        | 0.4   |
| fliplr       | 0.5   |
| flipud       | 0.0   |
| degrees      | 0.0   |
| scale        | 0.5   |
| translate    | 0.1   |

### Low-Light Simulation Parameters

| Parameter                    | Value               |
|------------------------------|---------------------|
| Base application probability | 0.25                |
| CCTV10 camera probability    | 0.60                |
| Rare class boost             | +0.15 (cap at 0.85) |
| Brightness range             | 0.45 – 0.72         |
| Gamma range                  | 0.60 – 0.85         |
| Gaussian noise std range     | 4 – 14              |
| Blur probability             | 0.4                 |
| Blur kernel sizes            | 3×3 or 5×5          |

### Synthetic Data Generation

| Parameter                   | Value                                      |
|-----------------------------|--------------------------------------------|
| Copy-paste synthetic images | 200                                        |
| Source classes              | Mini Truck (7), Agro Use (8), Trailer (12) |
| Min paste box size          | width ≥ 0.015, height ≥ 0.015             |
| Paste IoU reject threshold  | 0.30                                       |
| Paste y-range               | Bottom 60% of image                        |
| Pastes per synthetic image  | 1 – 2                                      |

---

## Inference Configuration

### Standard Inference

| Parameter        | Value |
|------------------|-------|
| Image size       | 1280  |
| Confidence       | 0.001 |
| IoU threshold    | 0.6   |

### SAHI Tiled Inference

| Parameter             | Value     |
|-----------------------|-----------|
| Slice height          | 640 px    |
| Slice width           | 640 px    |
| Overlap ratio (H & W) | 0.25      |
| Confidence threshold  | 0.05      |
| Postprocess type      | GREEDYNMM |
| Postprocess metric    | IOU       |
| Postprocess threshold | 0.5       |
| Standard pred enabled | Yes       |

### Weighted Box Fusion (WBF)

| Parameter          | Value |
|--------------------|-------|
| IoU threshold      | 0.55  |
| Skip box threshold | 0.05  |
| Confidence type    | avg   |
| Model weight       | 1.0   |
| Final conf filter  | 0.03  |

---

## Validation Results (Holdout — 20% of 810 images)

| Metric        | Value |
|---------------|-------|
| Precision     | 0.804 |
| Recall        | 0.742 |
| mAP@0.50      | 0.815 |
| mAP@0.50-0.95 | 0.556 |

---

## Output Verification

| Submission file                       | Rows | Empty rows |
|---------------------------------------|------|------------|
| submission_sahi_3fold_wbf.csv (final) | 327  | 0          |
| submission_normal_3fold_wbf.csv       | 327  | 0          |

---

## Competition Details

| Property              | Value                                   |
|-----------------------|-----------------------------------------|
| Host                  | Dept. of CSE, DUET, Gazipur, Bangladesh |
| Partial sponsor       | ICT Division, Bangladesh                |
| Total prize pool      | 90,000 BDT                              |
| Max team size         | 3 members                               |
| Max submissions / day | 5                                       |
| Final submissions     | 2 selectable                            |
| Online phase weight   | 70%                                     |
| Onsite phase weight   | 30%                                     |
| Evaluation metric     | mAP@0.50                                |
| True positive IoU     | ≥ 0.50                                  |

### Prize Allocation

| Place                    | Prize (BDT) |
|--------------------------|-------------|
| Champion                 | 35,000      |
| 1st Runner-up            | 20,000      |
| 2nd Runner-up            | 15,000      |
| DUET Rising Team Award   | 10,000      |
| Female Rising Team Award | 10,000      |

### Competition Timeline

| Milestone                          | Date / Time (BST, GMT+6)     |
|------------------------------------|------------------------------|
| Registration opens                 | 13 June 2026                 |
| Registration closes                | 20 June 2026                 |
| Competition start / dataset reveal | 23 June 2026, 8:00 AM        |
| Entry deadline                     | 24 June 2026, 11:55 PM       |
| Team merger deadline               | 24 June 2026, 11:55 PM       |
| Final submission deadline          | 24 June 2026, 11:55 PM       |

---

## Setup

```bash
pip install ultralytics sahi ensemble-boxes scikit-learn opencv-python pandas numpy torch
```

Update paths in Cell 1 before running:

```python
TRAIN_IMG    = "/path/to/train/images"
TEST_IMG     = "/path/to/test/images"
TRAIN_CSV    = "/path/to/train/train.csv"
SAMPLE_SUB   = "/path/to/test/sample_submission.csv"
BNVD_WEIGHTS = "/path/to/yolov8_ods_new_100e_best.pt"
```

Run pipeline cells 0 → 9 sequentially. Cell 10 is optional visualization (16 annotated test images at `conf=0.25`).

---

## Submission Format

```
image_id,PredictionString
img_0001,0 0.94 0.34 0.50 0.12 0.22 5 0.81 0.20 0.60 0.26 0.14
img_0002,
```

Each detection: `class_id confidence x_center y_center width height` (coordinates normalized 0–1). Empty `PredictionString` is valid for images with no predictions.

---

## Limitations

| Limitation                | Impact                                              |
|---------------------------|-----------------------------------------------------|
| 2-day competition window  | No time for deeper hyperparameter search            |
| No external datasets used | Permitted but unexplored                            |
| Synthetic augmentation    | Cannot fully replace real rare-class variety        |
| SAHI tiled inference      | Improves recall but significantly increases inference time |

---

## Integrity

- No manual labeling of test data
- No leaked labels used at any stage
- Submission fully automated by the pipeline
- Winning code licensed under OSI-approved open source license (per competition rules)
