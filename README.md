# RoadVision DUET — Vehicle Detection Pipeline

**Team:** Backprop SUST  
**Competition:** AI Hackathon, DUET CSE Carnival 2026 — hosted by Dept. of CSE, DUET, Gazipur, Bangladesh  
**Metric:** mAP@0.50


---

## Dataset at a Glance

| Property                            | Train   | Test |
|-------------------------------------|---------|------|
| Images                              | 810     | 327  |
| Annotations                         | 10,475  | —    |
| Avg objects / image                 | 12.93   | —    |
| Median objects / image              | 11      | —    |
| Max objects in one image            | 38      | —    |
| Camera sources                      | 4       | 4    |
| Vehicle classes                     | 13      | 13   |
| Images with exactly 1 object        | —       | —    |
| Images with > 20 objects            | —       | —    |
| Duplicate annotation rows           | 0       | —    |

---

## Class Distribution

| ID | Class        | Annotations | Share (%) | Unique Images | Ann / Image | Rare Class | Copy-Paste Target | Oversample Repeats |
|----|--------------|-------------|-----------|---------------|-------------|------------|--------------------|--------------------|
| 3  | Sedan Car    | 2,480       | 23.68     | —             | —           | No         | No                 | —                  |
| 1  | Motorcycle   | 1,715       | 16.37     | —             | —           | No         | No                 | —                  |
| 0  | Rickshaw     | 1,422       | 13.58     | —             | —           | No         | No                 | —                  |
| 5  | Microbus     | 1,120       | 10.69     | —             | —           | No         | No                 | —                  |
| 2  | Tempu        | 861         | 8.22      | —             | —           | No         | No                 | —                  |
| 10 | Large Bus    | 749         | 7.15      | —             | —           | No         | No                 | —                  |
| 3  | Pickup       | —           | —         | —             | —           | Yes        | No                 | 1×                 |
| 6  | Mini Bus     | —           | —         | —             | —           | Yes        | No                 | 1×                 |
| 9  | Medium Truck | —           | —         | —             | —           | Yes        | No                 | 1×                 |
| 11 | Heavy Truck  | —           | —         | —             | —           | Yes        | No                 | 1×                 |
| 7  | Mini Truck   | —           | —         | —             | —           | Yes        | Yes                | 3×                 |
| 12 | Trailer      | —           | —         | —             | —           | Yes        | Yes                | 2×                 |
| 8  | Agro Use     | 83          | 0.79      | —             | —           | Yes        | Yes                | 4×                 |

**Imbalance ratio (Sedan Car / Agro Use): 29.88×**  
**Shannon entropy: 3.063 / 3.700 bits (82.8% of maximum)**  
**Effective number of classes (Hill diversity q=1): varies — see D1 output**

---

## Bounding Box Geometry

### Global Stats (all 10,475 annotations)

| Stat    | Width   | Height  | Area      | Aspect Ratio | Diagonal |
|---------|---------|---------|-----------|--------------|----------|
| Min     | —       | —       | —         | —            | —        |
| p1      | —       | —       | —         | —            | —        |
| p5      | —       | —       | —         | —            | —        |
| Median  | —       | —       | 0.00421   | —            | —        |
| Mean    | —       | —       | 0.00862   | —            | —        |
| p95     | —       | —       | —         | —            | —        |
| p99     | —       | —       | —         | —            | —        |
| Max     | —       | —       | —         | —            | —        |

> Full percentile breakdown is printed by Cell [A6] / [D2] at runtime.

### BBox Size Category Breakdown

| Category | Area Range        | Count | Share (%) |
|----------|-------------------|-------|-----------|
| micro    | < 0.001           | —     | —         |
| small    | 0.001 – 0.01      | —     | —         |
| medium   | 0.01 – 0.05       | —     | —         |
| large    | 0.05 – 0.15       | —     | —         |
| huge     | > 0.15            | —     | —         |

### Tiny-Object Counts (at 640px training scale)

| Check                               | Count  | Share (%) |
|-------------------------------------|--------|-----------|
| Boxes with width < 0.05             | —      | —         |
| Boxes with height < 0.05            | —      | —         |
| Boxes tiny in either dimension      | —      | —         |
| Micro-tiny boxes (area < 0.0005)    | 120    | —         |
| Clipped boundary boxes              | 64     | —         |
| Zero / negative width or height     | 0      | 0.000     |
| NaN in any coordinate               | 0      | 0.000     |
| Out-of-range coordinate [0, 1]      | 0      | 0.000     |

---

## Per-Class BBox Geometry

| Class        | n     | Mean W | Mean H | Mean Area | Median Area | p5 Area | p95 Area | Mean AR | Mean Diag |
|--------------|-------|--------|--------|-----------|-------------|---------|----------|---------|-----------|
| Rickshaw     | 1,422 | —      | —      | —         | —           | —       | —        | —       | —         |
| Motorcycle   | 1,715 | —      | —      | —         | —           | —       | —        | —       | —         |
| Tempu        | 861   | —      | —      | —         | —           | —       | —        | —       | —         |
| Sedan Car    | 2,480 | —      | —      | —         | —           | —       | —        | —       | —         |
| Pickup       | —     | —      | —      | —         | —           | —       | —        | —       | —         |
| Microbus     | 1,120 | —      | —      | —         | —           | —       | —        | —       | —         |
| Mini Bus     | —     | —      | —      | —         | —           | —       | —        | —       | —         |
| Mini Truck   | —     | —      | —      | —         | —           | —       | —        | —       | —         |
| Agro Use     | 83    | —      | —      | —         | —           | —       | —        | —       | —         |
| Medium Truck | —     | —      | —      | —         | —           | —       | —        | —       | —         |
| Large Bus    | 749   | —      | —      | —         | —           | —       | —        | —       | —         |
| Heavy Truck  | —     | —      | —      | —         | —           | —       | —        | —       | —         |
| Trailer      | —     | —      | —      | —         | —           | —       | —        | —       | —         |

> Populated at runtime from EDA cell [A7] output (`cbs` dataframe). Motorcycle has the smallest footprint; Large Bus and Medium Truck have the largest.

---

## EDA → Modeling Decisions

| EDA Signal                           | Key Number                                    | Modeling Decision                                        |
|--------------------------------------|-----------------------------------------------|----------------------------------------------------------|
| Class imbalance                      | 29.88× (Sedan Car vs Agro Use)                | Rare class oversampling + class-aware copy-paste augmentation |
| Small object geometry                | Median box area 0.00421; min width ~0.006     | Image size 1280 + SAHI tiled inference (640×640 slices)  |
| Dense traffic scenes                 | Avg 12.93 objects / image; max 38             | Mosaic (1.0) + MixUp (0.10)                              |
| Box overlap and class similarity     | Trailer / Mini Truck / Microbus / Sedan / Pickup share similar geometry | WBF post-processing + low confidence preservation |
| Shannon entropy below maximum        | 3.063 / 3.700 bits (82.8%)                    | Focal-style rare class weighting                         |
| Clipped boundary boxes               | 64 boxes                                      | `safe_yolo_line` clipping in label conversion            |

---

## Training Configuration

| Parameter               | Value                                              |
|-------------------------|----------------------------------------------------|
| Framework               | Ultralytics YOLOv8                                 |
| Starting weights        | BNVD fine-tuned YOLOv8 (`yolov8x.pt` fallback)    |
| Image size              | 1280                                               |
| Batch size              | 8                                                  |
| Total epochs            | 60                                                 |
| Stage 1 — frozen        | 20 epochs                                          |
| Stage 2 — unfrozen      | 40 epochs                                          |
| Layers frozen (Stage 1) | all except last 10                                 |
| Early stopping patience | 15                                                 |
| Optimizer               | auto                                               |
| AMP                     | True                                               |
| Seed                    | 42                                                 |
| Device                  | GPU [0]                                            |
| Train / val split       | 80% / 20% (random, seed 42)                        |
| Train images            | ~648                                               |
| Val images              | ~162                                               |

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

| Parameter                    | Value              |
|------------------------------|--------------------|
| Base application probability | 0.25               |
| CCTV10 camera probability    | 0.60               |
| Rare class boost             | +0.15 (cap at 0.85)|
| Brightness range             | 0.45 – 0.72        |
| Gamma range                  | 0.60 – 0.85        |
| Gaussian noise std range     | 4 – 14             |
| Blur probability             | 0.4                |
| Blur kernel sizes            | 3×3 or 5×5         |

### Synthetic Data Generation

| Technique                    | Value                              |
|------------------------------|------------------------------------|
| Copy-paste synthetic images  | 200                                |
| Source classes               | Mini Truck (7), Agro Use (8), Trailer (12) |
| Min paste box size           | width ≥ 0.015, height ≥ 0.015     |
| Paste IoU reject threshold   | 0.30                               |
| Paste y-range                | bottom 60% of image                |
| Pastes per synthetic image   | 1 – 2                              |

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

| Property               | Value                                              |
|------------------------|----------------------------------------------------|
| Host                   | Dept. of CSE, DUET, Gazipur, Bangladesh            |
| Partial sponsor        | ICT Division, Bangladesh                           |
| Total prize pool       | 90,000 BDT                                         |
| Max team size          | 3 members                                          |
| Max submissions / day  | 5                                                  |
| Final submissions      | 2 selectable                                       |
| Online phase weight    | 70%                                                |
| Onsite phase weight    | 30%                                                |
| Evaluation metric      | mAP@0.50                                           |
| True positive IoU      | ≥ 0.50                                             |

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

EDA notebook paths (Cell 0):

```python
TRAIN_CSV     = "/path/to/train/train.csv"
TRAIN_IMG_DIR = "/path/to/train/images"
TEST_IMG_DIR  = "/path/to/test/images"
```

Run pipeline cells 0 → 9 sequentially. Cell 10 is optional visualization (16 annotated test images at `conf=0.25`). EDA notebook runs independently, top-to-bottom.

---

## Submission Format

```
image_id,PredictionString
img_0001,0 0.94 0.34 0.50 0.12 0.22 5 0.81 0.20 0.60 0.26 0.14
img_0002,
```

Each detection: `class_id confidence x_center y_center width height` (all coordinates normalized 0–1). Empty `PredictionString` is valid for images with no predictions.

---

## Limitations

| Limitation                    | Impact                                               |
|-------------------------------|------------------------------------------------------|
| 2-day competition window      | No time for deeper hyperparameter search             |
| No external datasets used     | Permitted but unexplored                             |
| Synthetic augmentation        | Cannot fully replace real rare-class variety         |
| SAHI tiled inference          | Improves recall but significantly increases inference time |

---

## Integrity

- No manual labeling of test data
- No leaked labels used at any stage
- Submission fully automated by the pipeline
- Winning code licensed under OSI-approved open source license (per competition rules)
