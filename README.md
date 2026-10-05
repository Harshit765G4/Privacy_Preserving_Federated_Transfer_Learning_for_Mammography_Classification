# 🩺 Privacy-Preserving Federated Transfer Learning for Mammography Classification

A research-stage deep-learning project for **mammography image classification** with a planned privacy-preserving federated-learning workflow.

The current repository focuses on building a clean, patient-aware mammography dataset and a transferable **ResNet-50** classification baseline. The federated-learning and differential-privacy components are explicitly marked as future work in the project configuration and TODO notes.

> **Research status:** This repository is not yet a complete federated-learning system. Dataset preparation and centralized transfer-learning components are implemented; Flower/FedAvg/FedProx/SCAFFOLD and DP-SGD are currently planned.

## 🎯 Project Objective

The project is designed around the following research direction:

```text
Mammography Data
      │
      ▼
Metadata Verification
      │
      ▼
Patient-aware Dataset Construction
      │
      ▼
Image Preprocessing
(CLAHE + 512×512)
      │
      ▼
Patient-wise Stratified Folds
      │
      ▼
Pretrained ResNet-50
      │
      ▼
Mammography Classification
      │
      └─────────────── Future ───────────────┐
                                            ▼
                              Federated Learning
                              + Privacy Protection
```

The intended research extension is to train across distributed clinical clients without centrally sharing raw mammography images.

## ✅ Current Project Status

According to the repository's status and research notes:

### Completed

- Dataset download
- CSV verification
- Metadata investigation
- Duplicate `SeriesInstanceUID` investigation
- Missing `SeriesDescription` investigation
- Image-dimension validation
- Master-label construction utilities
- Lesion-to-mammogram aggregation
- CLAHE preprocessing
- Patient-wise 5-fold splitting implementation
- Final training-dataset construction
- ResNet-50 transfer-learning model definition
- PyTorch dataset and training utilities
- Classification metrics
- Basic model/data unit checks

### Planned / Not Yet Implemented

- Flower integration
- FedAvg
- FedProx
- SCAFFOLD
- Differentially private SGD
- Federated statistical validation
- Final end-to-end evaluation
- Research paper packaging

These planned items are listed in `TODO.md` and the central configuration.

## 🗃️ Dataset Preparation

The repository contains metadata processing code rather than the original mammography image corpus.

The pipeline is designed around mammography metadata and image paths, with the processed training dataset expected at:

```text
data/processed/training_dataset.csv
```

The repository currently includes a processed metadata CSV:

```text
data/processed_meta_check.csv
```

### Metadata handling

The research scripts investigate:

- patient identifiers
- image paths
- breast side
- image view
- breast density
- pathology
- abnormality type
- assessment/subtlety
- dataset splits
- DICOM-derived metadata

A key design choice recorded in `RESEARCH_NOTES.md` is to **retain images with missing `SeriesDescription` when their dimensions are consistent with full mammograms**, rather than automatically discarding them.

Another important decision is to avoid treating duplicate `SeriesInstanceUID` values as a universally unique image key because ROI/crop records can reuse identifiers.

## 🧬 Mammogram-Level Label Construction

The repository includes `src/preprocessing/build_mammogram_dataset.py`.

The script groups lesion-level records by `image_path` and constructs a mammogram-level classification target.

Label logic:

```text
if any lesion in the mammogram is MALIGNANT:
    label = 1
else:
    label = 0
```

`BENIGN_WITHOUT_CALLBACK` is normalized to `BENIGN`.

The resulting dataset contains fields such as:

- `patient_id`
- `image_path`
- `processed_image_path`
- `pathology`
- `label`
- `breast_side`
- `image_view`
- `breast_density`
- `dataset_type`
- `dataset_split`
- `assessment`
- `subtlety`
- `abnormality_count`

## 🖼️ CLAHE Image Preprocessing

Implemented in `src/preprocessing/clahe.py`.

For each source mammogram:

1. Load the image as grayscale.
2. Resize it to **512 × 512**.
3. Apply OpenCV CLAHE.
4. Save the processed image as PNG.
5. Record processing statistics.

Current CLAHE parameters:

```text
clipLimit     = 3.0
tileGridSize  = (8, 8)
output size   = 512 × 512
```

The script also records:

- original dimensions
- mean before/after
- standard deviation before/after
- processing time
- success/failure status

Outputs include:

```text
data/processed/images_clahe/
data/processed/master_labels_clahe.csv
data/processed/clahe_processing_report.csv
```

## 👩‍⚕️ Patient-Wise Cross-Validation

Implemented in `src/preprocessing/create_patient_folds.py`.

The repository uses:

```text
StratifiedGroupKFold
n_splits = 5
shuffle = True
random_state = 42
```

Grouping is performed by `patient_id`, while stratification uses the binary mammography label.

This is important because multiple mammograms from the same patient must not be unintentionally distributed across training and validation folds.

The generated fold dataset is:

```text
data/processed/mammogram_dataset_with_folds.csv
```

The script also performs a sanity check for patients appearing in multiple folds.

## 🧩 Training Dataset Construction

`src/preprocessing/create_training_dataset.py` connects fold metadata with processed CLAHE images.

It:

- builds the expected processed-image filename
- checks whether the CLAHE image exists
- keeps only records with available processed images
- writes the final training dataset
- records missing-image cases separately

Outputs:

```text
data/processed/training_dataset.csv
data/processed/missing_processed_images.csv
```

## 🧠 ResNet-50 Transfer Learning

The current classification model is implemented in:

```text
src/models/resnet50.py
```

The model uses pretrained torchvision **ResNet-50** weights.

### Fine-tuning strategy

Initially, the backbone is frozen.

Then the implementation unfreezes:

```text
ResNet layer4
```

The original fully connected classifier is replaced with:

```text
Linear(2048 → 512)
        ↓
ReLU
        ↓
Dropout(0.5)
        ↓
Linear(512 → 2)
```

The output corresponds to two classes.

## 🖼️ Data Augmentation

Training transforms in `src/utils/transforms.py`:

- resize to **512 × 512**
- random horizontal flip with probability **0.5**
- random rotation up to **10°**
- conversion to tensor
- ImageNet normalization

Validation transforms:

- resize to 512 × 512
- tensor conversion
- ImageNet normalization

Using ImageNet statistics is consistent with the pretrained ResNet-50 initialization.

## 🏋️ Training Engine

The main training engine is in:

```text
src/training/trainer.py
```

The trainer supports:

- mini-batch training
- validation
- metric calculation
- learning-rate scheduling
- best-checkpoint selection based on validation AUC
- history collection
- restoration of the best model weights

During validation it calculates prediction probabilities and class predictions, then passes them to the project metric module.

## ⚙️ Current Configuration

Central configuration is in `config/config.py`.

| Setting | Current value |
|---|---:|
| Image size | 512 × 512 |
| Channels | 3 |
| Classes | 2 |
| Batch size | 16 |
| Workers | 4 |
| Epochs | 30 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| CV folds | 5 |
| Seed | 42 |
| Model | ResNet-50 |
| Pretrained | Yes |
| Early-stopping patience | 7 |

The configured device is automatically selected as:

```text
CUDA if available
otherwise CPU
```

## 📊 Evaluation Metrics

Implemented in `src/training/metrics.py`.

The project calculates:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Specificity
- Sensitivity
- Confusion matrix components

This is particularly relevant for medical classification, where accuracy alone is insufficient.

## 🔐 Planned Privacy-Preserving Federated Learning

The configuration reserves a federated-learning section:

```text
NUM_CLIENTS  = 3
LOCAL_EPOCHS = 2
ROUNDS       = 20
```

These values are currently **configuration placeholders**, not evidence of completed federated training.

The TODO roadmap explicitly lists:

### FedAvg

Federated averaging is planned as the initial aggregation baseline.

### FedProx

Planned to address client heterogeneity and stabilize local optimization when client distributions differ.

### SCAFFOLD

Planned to reduce client-drift effects using control variates.

## 🛡️ Planned Differential Privacy

The central configuration also includes future DP parameters:

```text
TARGET_EPSILON = 1.9
TARGET_DELTA   = 1e-5
MAX_GRAD_NORM  = 1.0
```

These are **target configuration values only** at the current repository stage.

A complete privacy-preserving implementation would need to actually:

1. clip per-sample gradients
2. add calibrated noise
3. track privacy expenditure
4. report the resulting ((epsilon,delta))
5. validate the accuracy/privacy trade-off

## 📁 Repository Structure

```text
Privacy_Preserving_Federated_Transfer_Learning_for_Mammography_Classification/
│
├── config/
│   └── config.py
│
├── data/
│   └── processed_meta_check.csv
│
├── research/
│   └── exploratory/
│       ├── Diagnostic.py
│       ├── dicom_info_coloumn_name.py
│       ├── image_Dimensions.py
│       ├── mass_case_column_name.py
│       ├── path_Implementation.py
│       └── verify_Data.py
│
├── src/
│   ├── datasets/
│   │   └── mammogram_dataset.py
│   │
│   ├── evaluation/
│   │   └── evaluate.py
│   │
│   ├── models/
│   │   └── resnet50.py
│   │
│   ├── preprocessing/
│   │   ├── analyze_duplicate_annotations.py
│   │   ├── build_mammogram_dataset.py
│   │   ├── clahe.py
│   │   ├── create_master_labels.py
│   │   ├── create_patient_folds.py
│   │   ├── create_training_dataset.py
│   │   ├── investigate_conflicting_labels.py
│   │   ├── map_jpeg_paths.py
│   │   ├── merge_case_csv.py
│   │   ├── validate_images.py
│   │   └── verify_full_images.py
│   │
│   ├── training/
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── train.py
│   │   └── trainer.py
│   │
│   └── utils/
│       └── transforms.py
│
├── PROJECT_STATUS.md
├── RESEARCH_NOTES.md
├── TODO.md
├── requirements.txt
├── test_config.py
├── test_dataset.py
├── test_metrics.py
└── test_model.py
```

## 🧪 Basic Verification

The repository includes lightweight verification scripts.

### Configuration

```bash
python test_config.py
```

### Dataset / DataLoader

```bash
python test_dataset.py
```

### Model forward pass

```bash
python test_model.py
```

This creates a dummy tensor with shape:

```text
(batch=2, channels=3, height=512, width=512)
```

and verifies the two-class output shape.

### Metrics

```bash
python test_metrics.py
```

The repository also includes `pytest` in the dependency environment, although not all test files are written as conventional pytest test cases.

## 📦 Installation

The repository currently uses a large pinned `requirements.txt` generated from a broader Python environment rather than a minimal project dependency file.

Core libraries used by the current mammography pipeline include:

- PyTorch
- torchvision
- pandas
- NumPy
- scikit-learn
- Pillow
- OpenCV
- tqdm
- matplotlib

Install the repository environment with:

```bash
pip install -r requirements.txt
```

For a cleaner research environment, a future improvement would be to maintain a minimal project-specific requirements file and a separate lock/export file.

## ⚠️ Current Limitations

- Federated learning is not implemented yet.
- There is no Flower server/client implementation in the repository.
- FedAvg, FedProx, and SCAFFOLD are planned but not executable from the current codebase.
- Differentially private training is configured conceptually but not implemented.
- `src/evaluation/evaluate.py` is currently empty.
- `src/training/train.py` and `src/training/losses.py` are currently empty placeholders.
- The repository contains exploratory scripts documenting dataset verification decisions.
- The mammography image corpus itself is not included in the repository; paths are resolved through generated metadata.
- Some configuration directories are created automatically at import time.
- No final research benchmark or privacy/utility comparison is stored in the repository yet.

## 🔬 Recommended Research Roadmap

A practical next sequence would be:

```text
1. Finalize master_labels.csv
        ↓
2. Generate CLAHE dataset
        ↓
3. Validate patient-wise folds
        ↓
4. Establish centralized ResNet-50 baseline
        ↓
5. Implement Flower clients/server
        ↓
6. Add FedAvg baseline
        ↓
7. Add FedProx + SCAFFOLD
        ↓
8. Add DP-SGD
        ↓
9. Compare centralized vs FL vs DP-FL
        ↓
10. Statistical significance + confidence intervals
        ↓
11. Privacy / utility trade-off analysis
        ↓
12. Paper-ready experiments
```

Recommended reporting should include at least:

- Accuracy
- Precision
- Recall / sensitivity
- Specificity
- F1
- ROC-AUC
- confusion matrices
- per-client metrics
- communication rounds
- local epochs
- client participation
- convergence curves
- privacy budget ((epsilon,delta))
- communication cost
- model size

## 🩻 Research Safety

This project concerns medical imaging and should be treated as a research/engineering system rather than a clinical diagnostic tool.

Model performance should not be interpreted as medical validation without appropriate dataset governance, external validation, clinical review, statistical analysis, and regulatory considerations.

## 👤 Author

**Harshit Garg**

GitHub: [Harshit765G4](https://github.com/Harshit765G4)

---

A research foundation for combining **mammography transfer learning, patient-aware validation, federated learning, and privacy-preserving training**.
