# Diabetes Patient Readmission Classification

A machine learning project that predicts whether a diabetes patient will be readmitted to the hospital within 30 days, using clinical encounter data.

## Overview

Hospital readmissions for diabetic patients are costly and often preventable. This project builds classification models to identify patients at high risk of early readmission (< 30 days) based on their clinical records. Three classifiers are trained and compared: **Logistic Regression**, **Decision Tree**, and **Random Forest**.

## Dataset

The project uses the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008), stored locally as `diabetic_data.csv`. The dataset contains over 100,000 inpatient diabetes encounters collected from 130 US hospitals between 1999 and 2008.

**Key columns include:**

| Column | Description |
|--------|-------------|
| `patient_nbr` | Unique patient identifier |
| `age` | Age bracket (e.g. `[50-60)`) |
| `race` | Patient's race |
| `gender` | Patient's gender |
| `time_in_hospital` | Number of days admitted |
| `num_medications` | Number of distinct medications administered |
| `num_lab_procedures` | Number of lab tests performed |
| `number_diagnoses` | Number of diagnoses recorded |
| `diag_1/2/3` | Primary, secondary, and additional ICD-9 diagnosis codes |
| `insulin`, `metformin`, … | Medication dosage change indicators |
| `A1Cresult` | HbA1c test result |
| `readmitted` | Target: `<30` (readmitted < 30 days), `>30`, or `NO` |

## Project Structure

```
.
├── main.ipynb          # End-to-end notebook: EDA, preprocessing, modelling, evaluation
├── diabetic_data.csv   # Raw dataset (not included in repo — download separately)
└── README.md
```

## Methodology

### 1. Data Cleaning
- Drop high-missingness columns (`weight`, `payer_code`, `medical_specialty`) and constant-value columns (`citoglipton`, `examide`).
- Remove rows where all three diagnosis codes are missing, or where race/gender is unknown.
- Exclude patients discharged to hospice (`discharge_disposition_id == 11`).

### 2. Feature Engineering
- **Service utilisation** – sum of outpatient, emergency, and inpatient visit counts.
- **Medication change count** (`numchange`) – number of diabetes medications whose dosage was adjusted.
- **Number of active medications** (`nummed`).
- **Diagnosis categorisation** – ICD-9 codes mapped to 8 broad disease groups (`level1_diag`) and 22 sub-groups (`level2_diag`).
- **Interaction terms** – pairwise products between key numeric features (e.g. `num_medications × time_in_hospital`).
- Age brackets converted to decade mid-point values (5, 15, …, 95).

### 3. Preprocessing
- Binary encoding for `gender`, `change`, `diabetesMed`.
- Ordinal encoding for `A1Cresult` and `max_glu_serum`.
- Grouping of sparse `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` categories.
- One-hot encoding of categorical features (`race`, `admission_type_id`, `discharge_disposition_id`, etc.).
- Duplicate patient records removed (first encounter kept).
- Z-score standardisation of numeric features; outliers (|z| > 3) removed.

### 4. Class Imbalance Handling
SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training split to balance the binary readmission classes before model fitting.

### 5. Models & Evaluation

| Model | Notes |
|-------|-------|
| Logistic Regression | L1 penalty (`liblinear` solver) |
| Decision Tree | Entropy criterion, `max_depth=28` |
| Random Forest | 10 estimators, Gini criterion, `max_depth=25` |

Models are evaluated on Accuracy, Precision, Recall, and F1-Score.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
scipy
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy
```

## Usage

1. Download `diabetic_data.csv` from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) and place it in the project root.
2. Launch Jupyter and open the notebook:

```bash
jupyter notebook main.ipynb
```

3. Run all cells from top to bottom. The final cell produces a bar chart comparing the three models across Accuracy, Precision, and Recall. F1-Score is also computed and printed per model during evaluation.
