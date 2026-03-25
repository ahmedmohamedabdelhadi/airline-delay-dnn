# ✈️ Airline Flight Delay Prediction — Deep Neural Network

> **Predicting the proportion of delayed flights** for a given airline, airport, and month using a full end-to-end deep learning pipeline — from raw BTS data to SHAP-based model explainability.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Summary](#pipeline-summary)
- [Key Results](#key-results)
- [Notebooks](#notebooks)
- [Source Modules](#source-modules)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Project Overview

Flight delays cost the US economy billions of dollars annually and affect millions of passengers. This project builds a **regression model** that predicts the **delay rate** (proportion of flights delayed ≥ 15 minutes) for a given combination of:

- **Airline carrier** (e.g. Delta, United, Southwest)
- **Airport** (420 US airports)
- **Month and Year** (2003–2022)

### What makes this project different

- Uses **aggregated BTS (Bureau of Transportation Statistics)** data — not individual flights
- Carefully avoids **data leakage** by excluding delay-cause breakdown columns from features
- Applies a **temporal train/val/test split** (no future data bleeding into training)
- Includes full **hyperparameter tuning** with Keras Tuner
- Provides **SHAP explainability** so predictions are interpretable

---

## Dataset

| Property | Value |
|---|---|
| Source | [Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) |
| File | `Airline_Delay_Cause.csv` |
| Rows | 318,017 |
| Columns | 21 |
| Time range | January 2003 – December 2022 |
| Granularity | Carrier × Airport × Year × Month |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `year` | int | Year of observation |
| `month` | int | Month (1–12) |
| `carrier` | str | IATA carrier code (e.g. `DL`, `UA`) |
| `carrier_name` | str | Full airline name |
| `airport` | str | IATA airport code (e.g. `JFK`, `LAX`) |
| `airport_name` | str | Full airport name with city/state |
| `arr_flights` | float | Total arriving flights |
| `arr_del15` | float | Flights delayed ≥ 15 minutes |
| `carrier_ct` | float | Flights delayed due to carrier (count) |
| `weather_ct` | float | Flights delayed due to weather (count) |
| `nas_ct` | float | Flights delayed due to National Air System (count) |
| `security_ct` | float | Flights delayed due to security (count) |
| `late_aircraft_ct` | float | Flights delayed due to late arriving aircraft (count) |
| `arr_cancelled` | float | Cancelled flights |
| `arr_diverted` | float | Diverted flights |
| `arr_delay` | float | Total delay minutes for all delayed flights |
| `carrier_delay` | float | Total carrier-caused delay minutes |
| `weather_delay` | float | Total weather-caused delay minutes |
| `nas_delay` | float | Total NAS-caused delay minutes |
| `security_delay` | float | Total security-caused delay minutes |
| `late_aircraft_delay` | float | Total late-aircraft-caused delay minutes |

> **⚠️ Leakage warning:** `carrier_ct`, `weather_ct`, `nas_ct`, `security_ct`, `late_aircraft_ct`, `arr_delay`, `carrier_delay`, `weather_delay`, `nas_delay`, `security_delay`, `late_aircraft_delay` are all **components of the target** and are excluded from model features.

### Target Variable

```
delay_rate = arr_del15 / arr_flights
```

- Continuous value in [0, 1]
- Mean: ~0.198 (≈ 20% of flights are delayed on average)
- Std: ~0.112

---

## Project Structure

```
airline-delay-dnn/
│
├── 📁 data/
│   ├── raw/                    # Original CSV (gitignored)
│   ├── processed/              # Cleaned & engineered features (gitignored)
│   └── external/               # Any supplementary data
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_preprocessing.ipynb  # Scaling, encoding, splits
│   ├── 04_baseline_models.ipynb
│   └── 05_dnn_model.ipynb      # DNN + tuning + SHAP
│
├── 📁 src/
│   ├── __init__.py
│   ├── data_loader.py          # Load & validate raw data
│   ├── feature_engineering.py  # All feature creation logic
│   ├── preprocessing.py        # Scalers, encoders, split logic
│   ├── model.py                # DNN architecture & training
│   └── evaluate.py             # Metrics, plots, SHAP wrapper
│
├── 📁 models/
│   ├── checkpoints/            # Keras training checkpoints
│   └── saved/                  # Final saved models (.keras)
│
├── 📁 reports/
│   └── figures/                # All saved plots (PNG/HTML)
│
├── 📁 tests/
│   └── test_features.py        # Unit tests for feature engineering
│
├── .gitignore
├── requirements.txt
└── README.md                   # ← You are here
```

---

## Pipeline Summary

```
Raw CSV
   │
   ▼
01_EDA ──────────────── Univariate · Bivariate · Temporal · Geographic analysis
   │
   ▼
02_Feature Engineering ─ delay_rate target · cyclic time features · carrier stats
   │                      cancel/divert ratios · route-level aggregations
   ▼
03_Preprocessing ──────── Temporal train/val/test split (2003–18 / 19–20 / 21–22)
   │                       StandardScaler · LabelEncoder · missing value imputation
   ▼
04_Baseline Models ─────── Linear Regression · Ridge · Random Forest · XGBoost
   │
   ▼
05_DNN Model ──────────── Architecture search (Keras Tuner) · BatchNorm · Dropout
   │                       EarlyStopping · LR scheduling · Final evaluation
   ▼
SHAP Explainability ────── Global feature importance · Local prediction breakdown
```

---

## Key Results

> Results will be filled in after running the full pipeline.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 06468810029846998 | 0944137601653098 | 330017791702639 |
| Ridge Regression | 06468810028658088 | 09441376015731269 | 3300177918161379 |
| Random Forest | 06465935678750134 | 09378616925337432 | 3388952529139023 |
| XGBoost | 06334353419312704 | 09299085401211328 | 3500601682989112 |
| **DNN (tuned)** | **06165146827697754** | **09757318301700893** | **2844275236129761** |

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_EDA.ipynb` | Deep exploratory analysis: distributions, correlations, temporal trends, carrier/airport rankings, missing value patterns |
| `02_feature_engineering.ipynb` | Create delay_rate target, cyclic month encoding, carrier/airport historical stats, cancellation ratio, volume bins |
| `03_preprocessing.ipynb` | Temporal split, imputation, scaling, categorical encoding, final feature matrix saved to `data/processed/` |
| `04_baseline_models.ipynb` | Train Linear Regression, Ridge, Random Forest, and XGBoost; compare with cross-validation |
| `05_dnn_model.ipynb` | Build DNN with Keras, hyperparameter tuning, training curves, test evaluation, SHAP analysis |

---

## Source Modules

| Module | Purpose |
|---|---|
| `src/data_loader.py` | `load_raw_data()` — loads CSV, validates schema, reports quality |
| `src/feature_engineering.py` | `build_features()` — all feature creation, returns clean DataFrame |
| `src/preprocessing.py` | `make_splits()`, `scale_features()` — reproducible preprocessing |
| `src/model.py` | `build_dnn()`, `train_model()` — architecture definition and training loop |
| `src/evaluate.py` | `evaluate_model()`, `plot_shap()` — metrics, residual plots, SHAP |

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/airline-delay-dnn.git
cd airline-delay-dnn
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the data

Download `Airline_Delay_Cause.csv` from [BTS](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) and place it in:

```
data/raw/Airline_Delay_Cause.csv
```

### 5. Run notebooks in order

```bash
jupyter notebook
```

Open and run notebooks `01` → `02` → `03` → `04` → `05` in sequence.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.13+-red?logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas)
![SHAP](https://img.shields.io/badge/SHAP-0.42+-brightgreen)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## Author

**Ahmed Mohamed Abdelhady**

- GitHub: [@ahmedmohamedabdelhadi](<https://github.com/ahmedmohamedabdelhadi>)
- Kaggle: (<https://www.kaggle.com/ahmedmohamedabdelhadi>)
- LinkedIn: [ahmed-mohamed-abdelhady](<https://linkedin.com/in/ahmed-mohamed-abdelhady/>)

---

*This project was built as a portfolio demonstration of a professional machine learning pipeline — from raw data exploration to deep learning and model explainability.*
