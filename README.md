# 🚦 Road Accident Risk Prediction

An end-to-end machine-learning project that learns from road, weather, lighting and traffic
conditions to **predict how risky a road situation is**, and explains *which factors drive that risk*
so that road-safety teams can prioritise interventions.

> **Data note:** the results in this repository were produced on a **synthetic demo dataset**
> (20,000 rows, schema-compatible with the Kaggle *Road Accident Risk* data) so that the whole
> pipeline is reproducible out of the box. To get real results, drop your own CSV into
> `data/road_accident_data.csv` (see [Using your own data](#-using-your-own-data)).

---

**Team:** Kshtitij D. Jadhav · Rohit S. Satre · Dipak B. Bhosale · Tanvi B. Sonawane

---

## 📌 Table of contents
1. [Problem statement](#-problem-statement)
2. [Project structure](#-project-structure)
3. [Quick start](#-quick-start)
4. [Pipeline overview](#-pipeline-overview)
5. [Results (demo data)](#-results-demo-data)
6. [Key insights](#-key-insights)
7. [Using your own data](#-using-your-own-data)
8. [Run in Google Colab](#-run-in-google-colab)
9. [Limitations and ethics](#-limitations-and-ethics)
10. [Deliverables](#-deliverables)

## 🎯 Problem statement
Road accidents cause injuries, deaths and large economic losses. Authorities have limited budgets,
so they need to know **where and under which conditions** risk is highest.
**Goal:** build a model that predicts an accident-risk score (0–1) from road and environment
attributes, validate it properly, and turn the findings into actionable recommendations.

**Stakeholders:** traffic police, highway/municipal authorities, emergency-response planners.

## 🗂 Project structure
```
road-accident-risk-prediction/
├── README.md                 <- you are here
├── requirements.txt          <- Python dependencies
├── main.py                   <- runs the whole pipeline end-to-end
├── data/
│   ├── road_accident_data.csv        <- dataset (demo data; replace with real CSV)
│   └── generate_demo_data.py         <- script that created the synthetic demo data
├── src/                      <- modular, reusable code
│   ├── config.py             <- paths, target column, random seed, split size
│   ├── data_loader.py        <- load + inspect data
│   ├── preprocessing.py      <- cleaning + sklearn ColumnTransformer pipeline
│   ├── features.py           <- target detection, feature engineering, train/test split
│   ├── models.py             <- model zoo, training, comparison, best-model selection
│   ├── evaluate.py           <- metrics, cross-validation, feature importance
│   ├── visualization.py      <- every plot (saved to outputs/figures)
│   └── predict.py            <- score a new scenario with the saved model
├── notebooks/
│   └── Road_Accident_Risk_Prediction.ipynb   <- original Colab notebook
├── outputs/                  <- generated: figures/, metrics.json, run_log.txt, model
├── screenshots/              <- images of the executed output of each step
├── report/                   <- project report (Word)
├── presentation/             <- slide deck (PPTX) + speaker script
└── docs/                     <- viva / Q&A preparation
```

## ⚡ Quick start
```bash
git clone https://github.com/<your-username>/road-accident-risk-prediction.git
cd road-accident-risk-prediction

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py            # train, evaluate, save figures + metrics
python -m src.predict     # try the saved model on two example scenarios
```
Outputs appear in `outputs/` (figures, `metrics.json`, `model_comparison.csv`, `best_model.joblib`).

## 🔄 Pipeline overview
| # | Stage | Code | What happens |
|---|-------|------|--------------|
| 1 | Data understanding | `data_loader.py` | shape, dtypes, missing values, duplicates, cardinality |
| 2 | Cleaning | `preprocessing.clean_data` | normalise column names, drop duplicates / ID / constant / empty columns |
| 2b | Outliers | `preprocessing.outlier_summary` | IQR-rule outlier counts and skewness (outliers kept, documented) |
| 3 | EDA | `visualization.py` | distributions, correlation heatmap, category counts, risk-by-category |
| 4 | Features | `features.py` | auto-detect task, date-part extraction, bool→int, drop high-cardinality text |
| 5 | Split | `features.make_split` | 80/20 split, fixed seed (stratified for classification) |
| 6 | Preprocessing | `preprocessing.build_preprocessor` | median-impute + scale numerics; mode-impute + one-hot categoricals (inside a Pipeline → no leakage) |
| 7 | Modelling | `models.py` | Linear Regression (baseline), Random Forest, Gradient Boosting |
| 8 | Validation | `evaluate.py` | MAE / RMSE / R², 5-fold cross-validation, feature importance |
| 9 | Deployment-style use | `predict.py` | load saved pipeline and score a new case |

The pipeline automatically switches to **classification** (Logistic Regression, Random Forest,
Extra Trees; accuracy/precision/recall/F1 + confusion matrix) if the target is categorical or has ≤ 15 distinct values.

## 📊 Results (demo data)
Hold-out test set: 4,000 rows.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|-------|:----:|:-----:|:----:|
| Linear Regression (baseline) | 0.0299 | 0.0373 | 0.9319 |
| Random Forest | 0.0298 | 0.0371 | 0.9327 |
| **Gradient Boosting (best)** | **0.0257** | **0.0319** | **0.9502** |

5-fold cross-validation R² of the best model: **0.9515 ± 0.0029**, consistent with the test score,
so there is no sign of overfitting.

| EDA | Model |
|---|---|
| ![risk](outputs/figures/04_risk_by_category.png) | ![fi](outputs/figures/08_feature_importance.png) |

Risk-band view (scores cut into Low/Medium/High): **86% accuracy, macro F1 0.86**, and no Low case was ever predicted High (see `outputs/figures/06b_risk_band_confusion.png`).

More screenshots are in [`screenshots/`](screenshots/).

## 💡 Key insights
* **Lighting, road curvature and speed limit** are the three most influential factors.
* **Night + sharp curves** together are far worse than either alone (mean risk at night on the sharpest
  curves ≈ 0.72 vs ≈ 0.45 on gentle curves).
* Fog and rain raise risk; road signs are associated with lower risk.
* **Recommendations:** improve lighting on curved rural roads, add advance warning/speed management at
  sharp curves, and target weather-based alerts at foggy periods.

> ⚠️ Demo-data insights illustrate the method; real conclusions require your real dataset.

## 🔧 Using your own data
1. Put your CSV at `data/road_accident_data.csv` (or change `DATA_PATH` in `src/config.py`).
2. Set `TARGET_COL` in `src/config.py` to your outcome column (lower-case, snake_case).
3. Run `python main.py`. Everything else (task type, encoders, plots) adapts automatically.
4. Re-check for **target leakage** (columns known only after the accident).

## ☁️ Run in Google Colab
The original interactive notebook is `notebooks/Road_Accident_Risk_Prediction.ipynb`:
upload it to Colab → *Runtime ▸ Run all* → upload your CSV when asked → type the target column.
To run the modular version in Colab:
```python
!git clone https://github.com/<your-username>/road-accident-risk-prediction.git
%cd road-accident-risk-prediction
!pip install -q -r requirements.txt
!python main.py
```

## ⚖️ Limitations and ethics
* Predictions are **decision support**, not a replacement for emergency services or expert safety audits.
* Models learn correlations in historical data; they do not prove causation.
* Check for class/geographic/time bias and re-validate on newer data before any real deployment.
* Predicted scores are clipped to the valid range [0, 1].

## 📦 Deliverables
| Item | Location |
|------|----------|
| Report | `report/Road_Accident_Risk_Report.docx` |
| Slides | `presentation/Road_Accident_Risk_Presentation.pptx` |
| Speaker script | `presentation/Presentation_Script.md` |
| Viva / Q&A | `docs/QnA_Preparation.md` |
| Code | `src/`, `main.py`, `notebooks/` |

## 📄 License
MIT – free to use for learning and academic work.

**Authors:** Kshtitij D. Jadhav, Rohit S. Satre, Dipak B. Bhosale, Tanvi B. Sonawane
