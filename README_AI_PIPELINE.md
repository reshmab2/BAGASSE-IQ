# BAGASSE IQ – AI Model Training & Adaptive Prediction Pipeline
==============================================================

## 1. Overview
**BAGASSE IQ** is an industrial cyber-physical decision support system for sugarcane bagasse quality assessment and adaptive Kraft pulping optimization. The platform incorporates a 5-stage, interconnected machine learning pipeline combining computer vision, chemical stoichiometry, multi-output XGBoost regression, and Bayesian parameter optimization.

---

## 2. The 5 Machine Learning Models

### Model 1: Bagasse Image Analysis (`ml_models/image_feature_extraction.py`)
- **Engine**: Python OpenCV (vectorized pure NumPy without skimage dependencies).
- **Features Extracted**:
  - RGB Channel Moments & Histograms
  - HSV Channel Moments & Histograms
  - Brown Pixel % (Lignin oxidation / aging proxy)
  - Green Pixel % (Chlorophyll / fresh rind residue proxy)
  - Image Brightness (Mean ROI luminance) & RMS Contrast
  - Gray-Level Co-occurrence Matrix (GLCM): Contrast, Dissimilarity, Homogeneity, Energy, Correlation, Composite Texture Score
  - Edge Density (Canny edge ratio) & Image Sharpness (Laplacian variance)
  - Sample Quality Status (`OPTIMAL`, `LOW_LIGHT`, `OVEREXPOSED`, `BLURRED`)

### Model 2: Bagasse Composition Predictor (`ml_models/train_composition_model.py`)
- **Engine**: 4 Separate XGBoost Regressors.
- **Targets**:
  - Cellulose Percentage (%) [Biological range: 38% – 52%]
  - Hemicellulose Percentage (%) [Biological range: 22% – 35%]
  - Lignin Percentage (%) [Biological range: 16% – 28%]
  - Ash Percentage (%) [Biological range: 1% – 6%]
- **Provenance**: `data/image_composition_dataset.csv` (10,000 records).

### Model 3: Bagasse Quality Classification & Scoring (`ml_models/train_quality_model.py`)
- **Engine**: XGBoost Regressor + XGBoost Classifier.
- **Targets**:
  - Continuous Quality Score (0 – 100)
  - Discrete Quality Category (`Premium`, `Standard`, `Degraded`)
- **Inputs**: Composition, Hardware sensors (pH, Temperature), Visual brightness, Texture, Sharpness.
- **Provenance**: `data/bagasse_quality_dataset.csv` (10,000 records).

### Model 4: Chemical Recommendations & Bayesian Optimizer (`ml_models/train_chemical_model.py`)
- **Engine**: Multi-output XGBoost Regressors + Gaussian Process Optimization (`scipy.optimize` / `scikit-learn`).
- **Predicted Parameters**:
  - Active Alkali (% NaOH equivalent on dry bagasse wt: 12% – 20%)
  - Sulfidity (% Na₂S of Active Alkali: 18% – 30%)
  - NaOH quantity (kg/ton dry bagasse)
  - Na₂S quantity (kg/ton dry bagasse)
  - Water requirement (L/ton) & Liquor-to-Bagasse Ratio (3.5:1 – 5.5:1)
  - Cooking Temperature (155°C – 175°C) & Cooking Time (45 – 90 minutes)
- **Bayesian Tuning**: Minimizes chemical consumption cost and digester steam energy while hitting target delignification Kappa targets.

### Model 5: Pulp Quality & Yield Prediction (`ml_models/train_pulp_model.py`)
- **Engine**: Multi-Target XGBoost Regressors.
- **Targets**:
  - Expected Pulp Quality Score (0 – 100)
  - Predicted Residual Lignin % (1.5% – 6.5%) & Estimated Kappa Number ($\text{Kappa} = \text{Residual Lignin} \times 6.55$)
  - Expected Pulp Yield % (40% – 56% dry unbleached pulp recovery)
- **Provenance**: `data/pulp_quality_dataset.csv` (10,000 records).

---

## 3. The 4 Synthetic Datasets (10,000 Records Each)

All datasets are located in `data/` and strictly tagged with `Data_Source = 'SYNTHETIC'`:
1. `data/image_composition_dataset.csv` (Vision features $\rightarrow$ Composition)
2. `data/bagasse_quality_dataset.csv` (Composition + Sensors + Vision $\rightarrow$ Quality Score & Grade)
3. `data/chemical_recommendation_dataset.csv` (Raw inputs $\rightarrow$ Kraft chemical recipe)
4. `data/pulp_quality_dataset.csv` (Chemical & Cooking parameters $\rightarrow$ Downstream Pulp Quality & Yield)

### Re-generating All Datasets:
```powershell
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" scripts/generate_all_datasets_10000.py
```

---

## 4. Model Training & Evaluation Commands

To train all models sequentially:
```powershell
# Model 2: Composition
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" ml_models/train_composition_model.py

# Model 3: Quality
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" ml_models/train_quality_model.py

# Model 4: Chemical Recommendations & Bayesian Optimization
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" ml_models/train_chemical_model.py

# Model 5: Pulp Quality
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" ml_models/train_pulp_model.py
```

All models save their serialized pipelines (`*.joblib`), evaluation reports (`*.json`), and Markdown summaries in `models/`.

---

## 5. Model Validation System & Regulatory Integrity

Every model card, API payload, and dashboard display carries a mandatory status pill:

| Validation Status Tag | Trigger Condition | Operational Implication |
| :--- | :--- | :--- |
| `SYNTHETIC MODEL – REAL-WORLD VALIDATION REQUIRED` | Trained on synthetic dataset records (`Data_Source = 'SYNTHETIC'`) | Strictly advisory estimates. Formal process engineer sign-off required prior to digester charging. |
| `MODEL NOT TRAINED – LAB DATA REQUIRED` | Model artifacts missing or uncalibrated | Never invents numbers; prompts operator to calibrate or train model. |
| `VALIDATED` | Calibrated on certified laboratory wet-chemistry assays (`Data_Source = 'LABORATORY'`) | Certified for automated operational recipe guidance. |

### Laboratory Wet-Chemistry Import:
To promote model calibration from Synthetic to Validated:
```powershell
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" scripts/import_laboratory_data.py <path_to_lab_assay.csv>
```
Or upload via the Web UI under `/models` $\rightarrow$ **Import Laboratory Wet-Chemistry Data**.

---

## 6. Apache Spark & Scala Distributed Analytics

Tested on **Apache Spark 4.2.0, Scala 2.13.18, Java 17**:
- `analytics/ChemicalDataProcessor.scala`: Mass balance, non-negativity checks, stoichiometric aggregations.
- `analytics/BatchAnalyticsProcessor.scala`: Historical temporal analysis, sensor drift detection, delignification correlation matrices.
- `analytics/run_spark_analytics.cmd`: Interactive Windows launcher.

Run via Spark:
```cmd
cd analytics
run_spark_analytics.cmd
```

---

## 7. Running the Full Application & Verification Suite

### Launch Verification Suite:
```powershell
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" -m unittest tests/test_full_ai_pipeline.py
```

### Start Web Server:
```powershell
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" app.py
```
Open [http://localhost:5000](http://localhost:5000) in browser:
- **Model Management Hub**: `/models`
- **Chemical Recommendations**: `/recommendations/chemical`
- **Bagasse Quality Profile**: `/profile/quality`
- **Webcam Preview & Capture**: `/camera/view`
