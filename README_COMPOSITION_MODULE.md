# BAGASSE IQ – AI-Based Image Composition Prediction Module
## Industrial Sugarcane Bagasse Constituents Estimation Platform

This module enhances the **BAGASSE IQ** industrial pulping analytics platform by providing non-invasive optical composition estimation of sugarcane bagasse directly from webcam captures using advanced OpenCV computer vision and XGBoost regression models.

---

## 1. Constituents Estimated

The AI module estimates four core chemical fractions on a dry-weight basis:
1. **Cellulose Percentage (%)**: Primary alpha-cellulose fiber yield indicator (Optimum Kraft range: 40.0% – 48.0%).
2. **Hemicellulose Percentage (%)**: Fiber matrix bonding agent (Optimum Kraft range: 25.0% – 32.0%).
3. **Klason Lignin Percentage (%)**: Primary delignification load determining alkaline chemical charging (Optimum Kraft range: 18.0% – 25.5%).
4. **Ash Percentage (%)**: Mineral & silica content impacting caustic recovery (Industrial range: 1.5% – 4.5%).

> [!IMPORTANT]
> **Scientific Integrity & Empirical Notice**
> Optical surface reflectance features (colour distribution, GLCM texture, edge density) provide surface reflectance proxies calibrated via laboratory wet-chemistry benchmarks (TAPPI T 222, NREL LAP). They do **NOT** directly measure molecular chemical bonds. If laboratory-labelled calibration data is unavailable, models are withheld with the status:
> `COMPOSITION MODEL NOT TRAINED – LAB DATA REQUIRED`

---

## 2. Architecture & File Structure

```
BAGASSE IQ_Python/
├── config.py                                   # Global paths & hardware settings
├── database.py                                 # SQLite schema migrations & tables
├── app.py                                      # Flask web backend & REST APIs
├── hardware/
│   ├── camera.py                               # Webcam controller & frame capture
│   └── esp8266.py                              # ESP8266 3-sensor serial telemetry
├── ml/
│   ├── image_feature_extractor.py              # OpenCV 10-feature extraction module
│   ├── composition_models.py                   # 4-Model XGBoost runtime inference engine
│   ├── train_composition_models.py             # 80:20 grouped split model training pipeline
│   ├── chemical_engine.py                      # Multi-stage Kraft chemical recommendation
│   └── chemical_xgboost_model.py               # Multi-output XGBoost chemical predictor
├── data/
│   └── bagasse_image_composition_dataset_10000.csv  # 10,000-record dataset template
├── dataset/
│   └── bagasse_images/                         # Captured bagasse images tagged by Batch ID
├── models/
│   ├── composition_cellulose_xgb.joblib        # Trained Cellulose XGBoost regressor
│   ├── composition_hemicellulose_xgb.joblib    # Trained Hemicellulose XGBoost regressor
│   ├── composition_lignin_xgb.joblib           # Trained Lignin XGBoost regressor
│   ├── composition_ash_xgb.joblib              # Trained Ash XGBoost regressor
│   ├── composition_scaler_pipeline.joblib      # StandardScaler pipeline
│   ├── composition_evaluation_report.json      # Evaluation metrics (MAE, RMSE, R²)
│   └── composition_evaluation_report.md        # Summary evaluation report
├── scripts/
│   └── generate_composition_dataset_10000.py   # Synthetic dataset generator
└── templates/
    ├── system_camera.html                      # Live webcam stream & reticle preview
    ├── quality_profile.html                    # AI Bagasse Composition Analysis section
    ├── quality_assessment.html                 # 7-stage quality assessment wizard
    └── chemical_recommendations.html           # 8-variable chemical input basis & safety
```

---

## 3. OpenCV Feature Extraction Pipeline (`ml/image_feature_extractor.py`)

The module extracts 10 distinct optical, textural, and chromatic features from each bagasse frame:

| # | Feature | Extraction Method | Pulping Significance |
|---|---------|-------------------|----------------------|
| 1 | **RGB Colour Histogram** | Normalized 8-bin histograms & channel moments (R, G, B means, std, ratios) | Optical reflection baseline & rind-to-pith balance |
| 2 | **HSV Colour Features** | Hue, Saturation, Value channel statistics | Separation of chromatic hue from intensity |
| 3 | **Brown Pixel Ratio (%)** | Calibrated HSV mask (`H: 8-32, S: 35-255, V: 25-220`) | Correlated with fiber oxidation, solar aging & Klason lignin |
| 4 | **Green Pixel Ratio (%)** | Calibrated HSV mask (`H: 35-85, S: 30-255, V: 35-255`) | Chlorophyll / young rind residue detection |
| 5 | **Visual Brightness** | Value (V) channel central ROI mean (0 – 255) | Fiber whiteness and initial reflectance |
| 6 | **RMS Contrast** | Standard deviation of grayscale pixel luminance | Surface texture depth and illumination consistency |
| 7 | **GLCM Texture Features** | Gray-Level Co-occurrence Matrix (16 levels, 4 angles) computing Contrast, Dissimilarity, Homogeneity, Energy, Correlation & Texture Score | Fiber bundle coarseness, depithing efficiency & roughness |
| 8 | **Edge Density** | Canny edge detection pixel count over total ROI pixels | Fiber fragmentation and bundle edge density |
| 9 | **Colour Distribution** | Grayscale histogram Shannon entropy | Chromatic complexity & material homogeneity |
| 10 | **Image Sharpness** | Variance of the Laplacian operator | Optical focus verification & camera distance check |

---

## 4. Execution & Quick-Start Instructions

### Step 1: Generate the 10,000-Record Dataset Template
Generates 10,000 synthetic records with physically consistent values, saves to `data/bagasse_image_composition_dataset_10000.csv`, and generates sample image files in `dataset/bagasse_images/`:
```powershell
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" scripts/generate_composition_dataset_10000.py
```

### Step 2: Train the 4 XGBoost Composition Models
Trains 4 separate single-output XGBoost models using an 80:20 grouped train/test split on `Batch_ID`:
```powershell
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" ml/train_composition_models.py
```
Outputs:
- `models/composition_cellulose_xgb.joblib`
- `models/composition_hemicellulose_xgb.joblib`
- `models/composition_lignin_xgb.joblib`
- `models/composition_ash_xgb.joblib`
- `models/composition_scaler_pipeline.joblib`
- `models/composition_evaluation_report.json`
- `models/composition_evaluation_report.md`

### Step 3: Run the Automated Verification Test Suite
Verifies all components, fallback behaviors, and web routes:
```powershell
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" tests/test_composition_module.py
```

### Step 4: Launch the BAGASSE IQ Application
```powershell
& "C:\Users\vivek\AppData\Local\Programs\Python\Python312\python.exe" app.py
```
Or double-click `run.bat`.

Access the application in your browser at:
`http://localhost:5000`

Default Credentials:
- **Admin**: `admin` / `Admin@123`
- **Operator**: `operator` / `Operator@123`

---

## 5. Dashboard Walkthrough & User Flows

### A. Webcam Capture & Retake (`/camera/view`)
- **Live Preview**: Real-time browser video stream with a dashed center reticle guiding sample distance and illumination.
- **Capture Bagasse Image**: Captures snapshot, stores image under `dataset/bagasse_images/bagasse_<Batch_ID>_<Timestamp>.jpg`.
- **Retake Image**: Resets the canvas and re-enables live optical stream.
- **Analyze Image**: Extracts all 10 OpenCV features and populates the **AI Bagasse Composition** cards in real-time.

### B. Quality Assessment Wizard (`/assessment`)
- 7-stage interactive pipeline fusing the 3 hardware sensors (IR, pH, Temperature) with OpenCV vision.
- Runs the 4-component composition prediction and logs results live in the console.

### C. Bagasse Quality Profile (`/profile/quality`)
- Dedicated **`AI BAGASSE COMPOSITION ANALYSIS`** section.
- Displays the captured bagasse image.
- Image processing status badge (`⚙️ Image Processing: Verified`).
- 4 rich constituent cards with percentage visualizer bars:
  - **Cellulose**: Progress bar against 40% – 48% target.
  - **Hemicellulose**: Progress bar against 25% – 32% target.
  - **Klason Lignin**: Progress bar against 18% – 25.5% target.
  - **Ash**: Progress bar against 1.5% – 4.5% standard.
- Extracted image features sub-panel showing all 10 optical parameters.
- Model validation status pill (`LAB VALIDATION REQUIRED` or `VALIDATED`).
- Data source indicator (`Laboratory` / `Synthetic` / `AI Prediction`).

### D. Chemical Recommendations Integration (`/recommendations/chemical`)
- Displays the **8-Variable Model Input Basis**:
  Initial Lignin, Cellulose, Hemicellulose, Ash, pH, Temperature, Batch Weight (kg), and Target Pulp Quality.
- **Safety Protocol**: Clearly marks unvalidated image predictions as advisory estimates and prevents them from triggering direct automated chemical charging. Operational dosages require Process Engineer authorization.
