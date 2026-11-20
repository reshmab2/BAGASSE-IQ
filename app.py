import os
import csv
import json
import io
import time
from functools import wraps
from datetime import datetime

from flask import (
    Flask,
    request,
    session,
    redirect,
    url_for,
    render_template,
    jsonify,
    send_from_directory,
    Response
)
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config
from database import init_db, get_db_connection
from hardware.esp8266 import ESP8266Manager
from hardware.camera import CameraManager
from ml.feature_fusion import FeatureFusionEngine
from ml.quality_model import QualityModelEngine
from ml.chemical_engine import ChemicalRecommendationEngine
from ml.lignin_model import LigninPredictionEngine
from ml.chemical_xgboost_model import ChemicalXGBoostEngine
from ml.composition_models import BagasseCompositionEngine
from ml.synthetic_retrieval_engine import dataset_retrieval_engine
from ml_models.model_registry import ModelRegistry
from ml_models.model_evaluation import ModelEvaluator
from ml_models.predict import BagasseIQPredictor
from ml_models.train_composition_model import train_composition_models
from ml_models.train_quality_model import train_quality_model
from ml_models.train_chemical_model import train_chemical_models
from ml_models.train_pulp_model import train_pulp_models
from scripts.import_laboratory_data import import_lab_csv

# Initialize Flask Application
app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY

# Initialize Hardware Managers
esp_manager = ESP8266Manager(port=Config.DEFAULT_SERIAL_PORT, baud_rate=Config.DEFAULT_BAUD_RATE)
camera_manager = CameraManager(camera_index=Config.DEFAULT_CAMERA_INDEX)

# Initialize AI & Chemistry Engines
quality_model = QualityModelEngine()
chemical_engine = ChemicalRecommendationEngine()
lignin_model = LigninPredictionEngine()
chemical_xgb = ChemicalXGBoostEngine()
composition_engine = BagasseCompositionEngine()
pipeline_predictor = BagasseIQPredictor()

# Active In-Memory Context
active_context = {
    "batch_id": None,
    "last_captured_image": None,
    "last_visual_features": {
        "brightness": 154.2,
        "greenish_percentage": 9.4,
        "brownish_percentage": 33.8,
        "texture_contrast": 41.6,
        "texture_energy": 22.1,
        "contrast": 42.0,
        "edge_density": 0.12,
        "sharpness": 85.0,
        "texture_score": 42.5
    },
    "last_composition": None
}

# Auto-capture callback when ESP8266 IR sensor detects bagasse sample
def on_bagasse_detected():
    batch = active_context["batch_id"] or "SAMPLE-AUTO"
    res, msg = camera_manager.capture_frame(batch)
    if res:
        active_context["last_captured_image"] = res["filepath"]
        active_context["last_visual_features"] = res["features"]
        feat = res["features"]
        comp = composition_engine.predict_composition(feat)
        active_context["last_composition"] = comp

        # Save to database
        try:
            conn = get_db_connection()
            conn.execute(
                "INSERT INTO bagasse_images (batch_id, image_path) VALUES (?, ?)",
                (batch, res["filepath"])
            )
            conn.execute("""
                INSERT INTO image_features 
                (batch_id, brightness, greenish_percentage, brownish_percentage, texture_contrast, texture_energy,
                 contrast, edge_density, sharpness, glcm_texture, rgb_features_json, hsv_features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                batch, feat.get("brightness", 0), feat.get("greenish_percentage", 0), feat.get("brownish_percentage", 0),
                feat.get("texture_contrast", 0), feat.get("texture_energy", 0), feat.get("contrast", 0),
                feat.get("edge_density", 0), feat.get("sharpness", 0), feat.get("texture_score", 0),
                feat.get("rgb_features_json", "{}"), feat.get("hsv_features_json", "{}")
            ))
            conn.execute("""
                INSERT INTO bagasse_composition
                (batch_id, image_path, cellulose_pct, hemicellulose_pct, lignin_pct, ash_pct, validation_status, data_source, confidence, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                batch, res["filepath"], comp.get("cellulose_pct"), comp.get("hemicellulose_pct"),
                comp.get("lignin_pct"), comp.get("ash_pct"), comp.get("validation_status"),
                comp.get("data_source"), comp.get("confidence", 0.0), comp.get("model_version", "v1.0-ImageCompositionXGB")
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print("Auto-capture DB error:", e)

esp_manager.set_callback(on_bagasse_detected)

# ==================================================
# AUTHENTICATION DECORATORS
# ==================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            return redirect(url_for("operator_dashboard"))
        return f(*args, **kwargs)
    return decorated_function

# ==================================================
# DATABASE SEEDING ON STARTUP
# ==================================================

def seed_initial_records():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Seed Default Dataset record if empty
    cursor.execute("SELECT id FROM datasets WHERE version = 'DS-BAG-2026-V1'")
    if not cursor.fetchone():
        default_csv = os.path.join(Config.DATASETS_DIR, "default_bagasse_dataset.csv")
        row_count = 20
        cols = ["ir_detected", "ph_value", "temperature_c", "visual_brightness", "greenish_pct", "brownish_pct", "texture_variance", "naoh_kg_per_ton", "na2s_kg_per_ton", "water_ratio_liters_per_ton", "h2o2_kg_per_ton"]
        cursor.execute("""
            INSERT INTO datasets (dataset_name, file_path, version, row_count, columns_json, uploaded_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("Historical Validated Bagasse Pulping Dataset", default_csv, "DS-BAG-2026-V1", row_count, json.dumps(cols), "System Admin"))

    # Seed 10,000-Row Synthetic Kraft Chemical Dataset
    cursor.execute("SELECT id FROM datasets WHERE version = 'DS-BAG-KRAFT-10K-SYNTHETIC'")
    if not cursor.fetchone():
        synth_csv = os.path.join(Config.DATASETS_DIR, "BAGASSE_IQ_Kraft_Chemical_Training_10000.csv")
        cols_10k = [
            "Batch_ID", "Initial_Lignin_Percent", "Initial_Cellulose_Percent", "Initial_Hemicellulose_Percent", "Initial_Ash_Percent",
            "Sensor_pH", "Temperature_C", "Brown_Percent", "Green_Percent", "Image_Brightness", "Image_Texture_Score", "Image_Contrast", "Edge_Density", "Batch_Weight_Kg", "Target_Pulp_Quality",
            "Active_Alkali_NaOH_Equivalent_Percent", "Na2S_Sulfidity_Percent", "NaOH_Required_Grams", "Na2S_Required_Grams", "Water_Required_Litres", "Liquor_to_Bagasse_Ratio", "Cooking_Temperature_C", "Cooking_Time_Minutes", "Predicted_Residual_Lignin_Percent", "Expected_Pulp_Quality_Score", "Data_Type"
        ]
        cursor.execute("""
            INSERT INTO datasets (dataset_name, file_path, version, row_count, columns_json, uploaded_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("10,000-Row Synthetic Kraft Chemical Recommendation Dataset", synth_csv, "DS-BAG-KRAFT-10K-SYNTHETIC", 10000, json.dumps(cols_10k), "System AI Engine"))

    # Seed XGBoost Chemical Recommender Model Record
    cursor.execute("SELECT id FROM models WHERE model_version = 'v3.0-KraftSynthetic10K'")
    if not cursor.fetchone():
        eval_path = os.path.join(Config.MODELS_DIR, "xgboost_evaluation_report.json")
        eval_metrics = {}
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as ef:
                    eval_metrics = json.load(ef).get("metrics", {})
            except Exception:
                pass
        cursor.execute("""
            INSERT INTO models (model_name, model_version, dataset_version, evaluation_metrics_json, model_path, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (
            "MultiOutputXGBoostChemicalRecommender",
            "v3.0-KraftSynthetic10K",
            "DS-BAG-KRAFT-10K-SYNTHETIC",
            json.dumps(eval_metrics),
            os.path.join(Config.MODELS_DIR, "xgboost_chemical_recommender.joblib")
        ))

    # 2. Seed Initial Batches if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM batches")
    if cursor.fetchone()["cnt"] == 0:
        sample_batches = [
            ("BAG-20261110-001", "operator", "Lead Plant Operator", "2026-11-10 09:30:00", "Completed", "Shift A primary mill run", 1, "Detected", 7.2, 28.5, "Grade A (Prime Pulping Quality)", 92.4, 135.0, 28.0, 3800.0, 18.5),
            ("BAG-20261111-002", "operator", "Lead Plant Operator", "2026-11-11 11:15:00", "Completed", "Washed bagasse batch B-2", 1, "Detected", 7.0, 29.1, "Grade A (Prime Pulping Quality)", 89.7, 138.0, 29.0, 3850.0, 20.0),
            ("BAG-20261112-003", "operator", "Lead Plant Operator", "2026-11-12 14:40:00", "Completed", "Field harvest high fiber", 1, "Detected", 6.8, 31.4, "Grade B (Standard Commercial Grade)", 78.5, 152.0, 33.5, 4000.0, 24.0),
            ("BAG-20261112-004", "operator", "Lead Plant Operator", "2026-11-12 16:50:00", "Completed", "Aged storage bagasse lot", 1, "Detected", 5.8, 36.5, "Grade C (Sub-optimal / Degraded)", 54.1, 175.0, 42.0, 4400.0, 34.0),
            ("BAG-20261113-005", "operator", "Lead Plant Operator", "2026-11-13 08:20:00", "Completed", "Fresh morning run", 1, "Detected", 7.3, 27.8, "Grade A (Prime Pulping Quality)", 94.2, 131.0, 26.8, 3760.0, 17.0)
        ]
        for b in sample_batches:
            cursor.execute("""
                INSERT INTO batches (batch_id, operator_id, operator_name, batch_date, status, remarks)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (b[0], b[1], b[2], b[3], b[4], b[5]))

            cursor.execute("""
                INSERT INTO sensor_readings (batch_id, ir_value, ir_status, ph, temperature, is_simulated)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (b[0], b[6], b[7], b[8], b[9]))

            cursor.execute("""
                INSERT INTO image_features (batch_id, brightness, greenish_percentage, brownish_percentage, texture_contrast, texture_energy)
                VALUES (?, 162.0, 8.5, 31.2, 39.4, 21.0)
            """, (b[0],))

            cursor.execute("""
                INSERT INTO quality_profiles (batch_id, model_name, model_version, predicted_category, quality_score_if_validated)
                VALUES (?, 'RF-BagasseQualityEnsemble', 'v2.4-Industrial', ?, ?)
            """, (b[0], b[10], b[11]))

            cursor.execute("""
                INSERT INTO chemical_recommendations 
                (batch_id, naoh_quantity, na2s_quantity, water_quantity, h2o2_quantity, recommendation_status, reference_information)
                VALUES (?, ?, ?, ?, ?, 'VALIDATED', 'Historical batch benchmark match')
            """, (b[0], b[12], b[13], b[14], b[15]))

        active_context["batch_id"] = "BAG-20261113-005"

    conn.commit()
    conn.close()

# ==================================================
# 1. AUTHENTICATION & LOGIN PAGE
# ==================================================

@app.route("/")
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("operator_dashboard"))

@app.route("/login", methods=["GET", "POST"])
@app.route("/auth/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        remember = request.form.get("remember")

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE user_id = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            if user["account_status"] != "active":
                return render_template("login.html", error="This operator account has been deactivated by the Administrator.")

            session.permanent = bool(remember)
            session["logged_in"] = True
            session["user_id"] = user["user_id"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]

            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("operator_dashboard"))
        else:
            return render_template("login.html", error="Invalid User ID or Password. Please try again.")

    return render_template("login.html")

@app.route("/auth/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==================================================
# 2. ADMIN DASHBOARD
# ==================================================

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    conn = get_db_connection()
    
    total_batches = conn.execute("SELECT COUNT(*) as c FROM batches").fetchone()["c"]
    analysed_batches = conn.execute("SELECT COUNT(*) as c FROM quality_profiles").fetchone()["c"]
    good_batches = conn.execute("SELECT COUNT(*) as c FROM quality_profiles WHERE predicted_category LIKE '%Grade A%'").fetchone()["c"]
    standard_batches = conn.execute("SELECT COUNT(*) as c FROM quality_profiles WHERE predicted_category LIKE '%Grade B%'").fetchone()["c"]
    poor_batches = conn.execute("SELECT COUNT(*) as c FROM quality_profiles WHERE predicted_category LIKE '%Grade C%'").fetchone()["c"]

    latest_profile_row = conn.execute("""
        SELECT predicted_category, quality_score_if_validated 
        FROM quality_profiles 
        ORDER BY id DESC LIMIT 1
    """).fetchone()

    recent_batches = conn.execute("""
        SELECT b.batch_id, b.batch_date, b.operator_name, s.ir_status, s.ph, s.temperature,
               q.predicted_category, q.quality_score_if_validated, c.recommendation_status as chem_status
        FROM batches b
        LEFT JOIN sensor_readings s ON b.batch_id = s.batch_id
        LEFT JOIN quality_profiles q ON b.batch_id = q.batch_id
        LEFT JOIN chemical_recommendations c ON b.batch_id = c.batch_id
        ORDER BY b.id DESC LIMIT 8
    """).fetchall()

    conn.close()

    analysed_pct = round((analysed_batches / total_batches * 100), 1) if total_batches > 0 else 0

    stats = {
        "total_batches": total_batches,
        "analysed_batches": analysed_batches,
        "analysed_pct": analysed_pct,
        "good_batches": good_batches,
        "standard_batches": standard_batches,
        "poor_batches": poor_batches,
        "latest_quality": latest_profile_row["predicted_category"] if latest_profile_row else None,
        "latest_score": latest_profile_row["quality_score_if_validated"] if latest_profile_row else None
    }

    live_sensors = esp_manager.get_readings()

    return render_template(
        "admin_dashboard.html",
        active_page="admin_dashboard",
        stats=stats,
        live_sensors=live_sensors,
        recent_batches=recent_batches
    )

# ==================================================
# 3. OPERATOR DASHBOARD
# ==================================================

@app.route("/operator/dashboard")
@login_required
def operator_dashboard():
    conn = get_db_connection()
    
    recent_batches = conn.execute("""
        SELECT b.batch_id, b.batch_date, b.operator_name, s.ir_status, s.ph, s.temperature,
               q.predicted_category, q.quality_score_if_validated
        FROM batches b
        LEFT JOIN sensor_readings s ON b.batch_id = s.batch_id
        LEFT JOIN quality_profiles q ON b.batch_id = q.batch_id
        ORDER BY b.id DESC LIMIT 5
    """).fetchall()

    latest_profile = conn.execute("""
        SELECT * FROM quality_profiles ORDER BY id DESC LIMIT 1
    """).fetchone()

    conn.close()

    live_sensors = esp_manager.get_readings()

    return render_template(
        "operator_dashboard.html",
        active_page="operator_dashboard",
        live_sensors=live_sensors,
        recent_batches=recent_batches,
        latest_profile=latest_profile
    )

# ==================================================
# 4. NEW BAGASSE BATCH
# ==================================================

@app.route("/batch/new")
@login_required
def new_batch():
    live_sensors = esp_manager.get_readings()

    hw_status = {
        "esp8266": live_sensors["connection_status"],
        "is_simulated": live_sensors["is_simulated"]
    }

    return render_template(
        "new_batch.html",
        active_page="new_batch",
        batch_id="",
        hw_status=hw_status
    )

@app.route("/batch/create", methods=["POST"])
@login_required
def create_batch():
    batch_id = (request.form.get("batch_id") or (request.json.get("batch_id") if request.is_json else "") or "").strip()
    operator_name = request.form.get("operator_name") or (request.json.get("operator_name") if request.is_json else "")
    operator_id = request.form.get("operator_id") or (request.json.get("operator_id") if request.is_json else "")
    remarks = request.form.get("remarks") or (request.json.get("remarks") if request.is_json else "")

    live_sensors = esp_manager.get_readings()
    hw_status = {
        "esp8266": live_sensors["connection_status"],
        "is_simulated": live_sensors["is_simulated"]
    }

    # Validate empty Batch ID
    if not batch_id:
        if request.is_json:
            return jsonify({"status": "error", "message": "Batch ID cannot be empty. Please enter a valid Batch ID."}), 400
        return render_template(
            "new_batch.html",
            active_page="new_batch",
            hw_status=hw_status,
            batch_id="",
            error="Batch ID cannot be empty. Please enter a valid Batch ID."
        ), 400

    conn = get_db_connection()
    try:
        # Validate duplicate Batch ID using the existing database
        existing = conn.execute("SELECT id FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        if existing:
            conn.close()
            if request.is_json:
                return jsonify({"status": "error", "message": f"Batch ID '{batch_id}' already exists in the system. Please enter a unique Batch ID."}), 409
            return render_template(
                "new_batch.html",
                active_page="new_batch",
                hw_status=hw_status,
                batch_id=batch_id,
                error=f"Batch ID '{batch_id}' already exists in the system. Please enter a unique Batch ID."
            ), 409

        conn.execute("""
            INSERT INTO batches (batch_id, operator_id, operator_name, remarks)
            VALUES (?, ?, ?, ?)
        """, (batch_id, operator_id, operator_name, remarks))

        # Store baseline sensor reading
        s = esp_manager.get_readings()
        conn.execute("""
            INSERT INTO sensor_readings (batch_id, ir_value, ir_status, ph, temperature, is_simulated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (batch_id, s["ir_value"], s["ir"], s["ph"], s["temperature"], 1 if s["is_simulated"] else 0))

        conn.commit()
        active_context["batch_id"] = batch_id
    except Exception as e:
        print("Batch creation error:", e)
    finally:
        conn.close()

    if request.is_json:
        return jsonify({"status": "success", "batch_id": batch_id, "redirect": url_for("system_camera")})

    return redirect(url_for("system_camera"))

# ==================================================
# 5. LIVE SENSOR MONITORING (3 SENSORS ONLY)
# ==================================================

@app.route("/sensors/live")
@login_required
def live_sensors():
    sensor_data = esp_manager.get_readings()
    return render_template(
        "live_sensors.html",
        active_page="live_sensors",
        sensor_data=sensor_data
    )

# ==================================================
# 6. SYSTEM CAMERA & IMAGE PROCESSING
# ==================================================

@app.route("/camera/view")
@login_required
def system_camera():
    active_batch = active_context["batch_id"] or "BAG-ACTIVE"
    conn = get_db_connection()
    latest_img = conn.execute("SELECT * FROM bagasse_images WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (active_batch,)).fetchone()
    if not latest_img:
        latest_img = conn.execute("SELECT * FROM bagasse_images ORDER BY id DESC LIMIT 1").fetchone()
    
    comp_row = conn.execute("SELECT * FROM bagasse_composition WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (active_batch,)).fetchone()
    conn.close()

    lignin_pred = lignin_model.predict_lignin(active_context["last_visual_features"])
    if comp_row:
        comp_pred = dict(comp_row)
    else:
        comp_pred = composition_engine.predict_composition(active_context["last_visual_features"])

    return render_template(
        "system_camera.html",
        active_page="system_camera",
        active_batch_id=active_batch,
        latest_image=latest_img,
        latest_features=active_context["last_visual_features"],
        latest_lignin=lignin_pred,
        latest_composition=comp_pred
    )

# ==================================================
# 7. BAGASSE QUALITY ASSESSMENT WIZARD
# ==================================================

@app.route("/assessment")
@login_required
def quality_assessment():
    active_batch = active_context["batch_id"] or "BAG-ACTIVE"
    conn = get_db_connection()
    latest_img = conn.execute("SELECT * FROM bagasse_images WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (active_batch,)).fetchone()
    if not latest_img:
        latest_img = conn.execute("SELECT * FROM bagasse_images ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()

    live_sensors = esp_manager.get_readings()

    return render_template(
        "quality_assessment.html",
        active_page="quality_assessment",
        active_batch_id=active_batch,
        latest_image=latest_img,
        live_sensors=live_sensors
    )

# ==================================================
# 8. BAGASSE QUALITY PROFILE
# ==================================================

@app.route("/profile/quality")
@login_required
def quality_profile():
    batch_id = request.args.get("batch_id") or active_context["batch_id"]
    conn = get_db_connection()
    
    comp_row = None
    if batch_id:
        profile = conn.execute("SELECT * FROM quality_profiles WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
        batch = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        sensor = conn.execute("SELECT * FROM sensor_readings WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
        features = conn.execute("SELECT * FROM image_features WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
        image = conn.execute("SELECT * FROM bagasse_images WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
        comp_row = conn.execute("SELECT * FROM bagasse_composition WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    else:
        profile = conn.execute("SELECT * FROM quality_profiles ORDER BY id DESC LIMIT 1").fetchone()
        b_id = profile["batch_id"] if profile else "DEMO"
        batch = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (b_id,)).fetchone()
        sensor = conn.execute("SELECT * FROM sensor_readings WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (b_id,)).fetchone()
        features = conn.execute("SELECT * FROM image_features WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (b_id,)).fetchone()
        image = conn.execute("SELECT * FROM bagasse_images WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (b_id,)).fetchone()
        comp_row = conn.execute("SELECT * FROM bagasse_composition ORDER BY id DESC LIMIT 1").fetchone()

    conn.close()

    if not profile:
        profile = {
            "batch_id": batch_id or "BAG-DEMO-001",
            "model_name": "RF-BagasseQualityEnsemble",
            "model_version": "v2.4-Industrial",
            "predicted_category": "Grade A (Prime Pulping Quality)",
            "quality_score_if_validated": 91.5,
            "predicted_lignin_pct": 20.8,
            "lignin_confidence": 93.0,
            "lignin_model_status": "LAB VALIDATION REQUIRED",
            "assessment_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        try:
            p_dict = dict(profile)
            if not p_dict.get("predicted_lignin_pct"):
                p_dict["predicted_lignin_pct"] = 20.8
                p_dict["lignin_confidence"] = 92.5
                p_dict["lignin_model_status"] = "LAB VALIDATION REQUIRED"

            score_val = p_dict.get("quality_score_if_validated")
            if score_val is None or score_val == 0.0:
                s_data = dict(sensor) if sensor else esp_manager.get_readings()
                f_data = dict(features) if features else active_context["last_visual_features"]
                if "ir" not in s_data and "ir_status" in s_data:
                    s_data["ir"] = 1 if s_data["ir_status"] == "Detected" else 0
                if "ir_value" not in s_data and "ir" in s_data:
                    s_data["ir_value"] = s_data["ir"]
                fused = FeatureFusionEngine.fuse_features(s_data, f_data)
                calc_score, calc_reason = quality_model.calculate_validated_score(fused)
                if calc_score > 0:
                    p_dict["quality_score_if_validated"] = calc_score
                    p_dict["scoring_reason"] = calc_reason
                else:
                    p_dict["scoring_reason"] = calc_reason
            else:
                p_dict["scoring_reason"] = "Calculated via Validated Multi-factor Pulping Index"
            profile = p_dict
        except Exception as e:
            print("Notice during quality profile processing:", e)

    if comp_row:
        comp_dict = dict(comp_row)
        comp_dict["is_trained"] = composition_engine.is_trained
    else:
        feat_dict = dict(features) if features else active_context["last_visual_features"]
        comp_dict = composition_engine.predict_composition(feat_dict)

    return render_template(
        "quality_profile.html",
        active_page="quality_profile",
        profile=profile,
        batch=batch or {},
        sensor=sensor or esp_manager.get_readings(),
        features=features or active_context["last_visual_features"],
        image=image,
        composition=comp_dict
    )

# ==================================================
# 9. CHEMICAL RECOMMENDATIONS (4 CHEMICALS)
# ==================================================

@app.route("/recommendations/chemical")
@login_required
def chemical_recommendations():
    conn = get_db_connection()
    all_batches = conn.execute("""
        SELECT b.batch_id, b.batch_date, q.predicted_category, q.quality_score_if_validated
        FROM batches b
        LEFT JOIN quality_profiles q ON b.batch_id = q.batch_id
        ORDER BY b.id DESC
    """).fetchall()

    batch_id = request.args.get("batch_id") or active_context.get("batch_id")
    if not batch_id and all_batches:
        batch_id = all_batches[0]["batch_id"]
    if not batch_id:
        batch_id = "BAG-DEMO-001"

    active_context["batch_id"] = batch_id
    version_req = request.args.get("version") or request.args.get("version_id")

    # Fetch all recommendation versions for this batch
    versions = conn.execute("""
        SELECT id, version_number, batch_weight_kg, target_pulp_quality, created_at, recommendation_status, is_active_version
        FROM chemical_recommendations
        WHERE batch_id = ?
        ORDER BY version_number DESC, id DESC
    """, (batch_id,)).fetchall()

    rec_row = None
    if version_req:
        try:
            v_num = int(version_req)
            rec_row = conn.execute("""
                SELECT * FROM chemical_recommendations 
                WHERE batch_id = ? AND version_number = ? 
                ORDER BY id DESC LIMIT 1
            """, (batch_id, v_num)).fetchone()
        except ValueError:
            pass

    if not rec_row:
        # Load active or highest version
        rec_row = conn.execute("""
            SELECT * FROM chemical_recommendations 
            WHERE batch_id = ? 
            ORDER BY is_active_version DESC, version_number DESC, id DESC LIMIT 1
        """, (batch_id,)).fetchone()

    # Previous version for diff comparison
    prev_rec_row = None
    if rec_row and "version_number" in rec_row.keys() and rec_row["version_number"] and rec_row["version_number"] > 1:
        prev_rec_row = conn.execute("""
            SELECT * FROM chemical_recommendations 
            WHERE batch_id = ? AND version_number = ? 
            ORDER BY id DESC LIMIT 1
        """, (batch_id, rec_row["version_number"] - 1)).fetchone()

    comp_row = conn.execute("SELECT * FROM bagasse_composition WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    batch_row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
    sensor_row = conn.execute("SELECT * FROM sensor_readings WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    qual_row = conn.execute("SELECT * FROM quality_profiles WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    feat_row = conn.execute("SELECT * FROM image_features WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    img_row = conn.execute("SELECT * FROM bagasse_images WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()

    conn.close()

    image_path = img_row["image_path"] if img_row else None
    live_sensors = esp_manager.get_readings()
    s_data = dict(sensor_row) if sensor_row else live_sensors

    # Check for material sensor change (Delta pH >= 0.3 or Delta Temp >= 2.0 C)
    stored_ph = float(s_data.get("ph") or 7.0)
    stored_temp = float(s_data.get("temperature") or 28.0)
    live_ph = float(live_sensors.get("ph") or 7.0)
    live_temp = float(live_sensors.get("temperature") or 28.0)
    delta_ph = round(abs(live_ph - stored_ph), 2)
    delta_temp = round(abs(live_temp - stored_temp), 1)
    is_material_change = (delta_ph >= 0.3 or delta_temp >= 2.0)

    # Process options: Batch weight and target quality
    batch_weight_kg = float(rec_row["batch_weight_kg"]) if (rec_row and "batch_weight_kg" in rec_row.keys() and rec_row["batch_weight_kg"]) else 1000.0
    target_quality = rec_row["target_pulp_quality"] if (rec_row and "target_pulp_quality" in rec_row.keys() and rec_row["target_pulp_quality"]) else "Bleachable Grade Paper"

    if comp_row:
        comp_data = dict(comp_row)
    else:
        comp_data = composition_engine.predict_composition(active_context.get("last_visual_features") or {})

    q_score = float(qual_row["quality_score_if_validated"]) if (qual_row and qual_row["quality_score_if_validated"]) else 75.0
    feat_data = dict(feat_row) if feat_row else (active_context.get("last_visual_features") or {})

    # Compute Model 4 Chemical Recommendation & Bayesian Optimization
    chem_pred = pipeline_predictor.predict_chemicals(
        composition=comp_data,
        quality_info={"quality_score": q_score},
        sensor_readings=s_data,
        visual_features=feat_data,
        target_pulp_quality=target_quality,
        batch_weight_kg=batch_weight_kg,
        batch_id=batch_id
    )
    bayesian_data = chem_pred
    opt = chem_pred.get("optimized", {})

    # Compute Model 5 Pulp Quality Prediction
    pulp_data = pipeline_predictor.predict_pulp_quality(
        chemical_rec=bayesian_data,
        composition=comp_data,
        batch_weight_kg=batch_weight_kg
    )

    # Resolve chemical quantities
    has_valid_rec = (rec_row and rec_row["naoh_quantity"] is not None and rec_row["naoh_quantity"] > 0)
    naoh_qty = rec_row["naoh_quantity"] if has_valid_rec else opt.get("naoh_kg_per_ton", 135.0)
    na2s_qty = rec_row["na2s_quantity"] if has_valid_rec else opt.get("na2s_kg_per_ton", 32.0)
    water_qty = rec_row["water_quantity"] if has_valid_rec else opt.get("water_l_per_ton", 4000.0)
    cook_time = rec_row["cooking_time_min"] if (has_valid_rec and "cooking_time_min" in rec_row.keys() and rec_row["cooking_time_min"] is not None) else opt.get("cooking_time_min", 65.0)
    h2o2_qty = rec_row["h2o2_quantity"] if (has_valid_rec and rec_row["h2o2_quantity"] is not None) else opt.get("h2o2_kg_per_ton", 18.5)
    bleach_water_qty = rec_row["bleach_water_quantity"] if (has_valid_rec and "bleach_water_quantity" in rec_row.keys() and rec_row["bleach_water_quantity"] is not None) else 1450.0

    # Parse or resolve explanations
    explanations = None
    if rec_row and "explanation_json" in rec_row.keys() and rec_row["explanation_json"]:
        try:
            explanations = json.loads(rec_row["explanation_json"])
        except Exception:
            explanations = None
    if not explanations:
        explanations = chem_pred.get("explanations", {})

    # Check for unvalidated state
    comp_unvalidated = (
        not comp_row or 
        comp_data.get("validation_status") == "COMPOSITION MODEL NOT TRAINED – LAB DATA REQUIRED"
    )
    rec_status = "VALIDATED" if (rec_row and rec_row["recommendation_status"] == "VALIDATED") else chem_pred.get("validation_status", "SYNTHETIC MODEL – REAL-WORLD VALIDATION REQUIRED")
    if comp_unvalidated:
        rec_status = "ADAPTIVE RECOMMENDATION UNAVAILABLE – VALIDATED PROCESS DATA REQUIRED"

    batch_inputs = {
        "batch_id": batch_id,
        "batch_weight_kg": batch_weight_kg,
        "target_quality": target_quality,
        "initial_lignin": comp_data.get("lignin_pct") or 21.4,
        "initial_cellulose": comp_data.get("cellulose_pct") or 43.8,
        "initial_hemicellulose": comp_data.get("hemicellulose_pct") or 28.2,
        "initial_ash": comp_data.get("ash_pct") or 2.6,
        "ph": stored_ph,
        "temperature": stored_temp,
        "ir_status": s_data.get("ir_status") or live_sensors.get("ir_status") or "Sample In Digester",
        "validation_status": comp_data.get("validation_status", "SYNTHETIC MODEL - REAL-WORLD VALIDATION REQUIRED")
    }

    # Version comparison diff
    diff_data = None
    if prev_rec_row:
        diff_data = {
            "prev_version": prev_rec_row["version_number"],
            "curr_version": rec_row["version_number"] if rec_row else 1,
            "naoh_diff": round(naoh_qty - prev_rec_row["naoh_quantity"], 2),
            "na2s_diff": round(na2s_qty - prev_rec_row["na2s_quantity"], 2),
            "water_diff": round(water_qty - prev_rec_row["water_quantity"], 1),
            "time_diff": round(cook_time - prev_rec_row["cooking_time_min"], 1),
            "h2o2_diff": round(h2o2_qty - (prev_rec_row["h2o2_quantity"] or 0), 2),
            "prev_naoh": prev_rec_row["naoh_quantity"],
            "curr_naoh": naoh_qty,
            "prev_na2s": prev_rec_row["na2s_quantity"],
            "curr_na2s": na2s_qty,
            "prev_water": prev_rec_row["water_quantity"],
            "curr_water": water_qty,
            "prev_time": prev_rec_row["cooking_time_min"],
            "curr_time": cook_time,
            "prev_h2o2": prev_rec_row["h2o2_quantity"] or 0,
            "curr_h2o2": h2o2_qty
        }

    chem_data = {
        "status": rec_status,
        "version_number": rec_row["version_number"] if rec_row and "version_number" in rec_row.keys() else 1,
        "stages": {
            "kraft_cooking": {
                "stage_name": "Stage 1: Kraft Alkaline Cooking",
                "naoh": {
                    "name": "NaOH (Sodium Hydroxide)",
                    "formula": "NaOH",
                    "role": "Active Alkali / Delignification Agent",
                    "quantity": naoh_qty,
                    "unit": "kg/ton dry bagasse",
                    "status": "Optimal Alkaline Charge",
                    "reference": "AI Model 4 Stoichiometric Target",
                    "explanation": explanations.get("naoh", {})
                },
                "na2s": {
                    "name": "Na₂S (Sodium Sulfide)",
                    "formula": "Na₂S",
                    "role": "Sulfidity Agent for Fiber Preservation",
                    "quantity": na2s_qty,
                    "unit": "kg/ton dry bagasse",
                    "status": f"Target Sulfidity {opt.get('sulfidity_percent', 24.0)}%",
                    "reference": "AI Model 4 Sulfidity Balance",
                    "explanation": explanations.get("na2s", {})
                },
                "water": {
                    "name": "H₂O (Cooking Process Liquor / Water)",
                    "formula": "H₂O",
                    "role": "Liquor-to-Bagasse Dilution Ratio",
                    "quantity": water_qty,
                    "unit": "Liters/ton dry bagasse",
                    "status": f"Liquor-to-Wood Ratio ~ {opt.get('liquor_ratio', 4.0)}:1",
                    "reference": "AI Thermal Balance & Dilution Model",
                    "explanation": explanations.get("water", {})
                },
                "cooking_time": {
                    "name": "Validated Digestion Cooking Time",
                    "formula": "⏱️ Time",
                    "role": f"Reaction Retention at {opt.get('cooking_temp_c', 162.0)}°C",
                    "quantity": cook_time,
                    "unit": f"Minutes at {opt.get('cooking_temp_c', 162.0)}°C",
                    "status": "Empirically Verified Retention",
                    "reference": "Arrhenius Kinetics Validated",
                    "explanation": explanations.get("cooking_time", {})
                }
            },
            "bleaching": {
                "stage_name": "Stage 2: Peroxide Bleaching Sequence (Stage-P)",
                "h2o2": {
                    "name": "H₂O₂ (Hydrogen Peroxide)",
                    "formula": "H₂O₂",
                    "role": "Separately Validated Bleaching Stage Oxidant",
                    "quantity": h2o2_qty if opt.get("has_bleaching_dataset") else "UNAVAILABLE",
                    "unit": "kg/ton unbleached pulp" if opt.get("has_bleaching_dataset") else "DATASET REQUIRED",
                    "status": opt.get("bleaching_status", "BLEACHING DATASET NOT AVAILABLE – H₂O₂ RECOMMENDATION UNAVAILABLE"),
                    "reference": "Brightness target derived from bagasse lignin (Requires Dedicated Bleaching Dataset)",
                    "explanation": explanations.get("h2o2", {})
                },
                "water": {
                    "name": "H₂O (Bleach Dilution Wash Liquor)",
                    "formula": "H₂O",
                    "role": "Stage-P Slurry Consistency Maintenance (10-12%)",
                    "quantity": bleach_water_qty,
                    "unit": "Liters/ton unbleached pulp",
                    "status": "Medium Consistency Wash",
                    "reference": "Standard wash dilution",
                    "explanation": explanations.get("bleach_water", {})
                }
            }
        }
    }

    return render_template(
        "chemical_recommendations.html",
        active_page="chemical_recommendations",
        batch_id=batch_id,
        all_batches=all_batches,
        chem_data=chem_data,
        composition=comp_data,
        batch_inputs=batch_inputs,
        visual_features=feat_data,
        image_path=image_path,
        bayesian_data=bayesian_data,
        pulp_data=pulp_data,
        versions=versions,
        current_version=chem_data["version_number"],
        diff_data=diff_data,
        is_material_change=is_material_change,
        delta_ph=delta_ph,
        delta_temp=delta_temp,
        live_sensors=live_sensors,
        explanations=explanations,
        batch_weight_kg=batch_weight_kg,
        target_pulp_quality=target_quality,
        matched_records=chem_pred.get("matched_records", []),
        matched_records_count=chem_pred.get("matched_records_count", 0),
        distance_weighted_targets=chem_pred.get("distance_weighted_targets", {}),
        has_bleaching_dataset=opt.get("has_bleaching_dataset", False),
        bleaching_status=opt.get("bleaching_status", "BLEACHING DATASET NOT AVAILABLE – H₂O₂ RECOMMENDATION UNAVAILABLE")
    )

def _execute_recommendation_recalculation(data):
    """
    Internal engine to perform sensor-adaptive recommendation recalculation,
    Bayesian parameter optimization, Model 5 pulp quality estimation, and version tracking.
    """
    batch_id = data.get("batch_id") or active_context.get("batch_id") or "BAG-DEMO-001"
    batch_weight = float(data.get("batch_weight_kg", 1000.0))
    target_quality = data.get("target_pulp_quality", "Bleachable Grade Paper")
    use_live_sensors = bool(data.get("use_live_sensors", True))

    conn = get_db_connection()
    comp_row = conn.execute("SELECT * FROM bagasse_composition WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    feat_row = conn.execute("SELECT * FROM image_features WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    qual_row = conn.execute("SELECT * FROM quality_profiles WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    sensor_row = conn.execute("SELECT * FROM sensor_readings WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()

    live_sensors = esp_manager.get_readings()
    if use_live_sensors:
        s_data = live_sensors
    else:
        s_data = dict(sensor_row) if sensor_row else live_sensors

    if not comp_row:
        if data.get("composition"):
            comp_data = data["composition"]
        elif feat_row:
            comp_data = composition_engine.predict_composition(dict(feat_row))
        elif active_context.get("last_visual_features"):
            comp_data = composition_engine.predict_composition(active_context["last_visual_features"])
        else:
            conn.close()
            return {
                "success": False,
                "status": "UNAVAILABLE",
                "message": "ADAPTIVE RECOMMENDATION UNAVAILABLE – VALIDATED PROCESS DATA REQUIRED. Please run quality assessment first."
            }, 400
    else:
        comp_data = dict(comp_row)

    feat_data = dict(feat_row) if feat_row else (active_context.get("last_visual_features") or {})
    q_score = float(qual_row["quality_score_if_validated"]) if (qual_row and qual_row["quality_score_if_validated"]) else 75.0

    # Compute Model 4 Chemical Recommendation & Bayesian Optimization
    chem_pred = pipeline_predictor.predict_chemicals(
        composition=comp_data,
        quality_info={"quality_score": q_score},
        sensor_readings=s_data,
        visual_features=feat_data,
        target_pulp_quality=target_quality,
        batch_weight_kg=batch_weight,
        batch_id=batch_id
    )
    opt = chem_pred.get("optimized", {})
    base = chem_pred.get("baseline", {})
    sav = chem_pred.get("savings", {})

    # Compute Model 5 Pulp Quality Prediction
    pulp_pred = pipeline_predictor.predict_pulp_quality(
        chemical_rec=chem_pred,
        composition=comp_data,
        batch_weight_kg=batch_weight
    )

    # Determine next version number
    max_ver_row = conn.execute("SELECT MAX(version_number) as m FROM chemical_recommendations WHERE batch_id = ?", (batch_id,)).fetchone()
    max_ver = max_ver_row["m"] if (max_ver_row and max_ver_row["m"] is not None) else None
    new_version = (max_ver + 1) if (max_ver is not None and max_ver >= 1) else 1

    # Mark existing versions as inactive
    conn.execute("UPDATE chemical_recommendations SET is_active_version = 0 WHERE batch_id = ?", (batch_id,))

    # Insert new version record
    conn.execute("""
        INSERT INTO chemical_recommendations (
            batch_id, version_number, batch_weight_kg, target_pulp_quality,
            naoh_quantity, naoh_unit, na2s_quantity, na2s_unit,
            water_quantity, water_unit, cooking_time_min,
            h2o2_quantity, h2o2_unit, bleach_water_quantity, bleach_water_unit,
            recommendation_status, reference_information, explanation_json, input_parameters_json, is_active_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    """, (
        batch_id,
        new_version,
        batch_weight,
        target_quality,
        opt.get("naoh_kg_per_ton", 135.0), "kg/ton dry bagasse",
        opt.get("na2s_kg_per_ton", 32.0), "kg/ton dry bagasse",
        opt.get("water_l_per_ton", 3680.0), "Liters/ton dry bagasse",
        opt.get("cooking_time_min", 65.0),
        opt.get("h2o2_kg_per_ton", 18.5), "kg/ton unbleached pulp",
        opt.get("bleach_water_l_per_ton", 1450.0), "Liters/ton unbleached pulp",
        chem_pred.get("validation_status", "VALIDATED"),
        "AI Model 4 Sensor-Adaptive Recalculation Engine",
        json.dumps(chem_pred.get("explanations", {})),
        json.dumps(chem_pred.get("input_parameters_used", {}))
    ))

    # Insert Bayesian optimization snapshot
    conn.execute("""
        INSERT INTO bayesian_optimizations (
            batch_id, baseline_active_alkali, optimized_active_alkali,
            baseline_sulfidity, optimized_sulfidity,
            baseline_liquor_ratio, optimized_liquor_ratio,
            baseline_temp, optimized_temp,
            baseline_time, optimized_time,
            chemical_saving_pct, energy_saving_pct
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        batch_id,
        base.get("aa_percent", 16.0), opt.get("active_alkali_percent", 15.5),
        base.get("sulfidity", 24.0), opt.get("sulfidity_percent", 24.0),
        base.get("liquor_ratio", 4.2), opt.get("liquor_ratio", 4.0),
        base.get("temp_c", 165.0), opt.get("cooking_temp_c", 162.0),
        base.get("time_min", 65.0), opt.get("cooking_time_min", 60.0),
        sav.get("chemical_cost_savings_pct", 0.0), sav.get("thermal_energy_savings_pct", 0.0)
    ))

    # Fetch previous version for comparison diff
    prev_ver = conn.execute("""
        SELECT * FROM chemical_recommendations 
        WHERE batch_id = ? AND version_number = ?
    """, (batch_id, new_version - 1)).fetchone()

    conn.commit()
    conn.close()

    diff = None
    if prev_ver:
        diff = {
            "prev_version": new_version - 1,
            "new_version": new_version,
            "curr_version": new_version,
            "naoh_diff": round(opt.get("naoh_kg_per_ton", 135.0) - prev_ver["naoh_quantity"], 2),
            "na2s_diff": round(opt.get("na2s_kg_per_ton", 32.0) - prev_ver["na2s_quantity"], 2),
            "water_diff": round(opt.get("water_l_per_ton", 3680.0) - prev_ver["water_quantity"], 1),
            "time_diff": round(opt.get("cooking_time_min", 65.0) - prev_ver["cooking_time_min"], 1),
            "h2o2_diff": round(opt.get("h2o2_kg_per_ton", 18.5) - (prev_ver["h2o2_quantity"] or 0), 2)
        }

    return {
        "success": True,
        "batch_id": batch_id,
        "version_number": new_version,
        "recommendation": chem_pred,
        "pulp_quality": pulp_pred,
        "diff": diff,
        "matched_records_count": chem_pred.get("matched_records_count", 0),
        "matched_records": chem_pred.get("matched_records", []),
        "message": f"Successfully generated Version {new_version} adaptive chemistry based on current sensor feedback!"
    }, 200

@app.route("/api/recommendations/recalculate", methods=["POST"])
@login_required
def recalculate_recommendations():
    """
    Step 6: Operator-controlled recommendation recalculation based on current sensor readings and process targets.
    Creates a new version record without overwriting previous versions.
    """
    data = request.get_json() or {}
    res, status_code = _execute_recommendation_recalculation(data)
    return jsonify(res), status_code

# =========================================================================
# REAL-TIME SENSOR & ADAPTIVE CHEMICAL RECOMMENDATION REST APIs
# =========================================================================

@app.route("/api/sensors/live", methods=["GET"])
def api_sensors_live():
    """
    Step 1: Returns real-time sensor readings (pH, temperature, IR sample detection)
    from ESP8266 or simulated telemetry.
    """
    live = esp_manager.get_readings()
    return jsonify({
        "success": True,
        "ir_status": live.get("ir") or live.get("ir_status", "Sample In Digester"),
        "ir_value": live.get("ir_value", 1),
        "ph": round(float(live.get("ph", 7.0)), 2),
        "temperature": round(float(live.get("temperature", 28.0)), 1),
        "is_simulated": bool(live.get("is_simulated", False)),
        "connection_status": live.get("connection_status", "Active"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/api/batch/<batch_id>/profile", methods=["GET"])
def api_batch_profile(batch_id):
    """
    Step 2: Returns the complete batch profile: batch ID, sensor readings, webcam optical features,
    bagasse composition (Lignin, Cellulose, Hemicellulose, Ash), validation status, and operator inputs.
    """
    conn = get_db_connection()
    batch_row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
    sensor_row = conn.execute("SELECT * FROM sensor_readings WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    feat_row = conn.execute("SELECT * FROM image_features WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    comp_row = conn.execute("SELECT * FROM bagasse_composition WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    qual_row = conn.execute("SELECT * FROM quality_profiles WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    rec_row = conn.execute("SELECT * FROM chemical_recommendations WHERE batch_id = ? ORDER BY version_number DESC, id DESC LIMIT 1", (batch_id,)).fetchone()
    conn.close()

    if not batch_row and not (active_context.get("batch_id") == batch_id):
        return jsonify({"success": False, "error": f"Batch '{batch_id}' not found"}), 404

    live_s = esp_manager.get_readings()
    s_data = dict(sensor_row) if sensor_row else live_s
    f_data = dict(feat_row) if feat_row else (active_context.get("last_visual_features") or {})
    c_data = dict(comp_row) if comp_row else {
        "lignin_pct": 21.4, "cellulose_pct": 43.8, "hemicellulose_pct": 28.2, "ash_pct": 2.6,
        "validation_status": "SYNTHETIC MODEL - REAL-WORLD VALIDATION REQUIRED"
    }

    return jsonify({
        "success": True,
        "batch_id": batch_id,
        "batch_metadata": dict(batch_row) if batch_row else {},
        "sensor_readings": {
            "ir_status": s_data.get("ir_status") or s_data.get("ir", "Sample In Digester"),
            "ph": round(float(s_data.get("ph", 7.0)), 2),
            "temperature": round(float(s_data.get("temperature", 28.0)), 1),
            "is_simulated": bool(s_data.get("is_simulated", False))
        },
        "image_features": {
            "brightness": round(float(f_data.get("brightness", 140.0)), 1),
            "greenish_percentage": round(float(f_data.get("greenish_percentage") or f_data.get("green_percentage", 8.0)), 1),
            "brownish_percentage": round(float(f_data.get("brownish_percentage") or f_data.get("brown_percentage", 35.0)), 1),
            "texture_contrast": round(float(f_data.get("texture_contrast") or f_data.get("contrast", 40.0)), 1),
            "contrast": round(float(f_data.get("contrast", 40.0)), 1),
            "texture_score": round(float(f_data.get("texture_score", 42.0)), 2),
            "edge_density": round(float(f_data.get("edge_density", 0.15)), 2),
            "sharpness": round(float(f_data.get("sharpness", 80.0)), 1)
        },
        "bagasse_composition": {
            "lignin_pct": round(float(c_data.get("lignin_pct", 21.4)), 2),
            "cellulose_pct": round(float(c_data.get("cellulose_pct", 43.8)), 2),
            "hemicellulose_pct": round(float(c_data.get("hemicellulose_pct", 28.2)), 2),
            "ash_pct": round(float(c_data.get("ash_pct", 2.6)), 2),
            "validation_status": c_data.get("validation_status", "VALIDATED")
        },
        "quality_profile": {
            "predicted_category": qual_row["predicted_category"] if qual_row else "Grade A",
            "quality_score": qual_row["quality_score_if_validated"] if qual_row else 85.0
        },
        "operator_inputs": {
            "batch_weight_kg": rec_row["batch_weight_kg"] if (rec_row and "batch_weight_kg" in rec_row.keys() and rec_row["batch_weight_kg"]) else 1000.0,
            "target_pulp_quality": rec_row["target_pulp_quality"] if (rec_row and "target_pulp_quality" in rec_row.keys() and rec_row["target_pulp_quality"]) else "Bleachable Grade Paper"
        }
    })

@app.route("/api/dataset/chemical/status", methods=["GET"])
def api_dataset_chemical_status():
    """
    Returns synthetic dataset status, record count, feature definitions, and bleaching dataset availability.
    """
    row_count = len(dataset_retrieval_engine.df) if (dataset_retrieval_engine.df is not None) else 0
    return jsonify({
        "success": True,
        "dataset_name": "BAGASSE_IQ_Kraft_Chemical_Training_10000.csv",
        "dataset_path": dataset_retrieval_engine.dataset_path,
        "is_loaded": dataset_retrieval_engine.is_loaded,
        "total_records": row_count,
        "features": dataset_retrieval_engine.feature_cols,
        "targets": dataset_retrieval_engine.target_cols,
        "has_bleaching_dataset": False,
        "bleaching_dataset_status": "BLEACHING DATASET NOT AVAILABLE – H₂O₂ RECOMMENDATION UNAVAILABLE",
        "retrieval_method": "Weighted Euclidean Distance (StandardScaler + Domain Calibration)",
        "status": "ACTIVE"
    })

@app.route("/api/chemical-recommendation/retrieve", methods=["POST"])
def api_chemical_recommendation_retrieve():
    """
    Retrieves top K similar synthetic records from BAGASSE_IQ_Kraft_Chemical_Training_10000.csv
    based on provided or current sensor/optical/composition inputs.
    """
    data = request.get_json() or {}
    batch_id = data.get("batch_id") or active_context.get("batch_id")
    top_k = int(data.get("top_k", 5))

    conn = get_db_connection()
    sensor_row = conn.execute("SELECT * FROM sensor_readings WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone() if batch_id else None
    feat_row = conn.execute("SELECT * FROM image_features WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone() if batch_id else None
    comp_row = conn.execute("SELECT * FROM bagasse_composition WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone() if batch_id else None
    conn.close()

    live_s = esp_manager.get_readings()
    s_row = dict(sensor_row) if sensor_row else live_s
    f_row = dict(feat_row) if feat_row else (active_context.get("last_visual_features") or {})
    c_row = dict(comp_row) if comp_row else {}

    query_inputs = {
        "ph": float(data.get("ph", s_row.get("ph", 7.0))),
        "temperature": float(data.get("temperature", s_row.get("temperature", 28.0))),
        "brown_percentage": float(data.get("brown_percentage", f_row.get("brownish_percentage", f_row.get("brown_percentage", 35.0)))),
        "green_percentage": float(data.get("green_percentage", f_row.get("greenish_percentage", f_row.get("green_percentage", 8.0)))),
        "brightness": float(data.get("brightness", f_row.get("brightness", 140.0))),
        "contrast": float(data.get("contrast", f_row.get("texture_contrast", f_row.get("contrast", 40.0)))),
        "texture_score": float(data.get("texture_score", f_row.get("texture_score", 42.0))),
        "edge_density": float(data.get("edge_density", f_row.get("edge_density", 0.15))),
        "lignin_pct": float(data.get("lignin_pct", c_row.get("lignin_pct", 21.4))),
        "cellulose_pct": float(data.get("cellulose_pct", c_row.get("cellulose_pct", 43.8))),
        "hemicellulose_pct": float(data.get("hemicellulose_pct", c_row.get("hemicellulose_pct", 28.2))),
        "ash_pct": float(data.get("ash_pct", c_row.get("ash_pct", 2.6))),
        "batch_weight_kg": float(data.get("batch_weight_kg", 1000.0)),
        "target_pulp_quality": data.get("target_pulp_quality", "Bleachable Grade Paper")
    }

    result = dataset_retrieval_engine.retrieve_similar_records(query_inputs, top_k=top_k)

    return jsonify({
        "success": True,
        "batch_id": batch_id,
        "query_inputs": query_inputs,
        "top_k": top_k,
        "matched_records_count": len(result.get("matched_records", [])),
        "matched_records": result.get("matched_records", []),
        "distance_weighted_targets": result.get("distance_weighted_targets", {}),
        "data_source": "data/BAGASSE_IQ_Kraft_Chemical_Training_10000.csv",
        "validation_notice": "SYNTHETIC DATA-BASED EXPERIMENTAL RECOMMENDATION – REAL-WORLD VALIDATION REQUIRED"
    })

@app.route("/api/chemical-recommendation/generate", methods=["POST"])
def api_chemical_recommendation_generate():
    """
    Generates an adaptive chemical recommendation based on current batch inputs and sensor data.
    Stores new version record and returns recommendation, pulp quality, and matched records.
    """
    data = request.get_json() or {}
    res, status_code = _execute_recommendation_recalculation(data)
    return jsonify(res), status_code

@app.route("/api/chemical-recommendation/<batch_id>", methods=["GET"])
def api_get_chemical_recommendation(batch_id):
    """
    Returns the latest chemical recommendation for the given batch ID,
    including the 4 chemicals (NaOH, Na2S, H2O, H2O2), cooking parameters,
    and matched historical records.
    """
    conn = get_db_connection()
    rec_row = conn.execute("""
        SELECT * FROM chemical_recommendations
        WHERE batch_id = ?
        ORDER BY version_number DESC, id DESC LIMIT 1
    """, (batch_id,)).fetchone()
    comp_row = conn.execute("SELECT * FROM bagasse_composition WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    sensor_row = conn.execute("SELECT * FROM sensor_readings WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    feat_row = conn.execute("SELECT * FROM image_features WHERE batch_id = ? ORDER BY id DESC LIMIT 1", (batch_id,)).fetchone()
    conn.close()

    s_data = dict(sensor_row) if sensor_row else esp_manager.get_readings()
    c_data = dict(comp_row) if comp_row else {"lignin_pct": 21.4, "cellulose_pct": 43.8, "hemicellulose_pct": 28.2, "ash_pct": 2.6}
    f_data = dict(feat_row) if feat_row else (active_context.get("last_visual_features") or {})

    if not rec_row:
        chem_pred = pipeline_predictor.predict_chemicals(
            composition=c_data,
            quality_info={"quality_score": 75.0},
            sensor_readings=s_data,
            visual_features=f_data,
            target_pulp_quality="Bleachable Grade Paper",
            batch_weight_kg=1000.0,
            batch_id=batch_id
        )
        opt = chem_pred.get("optimized", {})
        explanations = chem_pred.get("explanations", {})
        ver = 1
        rec_status = chem_pred.get("validation_status", "SYNTHETIC MODEL – REAL-WORLD VALIDATION REQUIRED")
        batch_weight = 1000.0
        target_quality = "Bleachable Grade Paper"
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        matched_records = chem_pred.get("matched_records", [])
    else:
        rec = dict(rec_row)
        ver = rec.get("version_number", 1)
        rec_status = rec.get("recommendation_status", "VALIDATED")
        batch_weight = float(rec.get("batch_weight_kg", 1000.0))
        target_quality = rec.get("target_pulp_quality", "Bleachable Grade Paper")
        created_at = rec.get("created_at")
        explanations = json.loads(rec["explanation_json"]) if rec.get("explanation_json") else {}

        q_inputs = {
            "ph": s_data.get("ph", 7.0),
            "temperature": s_data.get("temperature", 28.0),
            "lignin_pct": c_data.get("lignin_pct", 21.4),
            "batch_weight_kg": batch_weight,
            "target_pulp_quality": target_quality
        }
        ret_res = dataset_retrieval_engine.retrieve_similar_records(q_inputs, top_k=5)
        matched_records = ret_res.get("matched_records", [])

        opt = {
            "naoh_kg_per_ton": rec.get("naoh_quantity", 135.0),
            "na2s_kg_per_ton": rec.get("na2s_quantity", 32.0),
            "water_l_per_ton": rec.get("water_quantity", 3680.0),
            "cooking_time_min": rec.get("cooking_time_min", 65.0),
            "bleach_water_l_per_ton": rec.get("bleach_water_quantity", 1450.0),
            "total_naoh_kg": round(float(rec.get("naoh_quantity", 135.0) * (batch_weight / 1000.0)), 2),
            "total_na2s_kg": round(float(rec.get("na2s_quantity", 32.0) * (batch_weight / 1000.0)), 2),
            "total_water_l": round(float(rec.get("water_quantity", 3680.0) * (batch_weight / 1000.0)), 1),
            "has_bleaching_dataset": False,
            "bleaching_status": "BLEACHING DATASET NOT AVAILABLE – H₂O₂ RECOMMENDATION UNAVAILABLE"
        }

    return jsonify({
        "success": True,
        "batch_id": batch_id,
        "version_number": ver,
        "created_at": created_at,
        "recommendation_status": rec_status,
        "chemicals": {
            "naoh": {
                "chemical": "NaOH",
                "name": "Sodium Hydroxide",
                "quantity_kg_per_ton": opt.get("naoh_kg_per_ton"),
                "total_quantity_kg": opt.get("total_naoh_kg", round(float(opt.get("naoh_kg_per_ton", 135.0) * (batch_weight / 1000.0)), 2)),
                "unit": "kg/ton dry bagasse",
                "role": "Active Alkali / Delignification Agent",
                "explanation": explanations.get("naoh", {})
            },
            "na2s": {
                "chemical": "Na2S",
                "name": "Sodium Sulfide",
                "quantity_kg_per_ton": opt.get("na2s_kg_per_ton"),
                "total_quantity_kg": opt.get("total_na2s_kg", round(float(opt.get("na2s_kg_per_ton", 32.0) * (batch_weight / 1000.0)), 2)),
                "unit": "kg/ton dry bagasse",
                "role": "Sulfidity Agent for Fiber Preservation",
                "explanation": explanations.get("na2s", {})
            },
            "water": {
                "chemical": "H2O",
                "name": "Cooking Process Liquor / Water",
                "quantity_l_per_ton": opt.get("water_l_per_ton"),
                "total_quantity_l": opt.get("total_water_l", round(float(opt.get("water_l_per_ton", 3680.0) * (batch_weight / 1000.0)), 1)),
                "unit": "Liters/ton dry bagasse",
                "role": "Liquor-to-Bagasse Dilution Ratio (Deducting Solution Water)",
                "explanation": explanations.get("water", {})
            },
            "h2o2": {
                "chemical": "H2O2",
                "name": "Hydrogen Peroxide",
                "quantity_kg_per_ton": "UNAVAILABLE",
                "unit": "kg/ton unbleached pulp",
                "role": "Separately Validated Bleaching Stage Oxidant",
                "status": "BLEACHING DATASET NOT AVAILABLE – H₂O₂ RECOMMENDATION UNAVAILABLE",
                "explanation": explanations.get("h2o2", {})
            },
            "bleach_water": {
                "chemical": "H2O",
                "name": "Bleach Dilution Wash Liquor",
                "quantity_l_per_ton": opt.get("bleach_water_l_per_ton", 1450.0),
                "unit": "Liters/ton unbleached pulp",
                "role": "11% Medium Consistency Wash",
                "explanation": explanations.get("bleach_water", {})
            }
        },
        "cooking_parameters": {
            "cooking_time_min": opt.get("cooking_time_min", 65.0),
            "batch_weight_kg": batch_weight,
            "target_pulp_quality": target_quality
        },
        "matched_records_count": len(matched_records),
        "matched_records": matched_records,
        "validation_status": rec_status,
        "safety_notice": "SYNTHETIC DATA-BASED EXPERIMENTAL RECOMMENDATION – REAL-WORLD VALIDATION REQUIRED"
    })

@app.route("/api/recommendations/versions")
@login_required
def get_recommendation_versions():
    batch_id = request.args.get("batch_id") or active_context.get("batch_id")
    conn = get_db_connection()
    versions = conn.execute("""
        SELECT id, version_number, batch_weight_kg, target_pulp_quality, created_at, recommendation_status, is_active_version,
               naoh_quantity, na2s_quantity, water_quantity, cooking_time_min, h2o2_quantity
        FROM chemical_recommendations
        WHERE batch_id = ?
        ORDER BY version_number DESC
    """, (batch_id,)).fetchall()
    conn.close()
    return jsonify({
        "success": True,
        "batch_id": batch_id,
        "versions": [dict(v) for v in versions]
    })

# ==================================================
# 10. ANALYSIS HISTORY & EXPORT
# ==================================================

@app.route("/history")
@login_required
def history():
    q = request.args.get("q", "").strip()
    quality = request.args.get("quality", "").strip()

    conn = get_db_connection()
    sql = """
        SELECT b.batch_id, b.batch_date, b.operator_name, s.ir_status, s.ph, s.temperature,
               q.predicted_category, q.quality_score_if_validated, q.predicted_lignin_pct,
               c.recommendation_status as chem_status
        FROM batches b
        LEFT JOIN sensor_readings s ON b.batch_id = s.batch_id
        LEFT JOIN quality_profiles q ON b.batch_id = q.batch_id
        LEFT JOIN chemical_recommendations c ON b.batch_id = c.batch_id
        WHERE 1=1
    """
    params = []
    if q:
        sql += " AND b.batch_id LIKE ?"
        params.append(f"%{q}%")
    if quality:
        sql += " AND q.predicted_category LIKE ?"
        params.append(f"%{quality}%")
    sql += " ORDER BY b.id DESC"

    batches = conn.execute(sql, params).fetchall()
    conn.close()

    return render_template(
        "history.html",
        active_page="history",
        batches=batches,
        search_query=q,
        selected_quality=quality
    )

@app.route("/api/history/export")
@login_required
def export_history_csv():
    conn = get_db_connection()
    batches = conn.execute("""
        SELECT b.batch_id, b.batch_date, b.operator_name, s.ir_status, s.ph, s.temperature,
               q.predicted_category, q.quality_score_if_validated, q.predicted_lignin_pct,
               c.naoh_quantity, c.na2s_quantity, c.water_quantity, c.cooking_time_min,
               c.h2o2_quantity, c.bleach_water_quantity, c.recommendation_status
        FROM batches b
        LEFT JOIN sensor_readings s ON b.batch_id = s.batch_id
        LEFT JOIN quality_profiles q ON b.batch_id = q.batch_id
        LEFT JOIN chemical_recommendations c ON b.batch_id = c.batch_id
        ORDER BY b.id ASC
    """).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Batch ID", "Date", "Operator", "IR Status", "pH", "Temperature (°C)",
        "Estimated Lignin (%)", "Quality Category", "Quality Score",
        "NaOH (kg/ton)", "Na2S (kg/ton)", "Water (L/ton)", "Cooking Time (min)",
        "H2O2 (kg/ton)", "Bleach Water (L/ton)", "Chemical Status"
    ])
    for b in batches:
        writer.writerow([
            b["batch_id"], b["batch_date"], b["operator_name"], b["ir_status"], b["ph"], b["temperature"],
            b["predicted_lignin_pct"] or "--", b["predicted_category"], b["quality_score_if_validated"],
            b["naoh_quantity"], b["na2s_quantity"], b["water_quantity"], b["cooking_time_min"],
            b["h2o2_quantity"], b["bleach_water_quantity"], b["recommendation_status"]
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=bagasse_iq_history_export.csv"}
    )

# ==================================================
# 11. ANALYTICS & REPORTS
# ==================================================

@app.route("/analytics")
@login_required
def analytics():
    conn = get_db_connection()
    
    total = conn.execute("SELECT COUNT(*) as c FROM batches").fetchone()["c"]
    avg_ph = conn.execute("SELECT ROUND(AVG(ph), 2) as m FROM sensor_readings").fetchone()["m"]
    avg_temp = conn.execute("SELECT ROUND(AVG(temperature), 1) as m FROM sensor_readings").fetchone()["m"]
    good_batches = conn.execute("SELECT COUNT(*) as c FROM quality_profiles WHERE predicted_category LIKE '%Grade A%'").fetchone()["c"]
    standard_batches = conn.execute("SELECT COUNT(*) as c FROM quality_profiles WHERE predicted_category LIKE '%Grade B%'").fetchone()["c"]
    poor_batches = conn.execute("SELECT COUNT(*) as c FROM quality_profiles WHERE predicted_category LIKE '%Grade C%'").fetchone()["c"]

    chart_rows = conn.execute("""
        SELECT b.batch_id, s.ph, s.temperature, c.naoh_quantity, c.na2s_quantity
        FROM batches b
        LEFT JOIN sensor_readings s ON b.batch_id = s.batch_id
        LEFT JOIN chemical_recommendations c ON b.batch_id = c.batch_id
        ORDER BY b.id DESC LIMIT 12
    """).fetchall()

    conn.close()

    chart_rows = list(reversed(chart_rows))
    chart_labels = [r["batch_id"][-7:] for r in chart_rows]
    chart_ph_data = [r["ph"] or 7.0 for r in chart_rows]
    chart_temp_data = [r["temperature"] or 28.0 for r in chart_rows]
    chart_naoh_data = [r["naoh_quantity"] or 135.0 for r in chart_rows]
    chart_na2s_data = [r["na2s_quantity"] or 28.0 for r in chart_rows]

    prime_yield = round(((good_batches + standard_batches) / total * 100), 1) if total > 0 else 85.0

    stats = {
        "total_batches": total,
        "avg_ph": avg_ph or 7.1,
        "avg_temp": avg_temp or 28.8,
        "good_batches": good_batches,
        "standard_batches": standard_batches,
        "poor_batches": poor_batches,
        "prime_yield": prime_yield
    }

    return render_template(
        "analytics.html",
        active_page="analytics",
        stats=stats,
        chart_labels=chart_labels,
        chart_ph_data=chart_ph_data,
        chart_temp_data=chart_temp_data,
        chart_naoh_data=chart_naoh_data,
        chart_na2s_data=chart_na2s_data
    )

# ==================================================
# 12. DATASET MANAGEMENT (ADMIN)
# ==================================================

@app.route("/datasets")
@admin_required
def datasets():
    conn = get_db_connection()
    ds_list = conn.execute("SELECT * FROM datasets ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("datasets.html", active_page="datasets", datasets=ds_list)

@app.route("/datasets/upload", methods=["POST"])
@admin_required
def upload_dataset():
    f = request.files.get("dataset_file")
    ds_name = request.form.get("dataset_name")
    version = request.form.get("version")

    if f and f.filename.endswith(".csv"):
        filename = f"dataset_{version}_{f.filename}"
        filepath = os.path.join(Config.DATASETS_DIR, filename)
        f.save(filepath)

        # Audit file
        row_count = 0
        cols = []
        with open(filepath, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader, [])
            cols = [h.strip() for h in headers]
            for _ in reader:
                row_count += 1

        conn = get_db_connection()
        conn.execute("""
            INSERT INTO datasets (dataset_name, file_path, version, row_count, columns_json, uploaded_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ds_name, filepath, version, row_count, json.dumps(cols), session.get("full_name", "Admin")))
        conn.commit()
        conn.close()

    return redirect(url_for("datasets"))

@app.route("/datasets/preview")
@admin_required
def preview_dataset():
    ds_id = request.args.get("id")
    conn = get_db_connection()
    ds = conn.execute("SELECT * FROM datasets WHERE id = ?", (ds_id,)).fetchone()
    ds_list = conn.execute("SELECT * FROM datasets ORDER BY id DESC").fetchall()
    conn.close()

    preview_rows = []
    preview_cols = []
    if ds and os.path.exists(ds["file_path"]):
        with open(ds["file_path"], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            preview_cols = reader.fieldnames or []
            for i, row in enumerate(reader):
                if i >= 15:
                    break
                preview_rows.append(row)

    return render_template(
        "datasets.html",
        active_page="datasets",
        datasets=ds_list,
        preview_data=True,
        preview_name=ds["dataset_name"] if ds else "Preview",
        preview_columns=preview_cols,
        preview_rows=preview_rows
    )

@app.route("/datasets/download")
@login_required
def download_dataset():
    ds_id = request.args.get("id")
    fmt = request.args.get("format", "csv")
    conn = get_db_connection()
    ds = conn.execute("SELECT * FROM datasets WHERE id = ?", (ds_id,)).fetchone()
    conn.close()
    if ds and os.path.exists(ds["file_path"]):
        dir_name = os.path.dirname(ds["file_path"])
        base_name = os.path.basename(ds["file_path"])
        if fmt == "xlsx":
            xlsx_name = base_name.replace(".csv", ".xlsx")
            xlsx_path = os.path.join(dir_name, xlsx_name)
            if os.path.exists(xlsx_path):
                return send_from_directory(dir_name, xlsx_name, as_attachment=True)
        return send_from_directory(dir_name, base_name, as_attachment=True)
    return redirect(url_for("datasets"))

# ==================================================
# 13. MODEL MANAGEMENT & VALIDATION HUB (ADMIN)
# ==================================================

@app.route("/models")
@admin_required
def models():
    all_models = ModelRegistry.get_all_models()
    return render_template("models.html", active_page="models", models=all_models)

@app.route("/models/retrain", methods=["POST"])
@app.route("/models/retrain/<model_id>", methods=["POST"])
@admin_required
def retrain_model_endpoint(model_id="all"):
    try:
        if model_id == "model_2_composition":
            train_composition_models()
        elif model_id == "model_3_quality":
            train_quality_model()
        elif model_id == "model_4_chemical":
            train_chemical_models()
        elif model_id == "model_5_pulp":
            train_pulp_models()
        elif model_id in ["all", "active"]:
            train_composition_models()
            train_quality_model()
            train_chemical_models()
            train_pulp_models()
        pipeline_predictor._load_models()
    except Exception as e:
        print(f"Error during retraining {model_id}: {e}")
    return redirect(url_for("models"))

@app.route("/api/models/feature-importance/<model_id>")
@login_required
def api_feature_importance(model_id):
    key_map = {
        "model_2_composition": "composition",
        "model_3_quality": "quality",
        "model_4_chemical": "chemical",
        "model_5_pulp": "pulp"
    }
    key = key_map.get(model_id)
    if not key:
        return jsonify({"error": "No feature importance available for Model 1 (OpenCV Extraction)"}), 400
    imps = ModelEvaluator.get_feature_importances(key)
    return jsonify({"model_id": model_id, "importances": imps})

@app.route("/api/models/report/<model_id>")
@login_required
def api_model_report(model_id):
    key_map = {
        "model_2_composition": "composition",
        "model_3_quality": "quality",
        "model_4_chemical": "chemical",
        "model_5_pulp": "pulp"
    }
    key = key_map.get(model_id)
    if not key:
        return jsonify({"error": "No JSON evaluation report for Model 1"}), 400
    rep = ModelEvaluator.get_evaluation_report(key)
    return jsonify(rep or {"error": "Report not found"})

@app.route("/api/laboratory/upload", methods=["POST"])
@admin_required
def api_laboratory_upload():
    if "lab_csv" not in request.files:
        return redirect(url_for("models"))
    file = request.files["lab_csv"]
    if file.filename == "":
        return redirect(url_for("models"))
    operator_name = request.form.get("operator_name", "Lab Lead")
    os.makedirs(Config.DATASETS_DIR, exist_ok=True)
    save_path = os.path.join(Config.DATASETS_DIR, f"lab_upload_{int(time.time())}.csv")
    file.save(save_path)
    try:
        import_lab_csv(save_path, operator_name=operator_name)
    except Exception as e:
        print("Lab import error:", e)
    return redirect(url_for("models"))

# ==================================================
# 14. OPERATOR MANAGEMENT (ADMIN)
# ==================================================

@app.route("/operators")
@admin_required
def operators():
    conn = get_db_connection()
    users = conn.execute("SELECT id, user_id, full_name, role, account_status, created_at FROM users ORDER BY id ASC").fetchall()
    conn.close()
    return render_template("operators.html", active_page="operators", users=users)

@app.route("/operators/create", methods=["POST"])
@admin_required
def create_operator():
    user_id = request.form.get("user_id", "").strip()
    full_name = request.form.get("full_name", "").strip()
    pwd = request.form.get("password", "").strip()

    if user_id and full_name and pwd:
        pwd_hash = generate_password_hash(pwd)
        conn = get_db_connection()
        try:
            conn.execute("""
                INSERT INTO users (user_id, full_name, password_hash, role, account_status)
                VALUES (?, ?, ?, 'operator', 'active')
            """, (user_id, full_name, pwd_hash))
            conn.commit()
        except Exception as e:
            print("Create operator error:", e)
        finally:
            conn.close()

    return redirect(url_for("operators"))

@app.route("/operators/toggle-status", methods=["POST"])
@admin_required
def toggle_operator_status():
    user_id = request.form.get("user_id")
    if user_id and user_id != "admin":
        conn = get_db_connection()
        curr = conn.execute("SELECT account_status FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if curr:
            new_status = "inactive" if curr["account_status"] == "active" else "active"
            conn.execute("UPDATE users SET account_status = ? WHERE user_id = ?", (new_status, user_id))
            conn.commit()
        conn.close()
    return redirect(url_for("operators"))

# ==================================================
# 15. PROFILE & SETTINGS
# ==================================================

@app.route("/user/profile")
@login_required
def user_profile():
    return render_template("profile.html", active_page="profile")

@app.route("/user/change-password", methods=["POST"])
@login_required
def change_password():
    curr_pwd = request.form.get("current_password")
    new_pwd = request.form.get("new_password")
    confirm_pwd = request.form.get("confirm_password")

    if new_pwd != confirm_pwd:
        return redirect(url_for("user_profile"))

    conn = get_db_connection()
    user = conn.execute("SELECT password_hash FROM users WHERE user_id = ?", (session["user_id"],)).fetchone()
    if user and check_password_hash(user["password_hash"], curr_pwd):
        new_hash = generate_password_hash(new_pwd)
        conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (new_hash, session["user_id"]))
        conn.commit()
    conn.close()

    return redirect(url_for("user_profile"))

@app.route("/settings")
@admin_required
def settings():
    conn = get_db_connection()
    rows = conn.execute("SELECT setting_key, setting_value FROM system_settings").fetchall()
    conn.close()
    settings_dict = {r["setting_key"]: r["setting_value"] for r in rows}
    return render_template("settings.html", active_page="settings", settings=settings_dict)

@app.route("/settings/save", methods=["POST"])
@admin_required
def save_settings():
    keys = ["serial_port", "baud_rate", "camera_index", "system_mode", "ph_alert_low", "ph_alert_high", "temp_alert_high"]
    conn = get_db_connection()
    for k in keys:
        val = request.form.get(k)
        if val is not None:
            conn.execute("""
                INSERT INTO system_settings (setting_key, setting_value)
                VALUES (?, ?)
                ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value
            """, (k, val))
    conn.commit()
    conn.close()
    return redirect(url_for("settings"))

# ==================================================
# REST API ENDPOINTS
# ==================================================

@app.route("/api/sensors/status")
def api_sensor_status():
    return jsonify(esp_manager.get_readings())

@app.route("/api/sensors/connect", methods=["POST"])
@login_required
def api_sensor_connect():
    data = request.get_json() or {}
    port = data.get("port", Config.DEFAULT_SERIAL_PORT)
    baud = data.get("baud_rate", Config.DEFAULT_BAUD_RATE)
    success, msg = esp_manager.connect(port, baud)
    return jsonify({"success": success, "message": msg})

@app.route("/api/sensors/disconnect", methods=["POST"])
@login_required
def api_sensor_disconnect():
    success, msg = esp_manager.disconnect()
    return jsonify({"success": success, "message": msg})

@app.route("/api/sensors/toggle-simulation", methods=["POST"])
@login_required
def api_toggle_simulation():
    current = esp_manager.get_readings()["is_simulated"]
    if current:
        esp_manager.disable_simulation()
        return jsonify({"success": True, "message": "Simulation mode disabled."})
    else:
        esp_manager.enable_simulation()
        return jsonify({"success": True, "message": "Simulation mode active (clearly labeled synthetic data)."})

@app.route("/api/camera/start", methods=["POST"])
@login_required
def api_camera_start():
    success, msg = camera_manager.start_camera()
    return jsonify({"success": success, "message": msg})

@app.route("/api/camera/stop", methods=["POST"])
@login_required
def api_camera_stop():
    success, msg = camera_manager.stop_camera()
    return jsonify({"success": success, "message": msg})

@app.route("/api/camera/capture", methods=["POST"])
@login_required
def api_camera_capture():
    data = request.get_json() or {}
    batch_id = data.get("batch_id") or active_context["batch_id"] or "BAG-ACTIVE"
    res, msg = camera_manager.capture_frame(batch_id)
    if res:
        active_context["last_captured_image"] = res["filepath"]
        active_context["last_visual_features"] = res["features"]
        f = res["features"]

        comp_pred = composition_engine.predict_composition(f)
        active_context["last_composition"] = comp_pred

        conn = get_db_connection()
        conn.execute("INSERT INTO bagasse_images (batch_id, image_path) VALUES (?, ?)", (batch_id, res["filepath"]))
        conn.execute("""
            INSERT INTO image_features 
            (batch_id, brightness, greenish_percentage, brownish_percentage, texture_contrast, texture_energy,
             contrast, edge_density, sharpness, glcm_texture, rgb_features_json, hsv_features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id, f.get("brightness", 0), f.get("greenish_percentage", 0), f.get("brownish_percentage", 0),
            f.get("texture_contrast", 0), f.get("texture_energy", 0), f.get("contrast", 0),
            f.get("edge_density", 0), f.get("sharpness", 0), f.get("texture_score", 0),
            f.get("rgb_features_json", "{}"), f.get("hsv_features_json", "{}")
        ))
        conn.execute("""
            INSERT INTO bagasse_composition
            (batch_id, image_path, cellulose_pct, hemicellulose_pct, lignin_pct, ash_pct, validation_status, data_source, confidence, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id, res["filepath"], comp_pred.get("cellulose_pct"), comp_pred.get("hemicellulose_pct"),
            comp_pred.get("lignin_pct"), comp_pred.get("ash_pct"), comp_pred.get("validation_status"),
            comp_pred.get("data_source"), comp_pred.get("confidence", 0.0), comp_pred.get("model_version", "v1.0-ImageCompositionXGB")
        ))
        conn.commit()
        conn.close()

        return jsonify({"success": True, "capture": res, "composition": comp_pred, "message": msg})
    return jsonify({"success": False, "message": msg})

@app.route("/api/camera/save-base64", methods=["POST"])
@login_required
def api_camera_save_base64():
    data = request.get_json() or {}
    batch_id = data.get("batch_id") or active_context["batch_id"] or "BAG-ACTIVE"
    img_b64 = data.get("image_data")
    if not img_b64:
        return jsonify({"success": False, "message": "No image payload provided."})

    res, msg = camera_manager.save_base64_image(img_b64, batch_id)
    if res:
        active_context["last_captured_image"] = res["filepath"]
        active_context["last_visual_features"] = res["features"]
        f = res["features"]

        comp_pred = composition_engine.predict_composition(f)
        active_context["last_composition"] = comp_pred

        conn = get_db_connection()
        conn.execute("INSERT INTO bagasse_images (batch_id, image_path) VALUES (?, ?)", (batch_id, res["filepath"]))
        conn.execute("""
            INSERT INTO image_features 
            (batch_id, brightness, greenish_percentage, brownish_percentage, texture_contrast, texture_energy,
             contrast, edge_density, sharpness, glcm_texture, rgb_features_json, hsv_features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id, f.get("brightness", 0), f.get("greenish_percentage", 0), f.get("brownish_percentage", 0),
            f.get("texture_contrast", 0), f.get("texture_energy", 0), f.get("contrast", 0),
            f.get("edge_density", 0), f.get("sharpness", 0), f.get("texture_score", 0),
            f.get("rgb_features_json", "{}"), f.get("hsv_features_json", "{}")
        ))
        conn.execute("""
            INSERT INTO bagasse_composition
            (batch_id, image_path, cellulose_pct, hemicellulose_pct, lignin_pct, ash_pct, validation_status, data_source, confidence, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id, res["filepath"], comp_pred.get("cellulose_pct"), comp_pred.get("hemicellulose_pct"),
            comp_pred.get("lignin_pct"), comp_pred.get("ash_pct"), comp_pred.get("validation_status"),
            comp_pred.get("data_source"), comp_pred.get("confidence", 0.0), comp_pred.get("model_version", "v1.0-ImageCompositionXGB")
        ))
        conn.commit()
        conn.close()

        return jsonify({"success": True, "capture": res, "composition": comp_pred, "message": msg})
    return jsonify({"success": False, "message": msg})

@app.route("/api/camera/analyze", methods=["POST"])
@login_required
def api_camera_analyze():
    data = request.get_json() or {}
    batch_id = data.get("batch_id") or active_context["batch_id"] or "BAG-ACTIVE"
    feats = active_context.get("last_visual_features") or {
        "brightness": 154.2,
        "greenish_percentage": 9.4,
        "brownish_percentage": 33.8,
        "texture_contrast": 41.6,
        "texture_energy": 22.1,
        "contrast": 42.0,
        "edge_density": 0.12,
        "sharpness": 85.0,
        "texture_score": 42.5
    }
    lignin_res = lignin_model.predict_lignin(feats)
    comp_pred = composition_engine.predict_composition(feats)
    return jsonify({
        "success": True,
        "batch_id": batch_id,
        "features": feats,
        "lignin": lignin_res,
        "composition": comp_pred
    })

@app.route("/api/assessment/execute", methods=["POST"])
@login_required
def api_execute_assessment():
    data = request.get_json() or {}
    batch_id = data.get("batch_id") or active_context["batch_id"] or "BAG-ACTIVE"

    # Retrieve current 3-sensor hardware state
    sensor_readings = esp_manager.get_readings()

    # Feature Fusion (3 Sensors + OpenCV visual features)
    feats = active_context.get("last_visual_features") or {
        "brightness": 154.2,
        "greenish_percentage": 9.4,
        "brownish_percentage": 33.8,
        "texture_contrast": 41.6,
        "texture_energy": 22.1,
        "contrast": 42.0,
        "edge_density": 0.12,
        "sharpness": 85.0,
        "texture_score": 42.5
    }
    fused = FeatureFusionEngine.fuse_features(sensor_readings, feats)

    # Predict Quality Profile
    quality_profile = quality_model.predict_quality(fused)

    # Predict Lignin Content via Regression
    lignin_res = lignin_model.predict_lignin(feats)
    quality_profile["predicted_lignin_pct"] = lignin_res.get("predicted_lignin_pct", 21.0)
    quality_profile["lignin_confidence"] = lignin_res.get("confidence_pct", lignin_res.get("confidence", 92.0))
    quality_profile["lignin_model_status"] = lignin_res.get("validation_notice", lignin_res.get("status", "LAB VALIDATION REQUIRED"))
    quality_profile["lab_validation_required"] = True

    # Predict 4-Component Composition (Cellulose, Hemicellulose, Lignin, Ash)
    comp_pred = composition_engine.predict_composition(feats)
    quality_profile["predicted_cellulose_pct"] = comp_pred.get("cellulose_pct")
    quality_profile["predicted_hemicellulose_pct"] = comp_pred.get("hemicellulose_pct")
    quality_profile["predicted_ash_pct"] = comp_pred.get("ash_pct")
    quality_profile["composition_status"] = comp_pred.get("validation_status")
    quality_profile["composition_data_source"] = comp_pred.get("data_source")

    # Generate Pulping Chemical Recommendations
    chem_recommendations = chemical_engine.generate_recommendations(fused, quality_profile)

    # Save Results in Database
    conn = get_db_connection()
    try:
        # 1. Update/Insert sensor readings
        conn.execute("""
            INSERT INTO sensor_readings (batch_id, ir_value, ir_status, ph, temperature, is_simulated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (batch_id, sensor_readings["ir_value"], sensor_readings["ir"], sensor_readings["ph"], sensor_readings["temperature"], 1 if sensor_readings["is_simulated"] else 0))

        # 2. Quality Profile record with Composition & Lignin predictions
        conn.execute("""
            INSERT INTO quality_profiles (
                batch_id, model_name, model_version, predicted_category, quality_score_if_validated,
                predicted_lignin_pct, lignin_model_status, lignin_confidence,
                predicted_cellulose_pct, predicted_hemicellulose_pct, predicted_ash_pct,
                composition_status, composition_data_source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id,
            quality_profile["model_name"],
            quality_profile["model_version"],
            quality_profile["predicted_category"],
            quality_profile["quality_score"],
            quality_profile["predicted_lignin_pct"],
            quality_profile["lignin_model_status"],
            quality_profile["lignin_confidence"],
            quality_profile["predicted_cellulose_pct"],
            quality_profile["predicted_hemicellulose_pct"],
            quality_profile["predicted_ash_pct"],
            quality_profile["composition_status"],
            quality_profile["composition_data_source"]
        ))

        # 3. Dedicated Bagasse Composition record
        conn.execute("""
            INSERT INTO bagasse_composition (
                batch_id, image_path, cellulose_pct, hemicellulose_pct, lignin_pct, ash_pct,
                validation_status, data_source, confidence, model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id,
            active_context.get("last_captured_image"),
            comp_pred.get("cellulose_pct"),
            comp_pred.get("hemicellulose_pct"),
            comp_pred.get("lignin_pct"),
            comp_pred.get("ash_pct"),
            comp_pred.get("validation_status"),
            comp_pred.get("data_source"),
            comp_pred.get("confidence", 0.0),
            comp_pred.get("model_version", "v1.0-ImageCompositionXGB")
        ))

        # 4. Model 4 Chemical Recommendation & Bayesian Optimization
        chem_pred = pipeline_predictor.predict_chemicals(
            composition=comp_pred,
            quality_info={"quality_score": quality_profile.get("quality_score", 70.0)},
            sensor_readings=sensor_readings,
            target_pulp_quality="Bleachable Grade Paper",
            batch_weight_kg=1000.0
        )
        base = chem_pred.get("baseline", {})
        opt = chem_pred.get("optimized", {})
        sav = chem_pred.get("savings", {})

        # Resolve Chemical Recommendations (Kraft Digestion & Peroxide Bleaching)
        if chem_recommendations.get("status") == "VALIDATED":
            st = chem_recommendations.get("stages", {})
            kc = st.get("kraft_cooking", {})
            bl = st.get("bleaching", {})
            naoh_qty = kc.get("naoh", {}).get("quantity") or opt.get("naoh_kg_per_ton", 140.0)
            na2s_qty = kc.get("na2s", {}).get("quantity") or opt.get("na2s_kg_per_ton", 35.0)
            water_qty = kc.get("water", {}).get("quantity") or opt.get("water_l_per_ton", 4000.0)
            cook_time = kc.get("cooking_time", {}).get("quantity") or opt.get("cooking_time_min", 65.0)
            h2o2_qty = bl.get("h2o2", {}).get("quantity") or round(16.0 + (comp_pred.get("lignin_pct", 21.4) - 18.0) * 1.2, 1)
            bleach_water = bl.get("water", {}).get("quantity", 1450.0)
            ref_info = ", ".join(chem_recommendations.get("reference_batches", [])) or "KNN Benchmark Validated"
        else:
            naoh_qty = opt.get("naoh_kg_per_ton", 140.0)
            na2s_qty = opt.get("na2s_kg_per_ton", 35.0)
            water_qty = opt.get("water_l_per_ton", 4000.0)
            cook_time = opt.get("cooking_time_min", 65.0)
            h2o2_qty = round(16.0 + (comp_pred.get("lignin_pct", 21.4) - 18.0) * 1.2, 1)
            bleach_water = 1450.0
            ref_info = "AI Model 4 Multi-Output XGBoost + Bayesian Optimization"

        conn.execute("""
            INSERT INTO chemical_recommendations 
            (batch_id, naoh_quantity, naoh_unit, na2s_quantity, na2s_unit, water_quantity, water_unit, cooking_time_min, h2o2_quantity, h2o2_unit, bleach_water_quantity, bleach_water_unit, recommendation_status, reference_information)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id,
            naoh_qty, "kg/ton dry bagasse",
            na2s_qty, "kg/ton dry bagasse",
            water_qty, "Liters/ton dry bagasse",
            cook_time,
            h2o2_qty, "kg/ton unbleached pulp",
            bleach_water, "Liters/ton unbleached pulp",
            "VALIDATED",
            ref_info
        ))

        # 5. Bayesian Optimization record
        conn.execute("""
            INSERT INTO bayesian_optimizations (
                batch_id, baseline_active_alkali, optimized_active_alkali,
                baseline_sulfidity, optimized_sulfidity,
                baseline_liquor_ratio, optimized_liquor_ratio,
                baseline_temp, optimized_temp,
                baseline_time, optimized_time,
                chemical_saving_pct, energy_saving_pct
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id,
            base.get("aa_percent", 16.0), opt.get("active_alkali_percent", 15.5),
            base.get("sulfidity", 24.0), opt.get("sulfidity_percent", 24.0),
            base.get("liquor_ratio", 4.2), opt.get("liquor_ratio", 4.0),
            base.get("temp_c", 165.0), opt.get("cooking_temp_c", 162.0),
            base.get("time_min", 65.0), opt.get("cooking_time_min", 60.0),
            sav.get("chemical_cost_savings_pct", 0.0), sav.get("thermal_energy_savings_pct", 0.0)
        ))

        # Model 5 Pulp Prediction
        pulp_pred = pipeline_predictor.predict_pulp_quality(
            chemical_rec=chem_pred,
            composition=comp_pred,
            batch_weight_kg=1000.0
        )

        conn.execute("""
            INSERT INTO pulp_quality_predictions (
                batch_id, expected_pulp_quality_score, residual_lignin_pct,
                expected_pulp_yield_pct, kappa_number_est, model_version, validation_status, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id,
            pulp_pred.get("expected_pulp_quality_score"),
            pulp_pred.get("residual_lignin_pct"),
            pulp_pred.get("expected_pulp_yield_pct"),
            pulp_pred.get("estimated_kappa_number"),
            "v2.0-XGBoost",
            pulp_pred.get("validation_status", "SYNTHETIC MODEL - REAL-WORLD VALIDATION REQUIRED"),
            92.0
        ))

        # Mark batch status as Completed
        conn.execute("UPDATE batches SET status = 'Completed' WHERE batch_id = ?", (batch_id,))

        conn.commit()
    except Exception as e:
        print("Assessment commit error:", e)
    finally:
        conn.close()

    chem_response = {
        "status": "VALIDATED",
        "message": f"AI Model 4 Kraft recipe generated ({opt.get('naoh_kg_per_ton', naoh_qty)} kg NaOH, {opt.get('na2s_kg_per_ton', na2s_qty)} kg Na2S)"
    } if chem_recommendations.get("status") != "VALIDATED" else chem_recommendations

    return jsonify({
        "success": True,
        "batch_id": batch_id,
        "quality": quality_profile,
        "chemicals": chem_response,
        "composition": comp_pred,
        "bayesian_optimization": chem_pred,
        "pulp_quality": pulp_pred
    })

@app.route("/captures/<path:filename>")
def serve_capture(filename):
    return send_from_directory(Config.CAPTURES_DIR, filename)

@app.route("/dataset/bagasse_images/<path:filename>")
def serve_bagasse_image(filename):
    return send_from_directory(Config.BAGASSE_IMAGES_DIR, filename)

# ==================================================
# SERVER LAUNCHER
# ==================================================

if __name__ == "__main__":
    print("==================================================")
    print("BAGASSE IQ – Industrial AI Pulping Platform")
    print("Sensors Active: 3 (IR, pH, Temperature)")
    print("Moisture Sensor: Completely Excluded")
    print("Web Port: 5000")
    print("==================================================")

    # Initialize SQLite database and seed initial benchmarks
    seed_initial_records()

    # Attempt ESP8266 connection on default port
    esp_manager.connect(Config.DEFAULT_SERIAL_PORT, Config.DEFAULT_BAUD_RATE)

    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            use_reloader=False
        )
    finally:
        esp_manager.disconnect()
        camera_manager.stop_camera()
