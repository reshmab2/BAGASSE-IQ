import sqlite3
import os
import json
from datetime import datetime
from werkzeug.security import generate_password_hash
from config import Config

def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        user_id TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('admin', 'operator')),
        account_status TEXT DEFAULT 'active' CHECK(account_status IN ('active', 'inactive')),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Batches Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT UNIQUE NOT NULL,
        operator_id TEXT NOT NULL,
        operator_name TEXT NOT NULL,
        batch_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'Created',
        remarks TEXT
    );
    """)

    # 3. Sensor Readings Table (EXACTLY THREE SENSORS: IR, pH, Temperature - NO MOISTURE)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        ir_value INTEGER DEFAULT 0,
        ir_status TEXT DEFAULT 'Not Detected',
        ph REAL DEFAULT 7.0,
        temperature REAL DEFAULT 25.0,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_simulated INTEGER DEFAULT 0
    );
    """)

    # 4. Bagasse Images Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bagasse_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        image_path TEXT NOT NULL,
        image_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 5. Image Features Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS image_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        brightness REAL DEFAULT 0.0,
        greenish_percentage REAL DEFAULT 0.0,
        brownish_percentage REAL DEFAULT 0.0,
        texture_contrast REAL DEFAULT 0.0,
        texture_energy REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 6. Quality Profiles Table (with AI Lignin Prediction)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quality_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        predicted_category TEXT NOT NULL,
        quality_score_if_validated REAL,
        predicted_lignin_pct REAL DEFAULT 21.0,
        lignin_model_status TEXT DEFAULT 'LAB VALIDATION REQUIRED',
        lignin_confidence REAL DEFAULT 92.0,
        lab_lignin_ref REAL,
        assessment_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 7. Chemical Recommendations Table (Kraft Cooking & Bleaching Stages)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chemical_recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        version_number INTEGER DEFAULT 1,
        batch_weight_kg REAL DEFAULT 1000.0,
        target_pulp_quality TEXT DEFAULT 'Bleachable Grade Paper',
        naoh_quantity REAL,
        naoh_unit TEXT DEFAULT 'kg/ton',
        na2s_quantity REAL,
        na2s_unit TEXT DEFAULT 'kg/ton',
        water_quantity REAL,
        water_unit TEXT DEFAULT 'L/ton',
        cooking_time_min REAL DEFAULT 70.0,
        h2o2_quantity REAL,
        h2o2_unit TEXT DEFAULT 'kg/ton',
        bleach_water_quantity REAL DEFAULT 1450.0,
        bleach_water_unit TEXT DEFAULT 'L/ton',
        recommendation_status TEXT NOT NULL,
        reference_information TEXT,
        explanation_json TEXT,
        input_parameters_json TEXT,
        is_active_version INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 8. Datasets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        version TEXT NOT NULL,
        row_count INTEGER DEFAULT 0,
        columns_json TEXT,
        uploaded_by TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 9. Models Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        dataset_version TEXT NOT NULL,
        evaluation_metrics_json TEXT,
        model_path TEXT,
        is_active INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 10. System Settings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        setting_key TEXT UNIQUE NOT NULL,
        setting_value TEXT NOT NULL
    );
    """)

    # 11. Bagasse Image Composition Predictions Table (Cellulose, Hemicellulose, Lignin, Ash)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bagasse_composition (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        image_path TEXT,
        cellulose_pct REAL,
        hemicellulose_pct REAL,
        lignin_pct REAL,
        ash_pct REAL,
        validation_status TEXT DEFAULT 'COMPOSITION MODEL NOT TRAINED – LAB DATA REQUIRED',
        data_source TEXT DEFAULT 'AI Prediction',
        confidence REAL DEFAULT 0.0,
        model_version TEXT DEFAULT 'v1.0-XGBoost',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 12. Pulp Quality Predictions Table (Model 5: Residual Lignin, Yield, Pulp Quality Score)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pulp_quality_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        expected_pulp_quality_score REAL,
        residual_lignin_pct REAL,
        expected_pulp_yield_pct REAL,
        kappa_number_est REAL,
        model_version TEXT DEFAULT 'v1.0-XGBoost',
        validation_status TEXT DEFAULT 'SYNTHETIC MODEL – REAL-WORLD VALIDATION REQUIRED',
        confidence REAL DEFAULT 92.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 13. Actual Laboratory Pulp Results Table (Tracking Actual vs Predicted)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS actual_pulp_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        actual_pulp_yield_pct REAL,
        actual_residual_lignin_pct REAL,
        actual_kappa_number REAL,
        actual_tensile_index REAL,
        actual_burst_index REAL,
        lab_operator TEXT,
        notes TEXT,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 14. Bayesian Parameter Optimization Records Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bayesian_optimizations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        baseline_active_alkali REAL,
        optimized_active_alkali REAL,
        baseline_sulfidity REAL,
        optimized_sulfidity REAL,
        baseline_liquor_ratio REAL,
        optimized_liquor_ratio REAL,
        baseline_temp REAL,
        optimized_temp REAL,
        baseline_time REAL,
        optimized_time REAL,
        chemical_saving_pct REAL,
        energy_saving_pct REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Schema Migrations for Existing Databases
    migrations = [
        "ALTER TABLE quality_profiles ADD COLUMN predicted_lignin_pct REAL DEFAULT 21.0",
        "ALTER TABLE quality_profiles ADD COLUMN lignin_model_status TEXT DEFAULT 'LAB VALIDATION REQUIRED'",
        "ALTER TABLE quality_profiles ADD COLUMN lignin_confidence REAL DEFAULT 92.0",
        "ALTER TABLE quality_profiles ADD COLUMN lab_lignin_ref REAL",
        "ALTER TABLE quality_profiles ADD COLUMN predicted_cellulose_pct REAL DEFAULT 43.5",
        "ALTER TABLE quality_profiles ADD COLUMN predicted_hemicellulose_pct REAL DEFAULT 28.0",
        "ALTER TABLE quality_profiles ADD COLUMN predicted_ash_pct REAL DEFAULT 2.5",
        "ALTER TABLE quality_profiles ADD COLUMN composition_status TEXT DEFAULT 'COMPOSITION MODEL NOT TRAINED – LAB DATA REQUIRED'",
        "ALTER TABLE quality_profiles ADD COLUMN composition_data_source TEXT DEFAULT 'AI Prediction'",
        "ALTER TABLE image_features ADD COLUMN contrast REAL DEFAULT 42.0",
        "ALTER TABLE image_features ADD COLUMN edge_density REAL DEFAULT 0.12",
        "ALTER TABLE image_features ADD COLUMN sharpness REAL DEFAULT 85.0",
        "ALTER TABLE image_features ADD COLUMN glcm_texture REAL DEFAULT 45.0",
        "ALTER TABLE image_features ADD COLUMN rgb_features_json TEXT",
        "ALTER TABLE image_features ADD COLUMN hsv_features_json TEXT",
        "ALTER TABLE chemical_recommendations ADD COLUMN cooking_time_min REAL DEFAULT 70.0",
        "ALTER TABLE chemical_recommendations ADD COLUMN bleach_water_quantity REAL DEFAULT 1450.0",
        "ALTER TABLE chemical_recommendations ADD COLUMN bleach_water_unit TEXT DEFAULT 'L/ton'",
        "ALTER TABLE chemical_recommendations ADD COLUMN version_number INTEGER DEFAULT 1",
        "ALTER TABLE chemical_recommendations ADD COLUMN batch_weight_kg REAL DEFAULT 1000.0",
        "ALTER TABLE chemical_recommendations ADD COLUMN target_pulp_quality TEXT DEFAULT 'Bleachable Grade Paper'",
        "ALTER TABLE chemical_recommendations ADD COLUMN explanation_json TEXT",
        "ALTER TABLE chemical_recommendations ADD COLUMN input_parameters_json TEXT",
        "ALTER TABLE chemical_recommendations ADD COLUMN is_active_version INTEGER DEFAULT 1"
    ]
    for mig in migrations:
        try:
            cursor.execute(mig)
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Seed Default Users
    cursor.execute("SELECT id FROM users WHERE user_id = 'admin'")
    if not cursor.fetchone():
        admin_hash = generate_password_hash("Admin@123")
        cursor.execute("""
            INSERT INTO users (full_name, user_id, password_hash, role, account_status)
            VALUES (?, ?, ?, 'admin', 'active')
        """, ("System Administrator", "admin", admin_hash))

    cursor.execute("SELECT id FROM users WHERE user_id = 'operator'")
    if not cursor.fetchone():
        op_hash = generate_password_hash("Operator@123")
        cursor.execute("""
            INSERT INTO users (full_name, user_id, password_hash, role, account_status)
            VALUES (?, ?, ?, 'operator', 'active')
        """, ("Lead Plant Operator", "operator", op_hash))

    # Seed Default System Settings
    default_settings = [
        ("serial_port", Config.DEFAULT_SERIAL_PORT),
        ("baud_rate", str(Config.DEFAULT_BAUD_RATE)),
        ("camera_index", str(Config.DEFAULT_CAMERA_INDEX)),
        ("system_mode", "LIVE_HARDWARE"),
        ("ph_alert_low", "5.0"),
        ("ph_alert_high", "9.0"),
        ("temp_alert_high", "65.0")
    ]
    for key, val in default_settings:
        cursor.execute("""
            INSERT OR IGNORE INTO system_settings (setting_key, setting_value)
            VALUES (?, ?)
        """, (key, val))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database schema successfully initialized and migrated.")
