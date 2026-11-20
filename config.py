import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "bagasse-iq-secure-secret-key-2026")
    DATABASE_PATH = os.path.join(BASE_DIR, "bagasse_iq.db")
    CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
    BAGASSE_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "bagasse_images")
    DATASETS_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Hardware Defaults
    DEFAULT_SERIAL_PORT = "COM3"
    DEFAULT_BAUD_RATE = 115200
    DEFAULT_CAMERA_INDEX = 0
    
    # Exactly THREE sensors
    VALID_SENSORS = ["ir", "ph", "temperature"]
    
    # Acceptable industrial range bounds for bagasse pulping
    PH_MIN = 1.0
    PH_MAX = 14.0
    TEMP_MIN = 10.0   # °C
    TEMP_MAX = 100.0  # °C

# Ensure required directories exist
os.makedirs(Config.CAPTURES_DIR, exist_ok=True)
os.makedirs(Config.BAGASSE_IMAGES_DIR, exist_ok=True)
os.makedirs(Config.DATASETS_DIR, exist_ok=True)
os.makedirs(Config.MODELS_DIR, exist_ok=True)
