from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
RAW_FILE = DATA_DIR / "Principle_Data_Scientist_Tech_Assessment.xlsx"
SHEET_NAME = "daily records"

FORECAST_HORIZON = 90
TARGET_COL = "complaints"
DATE_COL = "date"