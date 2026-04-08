from pathlib import Path
import json
import logging
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ============================================================
# APP SETUP
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Retail Forecasting API",
    description="XGBoost-based revenue forecasting API",
    version="4.0.0"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "xgboost_model.pkl"
PIPELINE_FILE = BASE_DIR / "pipeline.pkl"
COLUMNS_FILE = BASE_DIR / "training_columns.pkl"
META_FILE = BASE_DIR / "model_metadata.json"
DATA_FILE = BASE_DIR / "retail_forecasting_70000.xlsx"

# ============================================================
# LOAD ARTIFACTS
# ============================================================
model = None
pipeline = None
training_columns = None
metadata = {}
model_error = None

try:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    training_columns = joblib.load(COLUMNS_FILE)

    if META_FILE.exists():
        with open(META_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    logger.info("Model artifacts loaded successfully.")
except Exception as e:
    model_error = f"Artifact load failed: {e}"
    logger.exception(model_error)

# ============================================================
# LOAD HISTORY
# ============================================================
history_df = pd.DataFrame()
history_error = None
VALID_CATEGORIES = []

try:
    sales_df = pd.read_excel(DATA_FILE)
    sales_df.columns = sales_df.columns.str.strip()
    sales_df["Date"] = pd.to_datetime(sales_df["Date"])

    required_cols = [
        "Date", "StoreID", "Category", "Revenue",
        "UnitsSold", "UnitPrice", "DiscountApplied", "HolidayFlag"
    ]
    missing = [c for c in required_cols if c not in sales_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in history file: {missing}")

    sales_df["StoreID"] = sales_df["StoreID"].astype(str)
    sales_df["Category"] = sales_df["Category"].astype(str)

    VALID_CATEGORIES = sorted(sales_df["Category"].dropna().unique().tolist())

    history_df = sales_df.groupby(["Date", "StoreID", "Category"], as_index=False).agg({
        "Revenue": "sum",
        "UnitsSold": "sum",
        "UnitPrice": "mean",
        "DiscountApplied": "mean",
        "HolidayFlag": "max"
    })

    history_df = history_df.sort_values(["StoreID", "Category", "Date"]).reset_index(drop=True)
    logger.info("History loaded successfully: %s rows", len(history_df))
except Exception as e:
    history_error = str(e)
    logger.exception("History load failed: %s", history_error)

# ============================================================
# HELPERS
# ============================================================
def ensure_ready():
    if model_error:
        raise HTTPException(status_code=503, detail=model_error)
    if history_error:
        raise HTTPException(status_code=503, detail=history_error)
    if model is None or pipeline is None or training_columns is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

def get_group_history(df: pd.DataFrame, store_id: str, category: str, before_date: pd.Timestamp) -> pd.DataFrame:
    temp = df[
        (df["StoreID"] == str(store_id)) &
        (df["Category"] == str(category)) &
        (df["Date"] < before_date)
    ].sort_values("Date")
    return temp.tail(30)

def build_features(
    history_source: pd.DataFrame,
    input_date: pd.Timestamp,
    store_id: str,
    category: str,
    unit_price: float,
    units_sold: int,
    discount_applied: float,
    holiday_flag: int
) -> pd.DataFrame:

    recent_rows = get_group_history(history_source, store_id, category, input_date)

    row = {
        "UnitsSold": units_sold,
        "UnitPrice": unit_price,
        "DiscountApplied": discount_applied,
        "HolidayFlag": holiday_flag,
        "year": input_date.year,
        "month": input_date.month,
        "day": input_date.day,
        "day_of_week": input_date.dayofweek,
        "week_of_year": int(input_date.isocalendar().week),
        "lag_1": float(recent_rows["Revenue"].iloc[-1]) if len(recent_rows) >= 1 else 0.0,
        "lag_7": float(recent_rows["Revenue"].iloc[-7]) if len(recent_rows) >= 7 else 0.0,
        "lag_30": float(recent_rows["Revenue"].iloc[-30]) if len(recent_rows) >= 30 else 0.0,
        "rolling_mean_7": float(recent_rows.tail(7)["Revenue"].mean()) if len(recent_rows) >= 1 else 0.0,
        "rolling_mean_30": float(recent_rows["Revenue"].mean()) if len(recent_rows) >= 1 else 0.0,
    }

    for col in training_columns:
        if col.startswith("StoreID_"):
            row[col] = 1 if col == f"StoreID_{store_id}" else 0
        elif col.startswith("Category_"):
            row[col] = 1 if col == f"Category_{category}" else 0

    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=training_columns, fill_value=0)
    return df_row

# ============================================================
# SCHEMAS
# ============================================================
class ForecastInput(BaseModel):
    date: str
    store_id: int
    category: str
    unit_price: float
    units_sold: int
    discount_applied: float
    holiday_flag: int

    @field_validator("store_id")
    @classmethod
    def validate_store_id(cls, v):
        if v < 1:
            raise ValueError("store_id must be >= 1")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        if VALID_CATEGORIES and v not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {VALID_CATEGORIES}")
        return v

    @field_validator("unit_price")
    @classmethod
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError("unit_price must be > 0")
        return v

    @field_validator("units_sold")
    @classmethod
    def validate_units(cls, v):
        if v < 0:
            raise ValueError("units_sold must be >= 0")
        return v

    @field_validator("discount_applied")
    @classmethod
    def validate_discount(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("discount_applied must be between 0 and 100")
        return v

    @field_validator("holiday_flag")
    @classmethod
    def validate_holiday(cls, v):
        if v not in [0, 1]:
            raise ValueError("holiday_flag must be 0 or 1")
        return v

class CategoryInput(BaseModel):
    date: str
    store_id: int
    unit_price: float
    units_sold: int
    discount_applied: float
    holiday_flag: int

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def home():
    return {"message": "Retail Forecasting API v4.0 running"}

@app.get("/health")
def health():
    return {
        "status": "ok" if not model_error and not history_error else "degraded",
        "model_loaded": model_error is None,
        "history_loaded": history_error is None,
        "history_rows": int(len(history_df)) if history_error is None else 0,
        "valid_categories": VALID_CATEGORIES,
        "model_error": model_error,
        "history_error": history_error
    }

@app.get("/model-info")
def model_info():
    ensure_ready()
    return {
        "model": metadata.get("model_name", "Tuned XGBoost"),
        "target": metadata.get("target", "Revenue"),
        "metrics": metadata.get("final_metrics", {}),
        "feature_count": metadata.get("feature_count", len(training_columns)),
        "status": "production-ready"
    }

@app.get("/history")
def get_history(
    store_id: int,
    category: str,
    days: int = 30,
    before_date: Optional[str] = None
):
    ensure_ready()

    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Use one of {VALID_CATEGORIES}")

    filtered = history_df[
        (history_df["StoreID"] == str(store_id)) &
        (history_df["Category"] == category)
    ].copy()

    if before_date:
        cutoff = pd.to_datetime(before_date)
        filtered = filtered[filtered["Date"] < cutoff]

    filtered = filtered.sort_values("Date").tail(days)

    data = [
        {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "revenue": round(float(row["Revenue"]), 2),
            "units_sold": int(row["UnitsSold"]),
            "unit_price": round(float(row["UnitPrice"]), 2),
            "discount_applied": round(float(row["DiscountApplied"]), 2),
            "holiday_flag": int(row["HolidayFlag"])
        }
        for _, row in filtered.iterrows()
    ]

    return {
        "store_id": store_id,
        "category": category,
        "records_found": len(data),
        "data": data
    }

@app.post("/predict")
def predict(data: ForecastInput):
    ensure_ready()

    try:
        input_date = pd.to_datetime(data.date)

        features = build_features(
            history_source=history_df,
            input_date=input_date,
            store_id=str(data.store_id),
            category=data.category,
            unit_price=data.unit_price,
            units_sold=data.units_sold,
            discount_applied=data.discount_applied,
            holiday_flag=data.holiday_flag
        )

        features_scaled = pipeline.transform(features)
        prediction = float(model.predict(features_scaled)[0])

        recent_rows = get_group_history(history_df, str(data.store_id), data.category, input_date).tail(7)
        recent_avg = float(recent_rows["Revenue"].mean()) if len(recent_rows) > 0 else None

        return {
            "date": data.date,
            "store_id": data.store_id,
            "category": data.category,
            "predicted_revenue": round(prediction, 2),
            "recent_7day_avg": round(recent_avg, 2) if recent_avg is not None else None,
            "model": metadata.get("model_name", "Tuned XGBoost")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast")
def forecast(
    days: int = 7,
    from_date: Optional[str] = None,
    store_id: int = 1,
    category: str = "Groceries",
    unit_price: float = 100.0,
    units_sold: int = 50,
    discount_applied: float = 10.0,
    holiday_flag: int = 0
):
    ensure_ready()

    if days < 1 or days > 90:
        raise HTTPException(status_code=400, detail="days must be between 1 and 90")

    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Use one of {VALID_CATEGORIES}")

    temp_history = history_df.copy()
    start_date = pd.to_datetime(from_date) if from_date else pd.Timestamp.today().normalize()

    predictions = []

    for i in range(days):
        target_date = start_date + pd.Timedelta(days=i)

        features = build_features(
            history_source=temp_history,
            input_date=target_date,
            store_id=str(store_id),
            category=category,
            unit_price=unit_price,
            units_sold=units_sold,
            discount_applied=discount_applied,
            holiday_flag=holiday_flag
        )

        features_scaled = pipeline.transform(features)
        pred = float(model.predict(features_scaled)[0])

        predictions.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "day_of_week": target_date.strftime("%A"),
            "predicted_revenue": round(pred, 2)
        })

        # recursive update
        temp_history = pd.concat([
            temp_history,
            pd.DataFrame([{
                "Date": target_date,
                "StoreID": str(store_id),
                "Category": category,
                "Revenue": pred,
                "UnitsSold": units_sold,
                "UnitPrice": unit_price,
                "DiscountApplied": discount_applied,
                "HolidayFlag": holiday_flag
            }])
        ], ignore_index=True)

    total = sum(p["predicted_revenue"] for p in predictions)

    return {
        "store_id": store_id,
        "category": category,
        "forecast_days": days,
        "predictions": predictions,
        "total_predicted": round(total, 2),
        "avg_daily_predicted": round(total / days, 2)
    }

@app.post("/category-comparison")
def category_comparison(data: CategoryInput):
    ensure_ready()

    try:
        input_date = pd.to_datetime(data.date)
        results = []

        for cat in VALID_CATEGORIES:
            features = build_features(
                history_source=history_df,
                input_date=input_date,
                store_id=str(data.store_id),
                category=cat,
                unit_price=data.unit_price,
                units_sold=data.units_sold,
                discount_applied=data.discount_applied,
                holiday_flag=data.holiday_flag
            )

            features_scaled = pipeline.transform(features)
            pred = float(model.predict(features_scaled)[0])

            results.append({
                "category": cat,
                "predicted_revenue": round(pred, 2)
            })

        results = sorted(results, key=lambda x: x["predicted_revenue"], reverse=True)

        return {
            "date": data.date,
            "store_id": data.store_id,
            "comparisons": results,
            "best_category": results[0]["category"],
            "best_revenue": results[0]["predicted_revenue"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))