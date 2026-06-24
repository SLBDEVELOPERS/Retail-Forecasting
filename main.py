from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("retail_forecasting_api")

app = FastAPI(
    title="Retail Forecasting API",
    description="Production-ready FastAPI backend for revenue forecasting.",
    version="5.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "xgboost_model.pkl"
PIPELINE_FILE = BASE_DIR / "pipeline.pkl"
TRAINING_COLUMNS_FILE = BASE_DIR / "training_columns.pkl"
METADATA_FILE = BASE_DIR / "model_metadata.json"
HISTORY_FILE = BASE_DIR / "retail_forecasting_70000.xlsx"


class AppState:
    def __init__(self) -> None:
        self.model: Any = None
        self.pipeline: Any = None
        self.training_columns: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.history_df: pd.DataFrame = pd.DataFrame()
        self.available_categories: set[str] = set()
        self.available_store_ids: set[int] = set()
        self.artifact_errors: dict[str, str] = {}
        self.history_error: Optional[str] = None

    @property
    def model_ready(self) -> bool:
        return (
            self.model is not None
            and self.pipeline is not None
            and bool(self.training_columns)
            and not self.artifact_errors
        )

    @property
    def history_ready(self) -> bool:
        return self.history_error is None and not self.history_df.empty

    @property
    def app_ready(self) -> bool:
        return self.model_ready and self.history_ready


state = AppState()


def safe_joblib_load(file_path: Path, label: str) -> Any:
    try:
        loaded = joblib.load(file_path)
        logger.info("Loaded %s from %s", label, file_path.name)
        return loaded
    except Exception as exc:
        state.artifact_errors[label] = str(exc)
        logger.exception("Failed to load %s from %s", label, file_path)
        return None


def load_model_artifacts() -> None:
    # Load each artifact independently so /health can report partial readiness.
    state.model = safe_joblib_load(MODEL_FILE,  "model")
    state.pipeline = safe_joblib_load(PIPELINE_FILE, "pipeline")

    training_columns = safe_joblib_load(TRAINING_COLUMNS_FILE, "training_columns")
    if isinstance(training_columns, list):
        state.training_columns = [str(column) for column in training_columns]
    elif training_columns is not None:
        state.artifact_errors["training_columns"] = "training_columns.pkl must contain a list of columns"
        logger.error(state.artifact_errors["training_columns"])

    if METADATA_FILE.exists():
        try:
            state.metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
            logger.info("Loaded metadata from %s", METADATA_FILE.name)
        except Exception as exc:
            state.artifact_errors["metadata"] = str(exc)
            logger.exception("Failed to load metadata from %s", METADATA_FILE)
    else:
        logger.warning("Metadata file not found at %s", METADATA_FILE)


def load_history() -> None:
    try:
        raw_df = pd.read_excel(HISTORY_FILE)
        raw_df.columns = raw_df.columns.str.strip()

        required_columns = {
            "Date",
            "StoreID",
            "Category",
            "Revenue",
            "UnitsSold",
            "UnitPrice",
            "DiscountApplied",
            "HolidayFlag",
        }
        missing_columns = sorted(required_columns.difference(raw_df.columns))
        if missing_columns:
            raise ValueError(f"Missing required columns in history file: {missing_columns}")

        raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce")
        if raw_df["Date"].isna().any():
            raise ValueError("History file contains invalid Date values")

        raw_df["StoreID"] = pd.to_numeric(raw_df["StoreID"], errors="coerce")
        if raw_df["StoreID"].isna().any():
            raise ValueError("History file contains invalid StoreID values")

        raw_df["StoreID"] = raw_df["StoreID"].astype(int)
        raw_df["Category"] = raw_df["Category"].astype(str).str.strip()

        # Match the training grain: daily totals by date, store, and category.
        aggregated_df = (
            raw_df.groupby(["Date", "StoreID", "Category"], as_index=False)
            .agg(
                {
                    "Revenue": "sum",
                    "UnitsSold": "sum",
                    "UnitPrice": "mean",
                    "DiscountApplied": "mean",
                    "HolidayFlag": "max",
                }
            )
            .sort_values(["StoreID", "Category", "Date"])
            .reset_index(drop=True)
        )

        state.history_df = aggregated_df
        state.available_categories = set(aggregated_df["Category"].dropna().unique().tolist())
        state.available_store_ids = set(aggregated_df["StoreID"].dropna().astype(int).unique().tolist())

        logger.info(
            "History loaded successfully from %s with %s aggregated rows",
            HISTORY_FILE.name,
            len(aggregated_df),
        )
    except Exception as exc:
        state.history_error = str(exc)
        state.history_df = pd.DataFrame()
        logger.exception("Failed to load history from %s", HISTORY_FILE)


def initialize_app_state() -> None:
    load_model_artifacts()
    load_history()


initialize_app_state()


def ensure_model_ready() -> None:
    if not state.model_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model artifacts are not ready",
                "errors": state.artifact_errors,
            },
        )


def ensure_history_ready() -> None:
    if not state.history_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "History data is not ready",
                "error": state.history_error,
            },
        )


def ensure_app_ready() -> None:
    ensure_model_ready()
    ensure_history_ready()


def validate_category(category: str) -> str:
    normalized = category.strip()
    if normalized not in state.available_categories:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid category",
                "available_categories": sorted(state.available_categories),
            },
        )
    return normalized


def parse_date_input(value: str, field_name: str = "date") -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": f"Invalid {field_name}. Expected ISO date format.", "error": str(exc)},
        ) from exc

    if pd.isna(parsed):
        raise HTTPException(
            status_code=400,
            detail={"message": f"Invalid {field_name}. Expected ISO date format."},
        )

    return parsed.normalize()


def get_group_history(
    history_source: pd.DataFrame,
    store_id: int,
    category: str,
    before_date: pd.Timestamp,
) -> pd.DataFrame:
    return (
        history_source[
            (history_source["StoreID"] == store_id)
            & (history_source["Category"] == category)
            & (history_source["Date"] < before_date)
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )


def build_feature_row(
    history_source: pd.DataFrame,
    target_date: pd.Timestamp,
    store_id: int,
    category: str,
    unit_price: float,
    units_sold: int,
    discount_applied: float,
    holiday_flag: int,
) -> pd.DataFrame:
    recent_history = get_group_history(history_source, store_id, category, target_date)

    # Start from the saved training schema so every inference row matches the model input exactly.
    feature_row = {column: 0 for column in state.training_columns}
    feature_row.update(
        {
            "UnitsSold": float(units_sold),
            "UnitPrice": float(unit_price),
            "DiscountApplied": float(discount_applied),
            "HolidayFlag": int(holiday_flag),
            "year": int(target_date.year),
            "month": int(target_date.month),
            "day": int(target_date.day),
            "day_of_week": int(target_date.dayofweek),
            "week_of_year": int(target_date.isocalendar().week),
            "lag_1": float(recent_history["Revenue"].iloc[-1]) if len(recent_history) >= 1 else 0.0,
            "lag_7": float(recent_history["Revenue"].iloc[-7]) if len(recent_history) >= 7 else 0.0,
            "lag_30": float(recent_history["Revenue"].iloc[-30]) if len(recent_history) >= 30 else 0.0,
            "rolling_mean_7": float(recent_history["Revenue"].tail(7).mean()) if len(recent_history) >= 1 else 0.0,
            "rolling_mean_30": float(recent_history["Revenue"].tail(30).mean()) if len(recent_history) >= 1 else 0.0,
        }
    )

    store_column = f"StoreID_{store_id}"
    category_column = f"Category_{category}"
    if store_column in feature_row:
        feature_row[store_column] = 1
    if category_column in feature_row:
        feature_row[category_column] = 1

    return pd.DataFrame([feature_row], columns=state.training_columns)


def predict_revenue(
    history_source: pd.DataFrame,
    target_date: pd.Timestamp,
    store_id: int,
    category: str,
    unit_price: float,
    units_sold: int,
    discount_applied: float,
    holiday_flag: int,
) -> tuple[float, dict[str, float]]:
    features = build_feature_row(
        history_source=history_source,
        target_date=target_date,
        store_id=store_id,
        category=category,
        unit_price=unit_price,
        units_sold=units_sold,
        discount_applied=discount_applied,
        holiday_flag=holiday_flag,
    )

    transformed_features = state.pipeline.transform(features)
    prediction = float(state.model.predict(transformed_features)[0])

    recent_history = get_group_history(history_source, store_id, category, target_date)
    feature_snapshot = {
        "lag_1": float(features.at[0, "lag_1"]) if "lag_1" in features.columns else 0.0,
        "lag_7": float(features.at[0, "lag_7"]) if "lag_7" in features.columns else 0.0,
        "lag_30": float(features.at[0, "lag_30"]) if "lag_30" in features.columns else 0.0,
        "rolling_mean_7": float(features.at[0, "rolling_mean_7"]) if "rolling_mean_7" in features.columns else 0.0,
        "rolling_mean_30": float(features.at[0, "rolling_mean_30"]) if "rolling_mean_30" in features.columns else 0.0,
        "recent_history_points": int(len(recent_history)),
    }
    return prediction, feature_snapshot


def append_prediction_to_history(
    history_source: pd.DataFrame,
    target_date: pd.Timestamp,
    store_id: int,
    category: str,
    revenue: float,
    unit_price: float,
    units_sold: int,
    discount_applied: float,
    holiday_flag: int,
) -> pd.DataFrame:
    # Recursive forecasting depends on feeding each predicted revenue back into temporary history.
    next_row = pd.DataFrame(
        [
            {
                "Date": target_date,
                "StoreID": store_id,
                "Category": category,
                "Revenue": revenue,
                "UnitsSold": units_sold,
                "UnitPrice": unit_price,
                "DiscountApplied": discount_applied,
                "HolidayFlag": holiday_flag,
            }
        ]
    )
    return pd.concat([history_source, next_row], ignore_index=True)


def metrics_from_metadata() -> dict[str, Any]:
    return (
        state.metadata.get("metrics")
        or state.metadata.get("model_metrics")
        or state.metadata.get("final_metrics")
        or {}
    )


class BaseRetailRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    date: str
    store_id: int = Field(..., ge=1)
    category: str
    unit_price: float = Field(..., gt=0)
    units_sold: int = Field(..., ge=0)
    holiday_flag: int

    @field_validator("category")
    @classmethod
    def category_must_exist(cls, value: str) -> str:
        normalized = value.strip()
        if state.available_categories and normalized not in state.available_categories:
            raise ValueError(f"category must be one of {sorted(state.available_categories)}")
        return normalized

    @field_validator("holiday_flag")
    @classmethod
    def holiday_flag_must_be_binary(cls, value: int) -> int:
        if value not in {0, 1}:
            raise ValueError("holiday_flag must be 0 or 1")
        return value


class PredictionRequest(BaseRetailRequest):
    discount_applied: float = Field(..., ge=0, le=100)


class CategoryComparisonRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    date: str
    store_id: int = Field(..., ge=1)
    unit_price: float = Field(..., gt=0)
    units_sold: int = Field(..., ge=0)
    discount_applied: float = Field(..., ge=0, le=100)
    holiday_flag: int

    @field_validator("holiday_flag")
    @classmethod
    def holiday_flag_must_be_binary(cls, value: int) -> int:
        if value not in {0, 1}:
            raise ValueError("holiday_flag must be 0 or 1")
        return value


class DiscountOptimizerRequest(BaseRetailRequest):
    pass


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Retail Forecasting API",
        "version": app.version,
        "ready": state.app_ready,
        "endpoints": [
            "/",
            "/health",
            "/model-info",
            "/history",
            "/predict",
            "/forecast",
            "/category-comparison",
            "/discount-optimizer",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready" if state.app_ready else "degraded",
        "model_ready": state.model_ready,
        "history_ready": state.history_ready,
        "artifacts": {
            "model_file": MODEL_FILE.name,
            "pipeline_file": PIPELINE_FILE.name,
            "training_columns_file": TRAINING_COLUMNS_FILE.name,
            "metadata_file": METADATA_FILE.name,
        },
        "history_file": HISTORY_FILE.name,
        "history_rows": int(len(state.history_df)) if state.history_ready else 0,
        "available_categories": sorted(state.available_categories),
        "artifact_errors": state.artifact_errors,
        "history_error": state.history_error,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    ensure_model_ready()
    return {
        "model_name": state.metadata.get("model_name", "Tuned XGBoost Regressor"),
        "target": state.metadata.get("target", "Revenue"),
        "feature_count": len(state.training_columns),
        "metrics": metrics_from_metadata(),
        "metadata": state.metadata,
    }


@app.get("/history")
def history(
    store_id: int = Query(..., ge=1),
    category: str = Query(...),
    days: Optional[int] = Query(default=None, ge=1, le=3650),
    before_date: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    ensure_history_ready()
    category = validate_category(category)

    filtered_history = state.history_df[
        (state.history_df["StoreID"] == store_id) & (state.history_df["Category"] == category)
    ].copy()

    if before_date:
        cutoff = parse_date_input(before_date, field_name="before_date")
        filtered_history = filtered_history[filtered_history["Date"] < cutoff]

    filtered_history = filtered_history.sort_values("Date")
    if days is not None:
        filtered_history = filtered_history.tail(days)

    records = [
        {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "store_id": int(row["StoreID"]),
            "category": row["Category"],
            "revenue": round(float(row["Revenue"]), 2),
            "units_sold": int(row["UnitsSold"]),
            "unit_price": round(float(row["UnitPrice"]), 2),
            "discount_applied": round(float(row["DiscountApplied"]), 2),
            "holiday_flag": int(row["HolidayFlag"]),
        }
        for _, row in filtered_history.iterrows()
    ]

    return {
        "store_id": store_id,
        "category": category,
        "count": len(records),
        "records": records,
    }


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    ensure_app_ready()
    target_date = parse_date_input(payload.date)

    try:
        prediction, feature_snapshot = predict_revenue(
            history_source=state.history_df,
            target_date=target_date,
            store_id=payload.store_id,
            category=payload.category,
            unit_price=payload.unit_price,
            units_sold=payload.units_sold,
            discount_applied=payload.discount_applied,
            holiday_flag=payload.holiday_flag,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail={"message": "Prediction failed", "error": str(exc)},
        ) from exc

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "store_id": payload.store_id,
        "category": payload.category,
        "inputs": {
            "unit_price": payload.unit_price,
            "units_sold": payload.units_sold,
            "discount_applied": payload.discount_applied,
            "holiday_flag": payload.holiday_flag,
        },
        "predicted_revenue": round(prediction, 2),
        "feature_snapshot": feature_snapshot,
    }


@app.get("/forecast")
def forecast(
    days: int = Query(default=7, ge=1, le=90),
    from_date: Optional[str] = Query(default=None),
    store_id: int = Query(..., ge=1),
    category: str = Query(...),
    unit_price: float = Query(..., gt=0),
    units_sold: int = Query(..., ge=0),
    discount_applied: float = Query(..., ge=0, le=100),
    holiday_flag: int = Query(...),
) -> dict[str, Any]:
    ensure_app_ready()
    category = validate_category(category)
    if holiday_flag not in {0, 1}:
        raise HTTPException(status_code=400, detail={"message": "holiday_flag must be 0 or 1"})

    start_date = parse_date_input(from_date, field_name="from_date") if from_date else pd.Timestamp.today().normalize()
    temporary_history = state.history_df.copy()
    forecast_rows: list[dict[str, Any]] = []

    try:
        for day_offset in range(days):
            target_date = start_date + pd.Timedelta(days=day_offset)
            prediction, feature_snapshot = predict_revenue(
                history_source=temporary_history,
                target_date=target_date,
                store_id=store_id,
                category=category,
                unit_price=unit_price,
                units_sold=units_sold,
                discount_applied=discount_applied,
                holiday_flag=holiday_flag,
            )

            forecast_rows.append(
                {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "day_of_week": target_date.day_name(),
                    "predicted_revenue": round(prediction, 2),
                    "feature_snapshot": feature_snapshot,
                }
            )

            temporary_history = append_prediction_to_history(
                history_source=temporary_history,
                target_date=target_date,
                store_id=store_id,
                category=category,
                revenue=prediction,
                unit_price=unit_price,
                units_sold=units_sold,
                discount_applied=discount_applied,
                holiday_flag=holiday_flag,
            )
    except Exception as exc:
        logger.exception("Forecast generation failed")
        raise HTTPException(
            status_code=500,
            detail={"message": "Forecast generation failed", "error": str(exc)},
        ) from exc

    total_predicted_revenue = round(sum(item["predicted_revenue"] for item in forecast_rows), 2)
    average_predicted_revenue = round(total_predicted_revenue / days, 2)

    return {
        "store_id": store_id,
        "category": category,
        "from_date": start_date.strftime("%Y-%m-%d"),
        "days": days,
        "inputs": {
            "unit_price": unit_price,
            "units_sold": units_sold,
            "discount_applied": discount_applied,
            "holiday_flag": holiday_flag,
        },
        "forecast": forecast_rows,
        "summary": {
            "total_predicted_revenue": total_predicted_revenue,
            "average_daily_revenue": average_predicted_revenue,
        },
    }


@app.post("/category-comparison")
def category_comparison(payload: CategoryComparisonRequest) -> dict[str, Any]:
    ensure_app_ready()
    target_date = parse_date_input(payload.date)
    comparison_rows: list[dict[str, Any]] = []

    try:
        for category in sorted(state.available_categories):
            prediction, _ = predict_revenue(
                history_source=state.history_df,
                target_date=target_date,
                store_id=payload.store_id,
                category=category,
                unit_price=payload.unit_price,
                units_sold=payload.units_sold,
                discount_applied=payload.discount_applied,
                holiday_flag=payload.holiday_flag,
            )
            comparison_rows.append(
                {
                    "category": category,
                    "predicted_revenue": round(prediction, 2),
                }
            )
    except Exception as exc:
        logger.exception("Category comparison failed")
        raise HTTPException(
            status_code=500,
            detail={"message": "Category comparison failed", "error": str(exc)},
        ) from exc

    comparison_rows.sort(key=lambda item: item["predicted_revenue"], reverse=True)

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "store_id": payload.store_id,
        "inputs": {
            "unit_price": payload.unit_price,
            "units_sold": payload.units_sold,
            "discount_applied": payload.discount_applied,
            "holiday_flag": payload.holiday_flag,
        },
        "comparisons": comparison_rows,
        "best_option": comparison_rows[0] if comparison_rows else None,
    }


@app.post("/discount-optimizer")
def discount_optimizer(payload: DiscountOptimizerRequest) -> dict[str, Any]:
    ensure_app_ready()
    target_date = parse_date_input(payload.date)
    discount_candidates = [0, 5, 10, 15, 20]
    scenarios: list[dict[str, Any]] = []

    try:
        for discount in discount_candidates:
            prediction, _ = predict_revenue(
                history_source=state.history_df,
                target_date=target_date,
                store_id=payload.store_id,
                category=payload.category,
                unit_price=payload.unit_price,
                units_sold=payload.units_sold,
                discount_applied=float(discount),
                holiday_flag=payload.holiday_flag,
            )
            scenarios.append(
                {
                    "discount_applied": discount,
                    "predicted_revenue": round(prediction, 2),
                }
            )
    except Exception as exc:
        logger.exception("Discount optimization failed")
        raise HTTPException(
            status_code=500,
            detail={"message": "Discount optimization failed", "error": str(exc)},
        ) from exc

    scenarios.sort(key=lambda item: item["predicted_revenue"], reverse=True)
    best_discount = scenarios[0] if scenarios else None

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "store_id": payload.store_id,
        "category": payload.category,
        "inputs": {
            "unit_price": payload.unit_price,
            "units_sold": payload.units_sold,
            "holiday_flag": payload.holiday_flag,
        },
        "discount_scenarios": scenarios,
        "best_discount": best_discount,
    }
