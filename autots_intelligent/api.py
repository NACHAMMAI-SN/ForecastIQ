"""
FastAPI app for time-series forecasting.
POST /predict accepts dates and values, returns selected model, MAPE, forecast, and anomalies.
GET / and POST /predict-ui provide a web UI for CSV upload and Plotly charts.
"""

import io
import os
import sys
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTOTS_PKG_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _AUTOTS_PKG_ROOT not in sys.path:
    sys.path.insert(0, _AUTOTS_PKG_ROOT)

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from intelligent_wrapper import predict_with_intelligence, FORECAST_LENGTH
from anomaly_detection import detect_anomalies

app = FastAPI(title="Time Series Forecast API", version="1.0.0")

# Jinja2 templates for UI (templates/ relative to this file)
TEMPLATES_DIR = os.path.join(_SCRIPT_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class PredictRequest(BaseModel):
    dates: list[str]
    values: list[float]


class PredictResponse(BaseModel):
    selected_model: str
    mape: float | None
    forecast: list[float]
    anomalies: list[bool]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Run intelligent forecasting: convert input to DataFrame, call predict_with_intelligence,
    run anomaly detection when we have a holdout, return selected model, MAPE, forecast, anomalies.
    """
    if len(request.dates) != len(request.values):
        raise HTTPException(
            status_code=400,
            detail="dates and values must have the same length",
        )
    if len(request.dates) < 10:
        raise HTTPException(
            status_code=400,
            detail="At least 10 observations required",
        )

    try:
        df = pd.DataFrame({"date": request.dates, "value": request.values})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df = df.set_index("date").sort_index()
        df = df[["value"]]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid dates/values: {e}") from e

    if len(df) < 120:
        raise HTTPException(
            status_code=400,
            detail="Time series too short. Minimum 120 points required.",
        )

    try:
        # If we have enough data, use last FORECAST_LENGTH as holdout for anomaly detection
        if len(df) > FORECAST_LENGTH:
            train_df = df.iloc[:-FORECAST_LENGTH]
            holdout = df.iloc[-FORECAST_LENGTH:]
            result = predict_with_intelligence(train_df)
            actual = holdout["value"].tolist()
            predicted = result["forecast"]
            if len(actual) == len(predicted):
                anomalies = detect_anomalies(actual, predicted)
            else:
                anomalies = [False] * len(result["forecast"])
        else:
            result = predict_with_intelligence(df)
            anomalies = [False] * len(result["forecast"])
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run training_pipeline.py and meta_classifier.py first.",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return PredictResponse(
        selected_model=result["selected_model"],
        mape=result["mape"],
        forecast=result["forecast"],
        anomalies=anomalies,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# --- UI routes ---

MIN_POINTS_UI = 120
MODEL_SPEED_DEFAULT = "superfast"


def _run_autots_ui(df: pd.DataFrame, model_speed: str) -> dict:
    """Run AutoTS with user-chosen model_list (superfast/fast/all). Returns selected_model, mape, forecast."""
    from autots import AutoTS
    model = AutoTS(
        forecast_length=FORECAST_LENGTH,
        frequency="infer",
        ensemble=None,
        model_list=model_speed,
        max_generations=1,
        num_validations=0,
        verbose=0,
    )
    model.fit(df)
    result = model.predict(forecast_length=FORECAST_LENGTH, just_point_forecast=True)
    if isinstance(result, pd.DataFrame):
        forecast = result.iloc[:, 0].tolist()
    else:
        forecast = result.tolist() if hasattr(result, "tolist") else list(result)
    mape = None
    try:
        res = model.results()
        if res is not None and not isinstance(res, str) and hasattr(res, "columns") and "Score" in res.columns:
            score = res.iloc[0].get("Score")
            if score is not None and isinstance(score, (int, float)):
                mape = float(score)
    except Exception:
        pass
    return {"selected_model": model.best_model_name, "mape": mape, "forecast": forecast}


def _parse_csv_upload(content: bytes) -> pd.DataFrame:
    """Parse CSV from uploaded file content. Expects date,value format (first two columns)."""
    df = pd.read_csv(io.BytesIO(content))
    if df.shape[1] < 2 or df.shape[0] == 0:
        raise ValueError("CSV must have at least 2 columns and 1 row")
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    df = df.set_index("date").sort_index()
    return df[["value"]]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page with file upload form."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "results_list": None,
            "output_mode": "charts",
            "prediction_mode": "validation",
            "error": None,
            "file_errors": [],
        },
    )


@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(
    request: Request,
    files: list[UploadFile] = File(...),
    model_speed: str = Form(MODEL_SPEED_DEFAULT),
    output_mode: str = Form("charts"),
    prediction_mode: str = Form("validation"),
):
    """
    Accept single/multiple CSV upload or folder, run prediction with chosen model_speed and prediction_mode.
    Returns table or charts per file. Validates 120 points per file; skips failures.
    """
    if not files:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "results_list": None,
                "output_mode": output_mode,
                "prediction_mode": prediction_mode,
                "error": "Please upload one or more CSV files.",
                "file_errors": [],
            },
        )
    # Normalize model_speed
    if model_speed not in ("superfast", "fast", "all"):
        model_speed = MODEL_SPEED_DEFAULT
    # Normalize prediction_mode
    if prediction_mode not in ("validation", "future"):
        prediction_mode = "validation"
    # Filter to CSV only
    csv_files = [f for f in files if f.filename and f.filename.lower().endswith(".csv")]
    if not csv_files:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "results_list": None,
                "output_mode": output_mode,
                "prediction_mode": prediction_mode,
                "error": "No CSV files found. Upload .csv files only.",
                "file_errors": [],
            },
        )
    results_list = []
    file_errors = []
    for upload in csv_files:
        filename = upload.filename or "unknown.csv"
        try:
            content = await upload.read()
            df = _parse_csv_upload(content)
        except Exception as e:
            file_errors.append(f"{filename}: Invalid CSV — {e}")
            continue
        if len(df) < MIN_POINTS_UI:
            file_errors.append(f"{filename}: Too short (min {MIN_POINTS_UI} points).")
            continue
        # Validation vs future prediction modes
        try:
            if prediction_mode == "future":
                # Train on full dataset; predict next 30 points; no MAPE, no anomaly detection
                pred_result = _run_autots_ui(df, model_speed)
                pred_result["mape"] = None
                chart_html = _build_chart_html(
                    df,
                    df,
                    None,
                    pred_result["forecast"],
                    [False] * len(pred_result["forecast"]),
                )
            else:
                # Validation mode: split last 30 points as holdout, train on remaining
                if len(df) <= FORECAST_LENGTH:
                    file_errors.append(f"{filename}: Not enough points for validation split.")
                    continue
                train_df = df.iloc[:-FORECAST_LENGTH]
                holdout = df.iloc[-FORECAST_LENGTH:]
                pred_result = _run_autots_ui(train_df, model_speed)
                actual_vals = holdout["value"].tolist()
                forecast_vals = pred_result["forecast"]
                anomalies = (
                    detect_anomalies(actual_vals, forecast_vals)
                    if len(actual_vals) == len(forecast_vals)
                    else [False] * len(pred_result["forecast"])
                )
                chart_html = _build_chart_html(
                    df, train_df, holdout, pred_result["forecast"], anomalies
                )
        except Exception as e:
            file_errors.append(f"{filename}: {e}")
            continue
        results_list.append({
            "filename": filename,
            "selected_model": pred_result["selected_model"],
            "mape": pred_result["mape"],
            "chart_html": chart_html,
        })
    if not results_list:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "results_list": None,
                "output_mode": output_mode,
                "prediction_mode": prediction_mode,
                "error": "No file could be processed.",
                "file_errors": file_errors,
            },
        )
    # Single file: legacy result shape + chart
    if len(results_list) == 1:
        r = results_list[0]
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": {"selected_model": r["selected_model"], "mape": r["mape"], "chart_html": r["chart_html"]},
                "results_list": None,
                "output_mode": output_mode,
                "prediction_mode": prediction_mode,
                "error": None,
                "file_errors": file_errors,
            },
        )
    # Multiple files: pass results_list and output_mode (table | charts)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "results_list": results_list,
            "output_mode": output_mode,
            "prediction_mode": prediction_mode,
            "error": None,
            "file_errors": file_errors,
        },
    )


def _build_chart_html(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    holdout: pd.DataFrame | None,
    forecast: list[float],
    anomalies: list[bool],
) -> str:
    """Build Plotly line chart (actual + forecast, with anomaly highlights) and return HTML."""
    import plotly.graph_objects as go

    # Actual series
    actual_dates = full_df.index.tolist()
    actual_vals = full_df["value"].tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual_dates,
            y=actual_vals,
            mode="lines",
            name="Actual",
            line=dict(color="rgb(31, 119, 180)", width=2),
        )
    )
    # Forecast: align with holdout dates if available, else extend from last date
    if holdout is not None and len(holdout) == len(forecast):
        forecast_dates = holdout.index.tolist()
    else:
        last_ts = full_df.index[-1]
        freq = pd.infer_freq(full_df.index) or "D"
        forecast_dates = pd.date_range(
            start=last_ts, periods=len(forecast) + 1, freq=freq
        )[1:].tolist()
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(color="rgb(255, 127, 14)", width=2, dash="dash"),
        )
    )
    # Anomaly points (where actual vs forecast was anomalous)
    if holdout is not None and len(anomalies) == len(forecast_dates):
        anom_dates = [d for d, a in zip(forecast_dates, anomalies) if a]
        anom_vals = [v for v, a in zip(forecast, anomalies) if a]
        if anom_dates and anom_vals:
            fig.add_trace(
                go.Scatter(
                    x=anom_dates,
                    y=anom_vals,
                    mode="markers",
                    name="Anomaly",
                    marker=dict(symbol="x", size=12, color="red", line=dict(width=2)),
                )
            )
    fig.update_layout(
        title="Time Series: Actual vs Forecast",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        height=450,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
