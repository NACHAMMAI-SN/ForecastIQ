I have cloned the AutoTS repository.

Project root:

D:\autots

Inside it:

autots/ → original AutoTS library (DO NOT MODIFY)

autots_intelligent/ → my extension layer (ALL new code goes here)

You must NOT modify anything inside:
autots/

All new code must be written ONLY inside:
autots_intelligent/

Goal:
Build an intelligent time-series forecasting system that learns which forecasting model to use before running AutoTS.

Datasets are already available inside:

autots_intelligent/data/stock_data/
autots_intelligent/data/synthetic_data/

Each dataset follows format:
date,value

IMPLEMENTATION PLAN

Implement step by step. Do NOT generate everything at once.

STEP 1 — feature_extraction.py

Create function:

extract_features(series: pd.Series) -> dict

Extract the following features:

mean

variance

skewness

kurtosis

rolling mean (window=7, last value)

rolling std (window=7, last value)

autocorrelation (lag 1, 7, 12)

Augmented Dickey-Fuller p-value

linear regression slope (trend)

simple seasonality strength (variance of seasonal component using simple rolling difference)

Return features as dictionary.

Use:

pandas

numpy

statsmodels

Write clean, reusable code.

STEP 2 — training_pipeline.py

For all CSV files inside:

data/stock_data/

data/synthetic_data/

Do the following:

Load CSV

Convert date column to datetime

Set date as index

Extract features using extract_features()

Run AutoTS:

forecast_length=30

frequency="infer"

ensemble=None

Get best model name

Build training dataset:

| feature_1 | feature_2 | ... | best_model |

Save as:
training_features.csv

Add error handling so one failed dataset does not stop execution.

STEP 3 — meta_classifier.py

Load training_features.csv

Separate X and y

Encode best_model using LabelEncoder

Train RandomForestClassifier

Save:

meta_model.pkl

label_encoder.pkl

Add basic model accuracy print.

STEP 4 — intelligent_wrapper.py

Create function:

predict_with_intelligence(df: pd.DataFrame)

Steps:

Extract features

Load meta_model.pkl

Predict best_model

Run AutoTS restricted to predicted model

Generate forecast

Compute MAPE

Return:
{
"selected_model": "...",
"mape": ...,
"forecast": [...]
}

Do NOT retrain classifier here.

STEP 5 — anomaly_detection.py

Create function:

detect_anomalies(actual, predicted)

Mark anomaly if:
abs(actual - predicted) > 2 * rolling_std

Return list of anomaly flags.

STEP 6 — api.py

Create FastAPI app.

POST /predict

Input:
{
"dates": [...],
"values": [...]
}

Steps:

Convert input to DataFrame

Call predict_with_intelligence()

Run anomaly detection

Return JSON:

{
"selected_model": "...",
"mape": ...,
"forecast": [...],
"anomalies": [...]
}

Use uvicorn to run.

RULES

Do NOT modify autots/ library

Keep modular architecture

Write production-ready clean code

Add comments

Add error handling

Assume CSV format: date,value

Keep file paths relative to autots_intelligent/
