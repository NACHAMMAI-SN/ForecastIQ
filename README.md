
#  ForecastIQ — Intelligent Time Series Forecasting Platform

ForecastIQ is an end-to-end AI-powered time series forecasting system that combines **meta-learning based model selection** with **AutoTS forecasting** to automatically choose and apply the best model for a given dataset.

The platform provides both:

*  REST API endpoints
*  Interactive web UI with CSV upload and Plotly visualizations

---
deployment link:
https://forecastiq-1kin.onrender.com

##  Key Features

* Intelligent model selection using a trained meta-classifier
* Automated forecasting using AutoTS
* Anomaly detection on validation forecasts
* CSV upload support (single or multiple files)
* Interactive Plotly charts
* Deployable with Docker or Render
* FastAPI-based production-ready backend

---

##  Architecture Overview

```
User Data (CSV / API)
        ↓
Feature Extraction
        ↓
Meta-Classifier (Select Best Model)
        ↓
AutoTS (Restricted to Selected Model)
        ↓
Forecast + MAPE + Anomaly Detection
        ↓
REST API / Web UI Visualization
```

---

##  Tech Stack

**Backend**

* FastAPI
* Uvicorn
* Python

**Machine Learning**

* AutoTS
* Scikit-learn
* Pandas
* NumPy
* Statsmodels

**Visualization**

* Plotly
* Jinja2 Templates

**Deployment**

* Render
* Docker (optional)

---

##  Project Structure

```
autots_intelligent/
│
├── api.py                     # FastAPI application
├── intelligent_wrapper.py     # Meta-learning + AutoTS logic
├── anomaly_detection.py       # Anomaly detection module
├── feature_extraction.py      # Feature engineering
├── meta_classifier.py         # Meta-model training
├── training_pipeline.py       # Training workflow
├── meta_model.pkl             # Trained meta-classifier
├── label_encoder.pkl          # Label encoder
├── templates/                 # HTML UI templates
├── requirements.txt
└── Dockerfile
```

---

##  API Endpoints

###  Health Check

```
GET /health
```

Returns:

```json
{
  "status": "ok"
}
```

---

###  Forecast Prediction (API)

```
POST /predict
```

Request Body:

```json
{
  "dates": ["2024-01-01", "2024-01-02", "..."],
  "values": [100, 102, ...]
}
```

Response:

```json
{
  "selected_model": "ARIMA",
  "mape": 0.12,
  "forecast": [...],
  "anomalies": [...]
}
```

---

###  Web UI

```
GET /
```

Upload CSV files and generate:

* Forecast charts
* Model details
* Validation metrics
* Anomaly markers

Minimum required: **120 data points**

---

##  Installation (Local)

```bash
git clone https://github.com/NACHAMMAI-SN/ForecastIQ.git
cd ForecastIQ/autots_intelligent

pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open:

```
http://localhost:8000
```

---

##  Deployment on Render

**Build Command**

```
pip install -r requirements.txt
```

**Start Command**

```
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Ensure these files are committed:

* `meta_model.pkl`
* `label_encoder.pkl`

---

##  Training the Meta-Classifier

If you need to retrain:

```bash
python training_pipeline.py
python meta_classifier.py
```

This regenerates:

* `meta_model.pkl`
* `label_encoder.pkl`

---

##  Anomaly Detection

In validation mode:

* Last 30 points are treated as holdout
* Forecast compared to actual
* Significant deviations flagged as anomalies

---

##  Why ForecastIQ is Unique

Unlike traditional forecasting tools, ForecastIQ:

* Learns which model performs best for specific time series characteristics
* Uses meta-learning to reduce trial-and-error
* Restricts AutoTS search space intelligently
* Provides production-ready API + UI

---

##  Future Improvements

* Model caching
* Background task execution
* Distributed training
* Model registry integration
* CI/CD automation

---

