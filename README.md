# Scania Trucks APS Failure Prediction — Industrial AI Predictive Maintenance

A modern, high-performance predictive maintenance web application built on top of the **Scania CV AB Air Pressure System (APS) Failure** dataset. This application integrates an end-to-end trained Scikit-Learn machine learning pipeline to detect imminent pneumatic component failures across commercial truck fleets.

---

## 🚛 Project Overview

The **Air Pressure System (APS)** generates pressurized air for vital heavy truck functions, including:
- Service and emergency braking circuits
- Pneumatic gear-shifting actuators
- Adaptive chassis suspension leveling bellows

### Industrial Cost-Metric (IDA 2016 Benchmark):
- **Cost 1 (Type 1 error / False Positive):** **$10** (Unnecessary diagnostic check at workshop)
- **Cost 2 (Type 2 error / False Negative):** **$500** (Missing a failure causes roadside breakdown, towing & logistical penalties)
- **Net Optimization:** Timely preventive maintenance catches **75.0%** of failures, reducing overall maintenance costs by **74.28%** ($141,500 net savings on the test fleet).

---

## 🌟 Key Application Features

1. **Industrial Fleet Dashboard:**
   - Automotive dark navy & Scania Griffin Red design system.
   - Real-time model readiness status indicator with live pulse.
   - Comprehensive interactive schematic of the 4 APS functional loops.

2. **Single Vehicle Telemetry Diagnostic:**
   - Form for entering the **20 most predictive sensor features** (`bj_000`, `aq_000`, `ci_000`, `dn_000`, `aa_000`, etc.).
   - Pre-loaded test fleet profiles:
     - *Healthy Fleet Truck* (Low wear, optimal pressures)
     - *Imminent APS Failure* (High probability component failure)
     - *Elevated Wear & Tear* (Moderate pressure loss)
     - *Partial Telemetry* (Demonstrates median imputer robustness with missing values)
   - Real-time animated probability gauge meter and tailored operational recommendations.

3. **Batch CSV Telemetry:**
   - Drag-and-drop CSV upload supporting multi-vehicle datasets.
   - Ingests raw 171-sensor Scania telemetry or pre-filtered 20-feature datasets.
   - Instant fleet KPIs: Total Units, Failure Count, Healthy Units, and Estimated Breakdown Savings.
   - Interactive table with search, status filters, and CSV export.

4. **Model Analytics & Evaluation KPIs:**
   - **Accuracy:** 98.04% (on official 16,000 Scania test cohort)
   - **F1-Score:** 0.6416
   - **Recall (Sensitivity):** 75.00%
   - **Precision:** 56.00%
   - **ROC-AUC:** 0.9634
   - Interactive Confusion Matrix with financial cost breakdown ($10 FP vs $500 FN).
   - Multi-model comparison benchmark (Tuned Random Forest vs Voting Ensemble vs Logistic Regression vs Decision Tree).
   - Top 10 feature importance ranking.

5. **Session History:**
   - Persists all diagnostics run in the active session for auditing and comparison.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.11, [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Pydantic](https://docs.pydantic.dev/)
- **Machine Learning:** [Scikit-Learn 1.6.1](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Frontend:** Pure HTML5, Vanilla CSS3 (Custom Scania Dark Navy & Red Theme), Modern ES6 JavaScript (No build tools or Node.js required)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11 installed on your system.

### 2. Install Dependencies
Run the following command in PowerShell / Terminal:
```bash
py -3.11 -m pip install -r requirements.txt
```
*(Or install packages directly)*:
```bash
py -3.11 -m pip install joblib pandas scikit-learn==1.6.1 fastapi uvicorn python-multipart httpx
```

### 3. Run the Application
To start the live web application:
```bash
py -3.11 run_app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:8050
```
- **Interactive UI Dashboard:** `http://127.0.0.1:8050`
- **Swagger / OpenAPI Documentation:** `http://127.0.0.1:8050/docs`

---

## 📁 Repository Structure

```text
├── Scania Trucks APS Failure.ipynb               # Project exploratory analysis & model training notebook
├── aps_failure_description.txt                   # Official Scania CV AB dataset documentation
├── aps_failure_test_set.csv                      # Official 16,000 vehicle test dataset
├── aps_failure_training_set.csv                  # Official 60,000 vehicle training dataset
├── run_app.py                                    # One-click application launcher
├── requirements.txt                              # Python package dependencies
├── webapp/
│   ├── backend/
│   │   ├── server.py                             # FastAPI server & REST API endpoints
│   │   ├── model_service.py                      # Preprocessing & ML inference engine
│   │   └── prepare_data.py                       # Preset generator & sample exporter
│   └── static/
│       ├── index.html                            # Dashboard UI interface
│       ├── css/
│       │   └── style.css                         # Automotive dark navy styling
│       ├── js/
│       │   └── app.js                            # Client application logic
│       └── data/
│           ├── presets.json                      # Pre-configured truck telemetry profiles
│           └── batch_sample_scania_trucks.csv    # Ready-to-test 15-truck fleet batch
└── ml final 2/
    └── New folder/
        └── scania_aps_pipeline.joblib            # Trained Scikit-Learn Random Forest Pipeline
```

---

## 👥 Authors & Academic Presentation
Developed for the **Scania Trucks APS Failure Prediction** machine learning project presentation.
Based on the IDA 2016 Industrial Challenge Dataset by **Scania CV AB**, Stockholm, Sweden.
