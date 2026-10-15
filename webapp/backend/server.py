import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure local imports work properly
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from model_service import model_service, TOP_FEATURES, SENSOR_METADATA

app = FastAPI(
    title="Scania Trucks APS Predictive Maintenance AI",
    description="Production-grade AI dashboard predicting Air Pressure System failures for Scania heavy transport fleets.",
    version="2.0.0"
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SinglePredictionRequest(BaseModel):
    features: Dict[str, Any]

@app.get("/api/health")
def health_check():
    """Health check endpoint reporting model status and top features."""
    return {
        "status": "online",
        "model_loaded": model_service.is_ready(),
        "model_path": str(model_service.model_path),
        "features_count": len(TOP_FEATURES),
        "top_features": TOP_FEATURES
    }

@app.post("/api/predict")
def predict_single(data: SinglePredictionRequest):
    """Predict failure status and probability for a single truck telemetry input."""
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Machine learning pipeline is not yet loaded.")
    
    try:
        result = model_service.predict_single(data.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """Upload a CSV dataset and receive full batch predictions, metrics, and cost savings."""
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Machine learning pipeline is not yet loaded.")
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are supported.")
    
    try:
        contents = await file.read()
        results = model_service.predict_batch_csv(contents)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")

@app.get("/api/metrics")
def get_model_metrics():
    """
    Return comprehensive model evaluation metrics, confusion matrix, 
    IDA Challenge cost calculations, and feature importances extracted from the project notebook.
    """
    return {
        "official_test_set": {
            "name": "Official Scania Test Dataset (16,000 Trucks)",
            "accuracy": 0.9804,
            "f1_score": 0.6416,
            "precision": 0.5600,
            "recall": 0.7500,
            "total_test_samples": 16000,
            "positive_failures": 375,
            "healthy_negatives": 15625
        },
        "confusion_matrix": {
            "true_negative": 15403,
            "false_positive": 222,
            "false_negative": 92,
            "true_positive": 283,
            "cost_fp_unnecessary_check": 10,
            "cost_fn_roadside_breakdown": 500,
            "total_model_cost": (222 * 10) + (92 * 500), # $48,220
            "baseline_unchecked_cost": 375 * 500, # $187,500
            "cost_reduction_percentage": round((1 - (48220 / 187500)) * 100, 2) # 74.28%
        },
        "model_comparison": [
            {
                "model": "Tuned Random Forest (Selected)",
                "accuracy": 0.9813,
                "f1_score": 0.5867,
                "official_test_f1": 0.6416,
                "recall": 0.7500,
                "precision": 0.5600,
                "status": "Production Pipeline"
            },
            {
                "model": "Base Random Forest",
                "accuracy": 0.9888,
                "f1_score": 0.5872,
                "official_test_f1": 0.5910,
                "recall": 0.4800,
                "precision": 0.7600,
                "status": "Evaluated"
            },
            {
                "model": "Soft Voting Ensemble (LR + DT + RF)",
                "accuracy": 0.9707,
                "f1_score": 0.5125,
                "official_test_f1": 0.5340,
                "recall": 0.6200,
                "precision": 0.4400,
                "status": "Evaluated"
            },
            {
                "model": "Balanced Logistic Regression",
                "accuracy": 0.9640,
                "f1_score": 0.4613,
                "official_test_f1": 0.4820,
                "recall": 0.9250,
                "precision": 0.3073,
                "status": "Evaluated"
            },
            {
                "model": "Balanced Decision Tree",
                "accuracy": 0.9577,
                "f1_score": 0.4240,
                "official_test_f1": 0.4310,
                "recall": 0.9400,
                "precision": 0.2700,
                "status": "Evaluated"
            }
        ],
        "feature_importances": [
            {"feature": k, "name": v["name"], "importance": v["importance"], "category": v["category"], "unit": v["unit"]}
            for k, v in SENSOR_METADATA.items()
        ],
        "roc_curve": {
            "auc": 0.9634,
            "fpr": [0.0, 0.005, 0.014, 0.025, 0.045, 0.08, 0.15, 0.30, 0.60, 1.0],
            "tpr": [0.0, 0.450, 0.680, 0.750, 0.830, 0.91, 0.95, 0.98, 0.99, 1.0]
        }
    }

# Mount static files directory
def _locate_static_dir() -> Path:
    candidates = [
        BASE_DIR.parent / "static",
        BASE_DIR / "static",
        Path("webapp/static"),
        Path("static")
    ]
    for c in candidates:
        if c.exists() and (c / "index.html").exists():
            return c
    return candidates[0]

STATIC_DIR = _locate_static_dir()
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_home():
    """Serve the single page application HTML."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(content={"message": "Scania APS AI API is active. Frontend index.html not yet placed."})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
