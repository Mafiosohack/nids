"""
NIDS REST API
FastAPI-based REST API for the intrusion detection system.

Endpoints:
  GET  /          - Welcome message
  GET  /health    - Health check
  POST /detect    - Run detection on traffic features
  GET  /alerts    - Get recent alerts
  GET  /stats     - System statistics
  POST /model/reload - Reload detection model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
from loguru import logger
from pathlib import Path


# ─── Pydantic Schemas ────────────────────────────────────────────

class DetectionRequest(BaseModel):
    """Request schema for detection endpoint."""
    features: List[List[float]] = Field(
        ...,
        description="List of feature vectors to classify"
    )
    return_scores: bool = Field(
        default=False,
        description="Whether to return anomaly scores"
    )


class DetectionResponse(BaseModel):
    """Response schema for detection endpoint."""
    predictions: List[str]
    anomaly_scores: Optional[List[float]] = None
    processing_time_ms: float
    total_samples: int
    anomalies_found: int


class SystemStats(BaseModel):
    """System statistics schema."""
    total_traffic_analyzed: int
    total_anomalies_detected: int
    detection_rate: float
    uptime_seconds: float
    model_loaded: bool
    model_type: str


# ─── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(
    title="🔐 NIDS API",
    description=(
        "Network Intrusion Detection System — "
        "Real-time anomaly detection powered by Machine Learning"
    ),
    version="1.0.0"
)

# CORS middleware (allows frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ────────────────────────────────────────────────

detector = None
preprocessor = None
alerts_store: List[Dict] = []
system_stats = {
    'total_analyzed': 0,
    'total_anomalies': 0,
    'start_time': datetime.now(),
    'model_type': 'none'
}


# ─── Startup ─────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load models when API starts."""
    global detector, preprocessor, system_stats

    logger.info("Starting NIDS API...")

    # Try to load Isolation Forest
    model_path = Path("models/isolation_forest.pkl")
    preprocessor_path = Path("models/preprocessor.pkl")

    if model_path.exists() and preprocessor_path.exists():
        try:
            from models.isolation_forest import IsolationForestDetector
            from feature_engineering.preprocessor import FeaturePreprocessor

            detector = IsolationForestDetector.load(str(model_path))
            preprocessor = FeaturePreprocessor.load(str(preprocessor_path))
            system_stats['model_type'] = 'isolation_forest'

            logger.success("Models loaded successfully!")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            logger.info("API running without pre-loaded models")
    else:
        logger.warning(
            "No trained models found. "
            "Run: python main.py train --dataset nsl-kdd "
            "--model isolation_forest --binary"
        )


# ─── Endpoints ───────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint."""
    return {
        "message": "🔐 NIDS API is running!",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_type": system_stats['model_type']
    }


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_intrusions(request: DetectionRequest):
    """
    Detect intrusions in network traffic.

    Send feature vectors and get back predictions of
    'normal' or 'anomaly' for each sample.
    """
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model loaded. "
                "Train a model first: python main.py train "
                "--dataset nsl-kdd --model isolation_forest --binary"
            )
        )

    start_time = datetime.now()

    # Convert to numpy array
    features = np.array(request.features)

    # Detect anomalies
    raw_predictions = detector.predict(features)
    predictions = [
        'anomaly' if p == -1 else 'normal'
        for p in raw_predictions
    ]

    # Get anomaly scores if requested
    anomaly_scores = None
    if request.return_scores:
        anomaly_scores = detector.predict_proba(features).tolist()

    # Count anomalies
    num_anomalies = sum(1 for p in predictions if p == 'anomaly')

    # Update system stats
    system_stats['total_analyzed'] += len(features)
    system_stats['total_anomalies'] += num_anomalies

    # Store alert if anomalies found
    if num_anomalies > 0:
        alerts_store.append({
            'id': f"alert_{len(alerts_store) + 1}",
            'timestamp': datetime.now().isoformat(),
            'anomalies_found': num_anomalies,
            'total_samples': len(features)
        })

    processing_time = (
        datetime.now() - start_time
    ).total_seconds() * 1000

    return DetectionResponse(
        predictions=predictions,
        anomaly_scores=anomaly_scores,
        processing_time_ms=processing_time,
        total_samples=len(features),
        anomalies_found=num_anomalies
    )


@app.get("/alerts", tags=["Alerts"])
async def get_alerts(limit: int = 50):
    """Get recent detection alerts."""
    return {
        "total_alerts": len(alerts_store),
        "alerts": alerts_store[-limit:]
    }


@app.get("/stats", response_model=SystemStats, tags=["Statistics"])
async def get_statistics():
    """Get system statistics and performance metrics."""
    uptime = (
        datetime.now() - system_stats['start_time']
    ).total_seconds()

    total = system_stats['total_analyzed']
    anomalies = system_stats['total_anomalies']
    detection_rate = (anomalies / total * 100) if total > 0 else 0.0

    return SystemStats(
        total_traffic_analyzed=total,
        total_anomalies_detected=anomalies,
        detection_rate=detection_rate,
        uptime_seconds=uptime,
        model_loaded=detector is not None,
        model_type=system_stats['model_type']
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model(model_type: str = "isolation_forest"):
    """Reload the detection model from disk."""
    global detector, system_stats

    model_path = f"models/{model_type}.pkl"

    if not Path(model_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model not found at {model_path}. Train it first."
        )

    try:
        from models.isolation_forest import IsolationForestDetector
        detector = IsolationForestDetector.load(model_path)
        system_stats['model_type'] = model_type

        return {
            "status": "success",
            "message": f"Model '{model_type}' reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )


# ─── Run directly ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )