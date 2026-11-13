import time
import logging
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from monitoring.drift_monitor import check_age_drift


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
# Logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "prediction.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Store recent Age values for drift monitoring
AGE_HISTORY = []
DRIFT_BATCH_SIZE = 5

MODEL_PATH = (
    BASE_DIR
    / "artifacts"
    / "models"
    / "gradient_boosting_v1.joblib"
)


# Load trained model
model = joblib.load(MODEL_PATH)


# FastAPI application
app = FastAPI(
    title="Spaceship Titanic Prediction API",
    description="Predict whether a passenger was transported.",
    version="1.0.0"
)


# Input validation
class PassengerInput(BaseModel):

    PassengerId: str = Field(
        examples=["0001_01"]
    )

    HomePlanet: str | None = None

    CryoSleep: bool | None = None

    Cabin: str | None = Field(
        default=None,
        examples=["B/0/P"]
    )

    Destination: str | None = None

    Age: float | None = Field(
        default=None,
        ge=0,
        le=120
    )

    VIP: bool | None = None

    RoomService: float | None = Field(
        default=None,
        ge=0
    )

    FoodCourt: float | None = Field(
        default=None,
        ge=0
    )

    ShoppingMall: float | None = Field(
        default=None,
        ge=0
    )

    Spa: float | None = Field(
        default=None,
        ge=0
    )

    VRDeck: float | None = Field(
        default=None,
        ge=0
    )

    GroupSize: int = Field(
        default=1,
        ge=1
    )


# Health check
@app.get("/")
def home():

    return {
        "status": "online",
        "message": "Spaceship Titanic API is running",
        "model": "Gradient Boosting",
        "version": "1.0.0"
    }


# Prediction endpoint
@app.post("/predict")
def predict_passenger(passenger: PassengerInput):

    try:

        start_time = time.perf_counter()

        passenger_data = passenger.model_dump()
        
        # Collect Age values for drift monitoring
        if passenger_data.get("Age") is not None:
            AGE_HISTORY.append(passenger_data["Age"])

        # Run drift check after collecting 5 requests
        if len(AGE_HISTORY) >= DRIFT_BATCH_SIZE:
            drift_result = check_age_drift(AGE_HISTORY)
            logger.info("Drift Monitoring Result: %s", drift_result)
            AGE_HISTORY.clear()

        df = pd.DataFrame([passenger_data])

        from src.preprocessing import create_features
        from src.preprocessing import (
            NUMERICAL_FEATURES,
            CATEGORICAL_FEATURES
        )

        processed_df = create_features(df)

        feature_columns = (
            NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        )

        X = processed_df[feature_columns]

        prediction = int(model.predict(X)[0])

        probabilities = model.predict_proba(X)[0]

        confidence = float(max(probabilities))
        
        logger.info(
    "Input: %s | Prediction: %s | Confidence: %.4f",
    passenger_data,
    prediction,
    confidence
)

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "prediction": prediction,
            "result": (
                "Transported"
                if prediction == 1
                else "Not Transported"
            ),
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 3)
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )