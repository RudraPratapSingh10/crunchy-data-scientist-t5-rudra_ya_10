import json
from pathlib import Path
from datetime import datetime

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

LOG_DIR = BASE_DIR / "monitoring" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_LOG = LOG_DIR / "drift_report.json"


# Training dataset Age baseline
BASELINE_AGE_MEAN = 28.83
BASELINE_AGE_STD = 14.49

# Drift threshold: 20% change in mean
DRIFT_THRESHOLD = 0.20


def check_age_drift(current_ages):

    current_ages = np.array(current_ages, dtype=float)

    current_mean = float(np.mean(current_ages))

    relative_change = abs(
        current_mean - BASELINE_AGE_MEAN
    ) / BASELINE_AGE_MEAN

    drift_detected = relative_change > DRIFT_THRESHOLD

    report = {
        "timestamp": datetime.now().isoformat(),
        "feature": "Age",
        "baseline_mean": BASELINE_AGE_MEAN,
        "current_mean": round(current_mean, 2),
        "relative_change": round(relative_change, 4),
        "drift_detected": drift_detected
    }

    with open(DRIFT_LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(report) + "\n")

    return report