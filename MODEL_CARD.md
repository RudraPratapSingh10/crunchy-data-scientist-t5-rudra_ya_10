# Model Card: Spaceship Titanic Prediction

## 1. Model Overview

| Property | Details |
|---|---|
| Project | Spaceship Titanic Prediction API |
| Model | Gradient Boosting Classifier |
| Version | v1 |
| Problem Type | Binary Classification |
| Target Variable | Transported |
| Framework | Scikit-learn |
| Model Serialization | Joblib |

This model predicts whether a passenger was transported to another dimension during the fictional Spaceship Titanic incident.

## 2. Intended Use

The model is designed for:
- Educational and portfolio demonstration.
- Demonstrating an end-to-end machine learning pipeline.
- Exploring passenger characteristics and transportation outcomes.
- Demonstrating API deployment, input validation, logging and drift monitoring.

It is not intended for real-world safety-critical or high-stakes decisions.

## 3. Training Data

- **Dataset:** Spaceship Titanic
- **Source:** Kaggle Spaceship Titanic competition.
- **Training file:** `data/train.csv`
- **Data type:** Structured tabular data.
- **Training records:** 6,954
- **Validation records:** 1,739
- **Data split:** 80% training / 20% validation.
- **Random state:** 42
- **Collection period:** Not documented in the provided dataset.

### Features Used

**Numerical Features**
- Age
- RoomService
- FoodCourt
- ShoppingMall
- Spa
- VRDeck
- PassengerNumber
- GroupSize
- CabinNumber
- TotalSpending

**Categorical Features**
- HomePlanet
- CryoSleep
- Destination
- VIP
- Deck
- Side

## 4. Model Architecture and Training

The model uses a Scikit-learn Pipeline consisting of:

1. Feature engineering.
2. Numerical preprocessing.
3. Categorical preprocessing.
4. Gradient Boosting Classifier.

### Preprocessing

**Numerical Features**
- Missing values handled using median imputation.
- StandardScaler used for scaling.

**Categorical Features**
- Missing values handled using most-frequent imputation.
- OneHotEncoder used for encoding.
- Unknown categories handled using `handle_unknown="ignore"`.

### Model Hyperparameters

| Parameter | Value |
|---|---|
| n_estimators | 150 |
| learning_rate | 0.1 |
| max_depth | 3 |
| random_state | 42 |

**Saved Model:** `artifacts/models/gradient_boosting_v1.joblib`

## 5. Evaluation Results

The model was evaluated on a held-out validation dataset.

| Metric | Score |
|---|---:|
| Accuracy | 0.8039 |
| ROC-AUC | 0.9058 |

### Confusion Matrix

| Actual / Predicted | Not Transported | Transported |
|---|---:|---:|
| Not Transported | 679 | 184 |
| Transported | 157 | 719 |

The model achieved approximately 80.39% validation accuracy.

These results represent performance on the project's validation dataset and do not guarantee performance on unseen real-world data.

### Performance by Passenger Group

Detailed subgroup evaluation has not yet been completed. Therefore, equal performance across different passenger groups cannot be guaranteed.

## 6. Limitations and Potential Failure Modes

- The model is trained on a fictional competition dataset.
- Missing or unusual input values may affect prediction quality.
- Previously unseen passenger patterns may reduce model reliability.
- The training dataset may contain biases that affect predictions.
- High prediction confidence does not guarantee correctness.
- Changes in incoming data may reduce model performance.
- Basic Age drift monitoring does not detect every possible type of data drift.

## 7. Out-of-Scope Uses

The model must not be used for:
- Real-world passenger safety decisions.
- Medical, financial, legal or employment decisions.
- Decisions affecting individuals without independent verification.
- Applications requiring guaranteed accuracy or fairness.

## 8. Potentially Affected Groups

The dataset contains passenger characteristics such as age, home planet, destination and VIP status.

Prediction errors or biases could disproportionately affect certain passenger groups if the model is used outside its intended educational purpose.

Subgroup fairness has not yet been established through dedicated testing.

## 9. Monitoring and Logging

The deployed FastAPI application includes:

- Input validation using Pydantic.
- Prediction logging.
- Confidence score logging.
- API latency measurement.
- Basic Age feature drift monitoring.

### Drift Monitoring

The monitoring system:
- Uses a training baseline Age mean of 28.83.
- Collects incoming Age values in batches of five requests.
- Calculates the current batch mean.
- Compares the current mean against the baseline.
- Triggers a drift alert when relative change exceeds 20%.

This is a basic demonstration of drift monitoring, not a complete production monitoring solution.

## 10. Reproducibility and Project Files

| Component | Location |
|---|---|
| Training Script | `src/train.py` |
| Preprocessing | `src/preprocessing.py` |
| Model Artifact | `artifacts/models/gradient_boosting_v1.joblib` |
| Training Metrics | `reports/training_metrics.json` |
| FastAPI Application | `api/main.py` |
| Drift Monitoring | `monitoring/drift_monitor.py` |
| Prediction Logs | `logs/prediction.log` |

### Reproduce Training

Run the following command from the project root:

```bash
python -m src.train
```

The pipeline trains the model and saves the resulting model artifact and evaluation metrics.

## 11. Disclaimer

This model is an educational machine learning project developed for learning and demonstration purposes.

Its predictions should not be considered factual, authoritative or suitable for high-stakes decisions.