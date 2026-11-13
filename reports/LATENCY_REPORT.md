# API Latency Report

## 1. Overview

This report documents the prediction response latency of the Spaceship Titanic Prediction API built using FastAPI.

## 2. Test Environment

- **API Framework:** FastAPI
- **Server:** Uvicorn
- **Model:** Gradient Boosting Classifier
- **Model Version:** v1
- **Testing Environment:** Localhost
- **Endpoint:** `POST /predict`
- **Test Date:** 14 November 2026

## 3. Test Methodology

A total of 20 consecutive prediction requests were sent to the API using PowerShell.

The API's returned `latency_ms` field was recorded for each request.

Latency represents the prediction processing time measured inside the API, not the complete client-to-server network response time.

## 4. Results

| Metric | Measured Value |
|---|---:|
| Total Requests | 20 |
| Average Latency | 13.07 ms |
| Minimum Latency | 9.11 ms |
| Maximum Latency | 38.45 ms |

## 5. Analysis

- The average prediction latency was 13.07 ms.
- The fastest recorded prediction took 9.11 ms.
- The slowest recorded prediction took 38.45 ms.
- All 20 prediction requests completed successfully.

The results indicate that the model provides fast prediction responses in the local testing environment.

## 6. Limitations

- Testing was performed locally on localhost.
- Network latency was not included in the reported measurements.
- Only 20 sequential requests were tested.
- Concurrent user and production load testing have not yet been performed.
- Results may vary across hardware and deployment environments.

## 7. Conclusion

The Spaceship Titanic Prediction API demonstrated an average prediction latency of 13.07 ms across 20 requests.

Further testing under concurrent workloads and a deployed environment would be necessary before making production performance claims.