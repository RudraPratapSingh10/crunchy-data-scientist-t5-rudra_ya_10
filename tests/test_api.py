from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200


def test_valid_prediction():
    payload = {
        "PassengerId": "9999_01",
        "HomePlanet": "Earth",
        "CryoSleep": False,
        "Cabin": "G/100/P",
        "Destination": "TRAPPIST-1e",
        "Age": 30,
        "VIP": False,
        "RoomService": 100,
        "FoodCourt": 200,
        "ShoppingMall": 50,
        "Spa": 0,
        "VRDeck": 0,
        "GroupSize": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["result"] in ["Transported", "Not Transported"]
    assert 0 <= data["confidence"] <= 1
    assert data["latency_ms"] >= 0


def test_invalid_age():
    payload = {
        "PassengerId": "9999_01",
        "Age": 150
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422