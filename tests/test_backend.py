import os
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from backend.main import app

client = TestClient(app)


def make_plan_request(**overrides):
    base = {
        "age": 28,
        "sex": "female",
        "goal": "general_fitness",
        "experience": "beginner",
        "injuries": [],
        "days_per_week": 3,
        "time_per_workout_min": 45,
        "equipment_available": ["bodyweight", "dumbbell"],
    }
    base.update(overrides)
    return base


def test_recommend_plan_basic_structure():
    payload = make_plan_request()
    resp = client.post("/recommend-plan", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Basic keys
    assert "summary" in data
    assert "weekly_schedule" in data
    assert isinstance(data["weekly_schedule"], list)
    assert len(data["weekly_schedule"]) == payload["days_per_week"]

    # Check structure of a day
    day0 = data["weekly_schedule"][0]
    assert set(["day", "focus", "exercises"]) <= set(day0.keys())
    assert isinstance(day0["exercises"], list)
    if day0["exercises"]:
        ex0 = day0["exercises"][0]
        # exercise item basic fields
        assert "title" in ex0


def test_validation_error_for_age_out_of_range():
    payload = make_plan_request(age=8)  # below minimum 13
    resp = client.post("/recommend-plan", json=payload)
    assert resp.status_code == 422

    payload = make_plan_request(age=150)  # above max 100
    resp = client.post("/recommend-plan", json=payload)
    assert resp.status_code == 422


def test_validation_error_for_days_per_week():
    payload = make_plan_request(days_per_week=0)
    resp = client.post("/recommend-plan", json=payload)
    assert resp.status_code == 422

    payload = make_plan_request(days_per_week=8)
    resp = client.post("/recommend-plan", json=payload)
    assert resp.status_code == 422


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_rag_build_index_then_ask():
    # Build or refresh index
    resp_build = client.post("/rag/build-index")
    assert resp_build.status_code == 200, resp_build.text
    data_build = resp_build.json()
    assert data_build.get("status") in {"ok", "exists", "built"} or "ok" in data_build.get("status", "")

    # Ask a basic question
    question = "What are good beginner back exercises?"
    resp_ask = client.post("/rag/ask", json={"question": question})
    assert resp_ask.status_code == 200, resp_ask.text
    data_ask = resp_ask.json()
    assert "answer" in data_ask
    assert isinstance(data_ask["answer"], str) and len(data_ask["answer"]) > 0
