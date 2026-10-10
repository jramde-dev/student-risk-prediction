def _train(client, **body):
    r = client.post("/train", json=body)
    assert r.status_code == 202, r.text
    job = client.get(f"/train/{r.json()['job_id']}").json()
    return job


def test_health_without_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] == {"regression": False, "classification": False}


def test_predict_before_training_returns_503(client, student):
    assert client.post("/predict", json=student).status_code == 503


def test_train_then_predict(client, student):
    job = _train(client)
    assert job["status"] == "succeeded", job
    assert client.get("/health").json()["models_loaded"] == {"regression": True, "classification": True}

    pred = client.post("/predict", json=student).json()
    assert 0 <= pred["mental_health_score"] <= 100
    assert 0 <= pred["at_risk_probability"] <= 1
    assert isinstance(pred["at_risk"], bool)

    info = client.get("/models").json()
    assert "r2" in info["metadata"]["regression"]["metrics"]
    assert "pr_auc" in info["metadata"]["classification"]["metrics"]


def test_risky_profile_scores_higher_than_healthy_one(client, student):
    _train(client)
    healthy = {**student, "Daily_Social_Media_Hours": 1.0, "Sleep_Hours": 8.5, "Physical_Activity_Hours": 2.0}
    p_risk = client.post("/predict", json=student).json()["at_risk_probability"]
    p_ok = client.post("/predict", json=healthy).json()["at_risk_probability"]
    assert p_risk > p_ok


def test_threshold_override(client, student):
    _train(client)
    assert client.post("/predict?threshold=0", json=student).json()["at_risk"] is True
    assert client.post("/predict?threshold=1", json=student).json()["at_risk"] is False


def test_partial_training_only_classification(client, student):
    assert _train(client, tasks=["classification"])["status"] == "succeeded"
    pred = client.post("/predict", json=student).json()
    assert pred["mental_health_score"] is None and pred["at_risk_probability"] is not None


def test_batch(client, student):
    _train(client)
    r = client.post("/predict/batch", json={"students": [student, student]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_validation_errors(client, student):
    assert client.post("/predict", json={**student, "Age": 5}).status_code == 422
    assert client.post("/predict", json={**student, "Gender": "X"}).status_code == 422
    assert client.post("/predict", json={**student, "Sleep_Hours": 23}).status_code == 422  # > 24 h au total
    assert client.post("/predict/batch", json={"students": []}).status_code == 422


def test_unknown_job_and_algorithm(client):
    assert client.get("/train/nope").status_code == 404
    assert client.post("/train", json={"regression_algorithm": "svm"}).status_code == 422


def test_missing_dataset_is_rejected(client):
    settings = client.app.state.settings
    settings.data_path = settings.data_path.parent / "absent.csv"
    assert client.post("/train", json={}).status_code == 422


def test_failed_job_is_reported_and_lock_released(client):
    settings = client.app.state.settings
    good = settings.data_path
    bad = good.parent / "bad.csv"
    bad.write_text("a,b\n1,2\n")  # colonnes requises absentes
    settings.data_path = bad
    job = _train(client)
    assert job["status"] == "failed" and "Colonnes manquantes" in job["error"]

    settings.data_path = good  # le verrou doit avoir été libéré
    assert _train(client)["status"] == "succeeded"


def test_train_with_optuna(client):
    job = _train(client, tasks=["classification"], optuna_trials=2)
    assert job["status"] == "succeeded", job
    best = client.get("/models").json()["metadata"]["classification"]["best_params"]
    assert "num_leaves" in best


def test_training_disabled_returns_403(client):
    client.app.state.settings.training_enabled = False
    r = client.post("/train", json={})
    assert r.status_code == 403
    assert client.get("/health").json()["training_enabled"] is False
