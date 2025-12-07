import numpy as np
from train_model import (
    read_features, read_model_options, read_model_params,
    initialize_model, evaluate_model
)


def test_read_features():
    features = read_features("features_config.json")
    assert isinstance(features, list)
    assert len(features) > 0


def test_read_model_options():
    opts = read_model_options("model_params.json")
    assert "LogisticRegression" in opts


def test_read_model_params():
    params = read_model_params("model_params.json", "LogisticRegression")
    assert "C" in params


def test_initialize_model():
    params = {"C": 1.0, "max_iter": 100}
    model = initialize_model("LogisticRegression", params)
    assert model is not None


def test_evaluate_model_runs():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    evaluate_model("TestModel", y_true, y_pred)
