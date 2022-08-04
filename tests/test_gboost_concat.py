from gboost_models import GBoostModel
import numpy as np
import pathlib

from assessment_metrics_loader import available_metrics

xgboost_model = GBoostModel()
lgbm = GBoostModel()

X, y, X_test, y_test = xgboost_model.load_csv(
    dataset_path="csv_data/cartpole_st1_at.csv",
    max_rows=1000,
    augm_cols=["action_command", "config_length", "config_masspole"],
    test_perc=0.15,
    debug=True,
    concatenated_steps=6,
    concatenated_zero_padding=True,
)


def test_shape():

    # TRAIN SET
    assert (
        X.shape[0] == 833 == y.shape[0]
    ), f"X.shape[0] ({X.shape[0]}) -- y.shape[0] ({y.shape[0]})"
    assert (
        X.shape[1] == 42 == xgboost_model.input_dim
    ), f"X.shape[1] ({X.shape[1]}) -- input_dim ({xgboost_model.input_dim})"
    assert (
        y.shape[1] == 4 == xgboost_model.output_dim
    ), f"y.shape[1] ({y.shape[1]}) -- output_dim ({xgboost_model.output_dim})"

    # TEST SET
    assert (
        X_test.shape[0] == 147 == y_test.shape[0]
    ), f"X_test.shape[0] ({X_test.shape[0]}) -- y_test.shape[0] ({y_test.shape[0]})"
    assert (
        X_test.shape[1] == 42 == xgboost_model.input_dim
    ), f"X.shape[1] ({X.shape[1]}) -- input_dim ({xgboost_model.input_dim})"
    assert (
        y_test.shape[1] == 4 == xgboost_model.output_dim
    ), f"y.shape[1] ({y.shape[1]}) -- output_dim ({xgboost_model.output_dim})"


def test_lgm_train():

    if not pathlib.Path("tmp").exists():
        pathlib.Path("tmp").mkdir(parents=True, exist_ok=True)

    lgbm.build_model(model_type="lightgbm")
    lgbm.fit(X, y)
    lgbm.save_model(filename="tmp/gbm_pole.pkl")

    lgbm2 = GBoostModel()
    lgbm2.load_model(filename="tmp/gbm_pole.pkl")
    yhat0 = lgbm.predict(X)
    yhat = lgbm2.predict(X)

    assert np.array_equal(yhat, yhat0)


def test_xgb_train():

    xgboost_model.build_model(model_type="xgboost")
    xgboost_model.fit(X, y)
    xgboost_model.save_model(filename="tmp/gbm_pole.pkl")

    xgm2 = GBoostModel()
    xgm2.load_model(filename="tmp/gbm_pole.pkl")
    yhat0 = xgboost_model.predict(X)
    yhat = xgm2.predict(X)

    assert np.array_equal(yhat, yhat0)


def test_xgb_eval():

    xgboost_model.build_model(model_type="xgboost")
    xgboost_model.fit(X, y)
    xgboost_model.save_model(filename="tmp/gbm_pole.pkl")

    eval_metric = available_metrics["root_mean_squared_error"]
    y_hat = xgboost_model.predict(X_test)
    eval_out = xgboost_model.evaluate(eval_metric, y_hat, y_test)

    # Add evaluation metric check to test changes to model
    assert round(eval_out, 10) == round(0.10077187689939525, 10)
