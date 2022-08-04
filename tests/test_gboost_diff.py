from gboost_models import GBoostModel
import numpy as np
import pathlib

xgboost_model = GBoostModel()
lgbm = GBoostModel()

X, y, _, _ = xgboost_model.load_csv(
    dataset_path="csv_data/cartpole_st1_at.csv",
    max_rows=1000,
    augm_cols=["action_command", "config_length", "config_masspole"],
    test_perc=0.15,
    diff_state=True,
)

X_test, y_test = xgboost_model.get_test_set(grouped_per_episode=False)


def test_shape():

    # TRAIN SET
    assert (
        X.shape[0] == 833 == y.shape[0]
    ), f"X.shape[0] ({X.shape[0]}) -- y.shape[0] ({y.shape[0]})"
    assert (
        X.shape[1] == 7 == xgboost_model.input_dim
    ), f"X.shape[1] ({X.shape[1]}) -- input_dim ({xgboost_model.input_dim})"
    assert (
        y.shape[1] == 4 == xgboost_model.output_dim
    ), f"y.shape[1] ({y.shape[1]}) -- output_dim ({xgboost_model.output_dim})"

    # TEST SET
    assert (
        X_test.shape[0] == 147 == y_test.shape[0]
    ), f"X_test.shape[0] ({X_test.shape[0]}) -- y_test.shape[0] ({y_test.shape[0]})"
    assert (
        X_test.shape[1] == 7 == xgboost_model.input_dim
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
