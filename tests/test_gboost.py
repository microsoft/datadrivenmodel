from gboost_models import GBoostModel
import numpy as np
import pathlib

xgboost_model = GBoostModel()
lgbm = GBoostModel()

X, y = xgboost_model.load_csv(
    dataset_path="csv_data/cartpole-log.csv",
    max_rows=1000,
    augm_cols=["action_command", "config_length", "config_masspole"],
)


def test_shape():

    assert X.shape[0] == 980 == y.shape[0]
    assert X.shape[1] == xgboost_model.input_dim
    assert y.shape[1] == xgboost_model.output_dim


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

