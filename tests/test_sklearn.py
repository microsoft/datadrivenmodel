from skmodels import SKModel
import numpy as np
import pathlib


X, y = SKModel.load_csv(
    dataset_path="csv_data/cartpole-log.csv",
    max_rows=1000,
    augm_cols=["action_command", "config_length", "config_masspole"],
)


def test_shape():

    assert X.shape[0] == 980 == y.shape[0]
    assert X.shape[1] == SKModel.input_dim
    assert y.shape[1] == SKModel.output_dim


def test_svm_train():

    if not pathlib.Path("tmp").exists():
        pathlib.Path("tmp").mkdir(parents=True, exist_ok=True)
    lsvm = SKModel()
    lsvm.build_model(model_type="LinearSVC")
    lsvm.fit(X, y)
    lsvm.save_model(dir_path="tmp/lsvm_pole.pkl")

    lsvm2 = SKModel()
    lsvm2.load_model(dir_path="tmp/lsvm_pole.pkl")

    yhat0 = lsvm.predict(X)
    yhat = lsvm2.predict(X)

    assert np.array_equal(yhat, yhat0)


def test_linear_train():

    if not pathlib.Path("tmp").exists():
        pathlib.Path("tmp").mkdir(parents=True, exist_ok=True)
    linear = SKModel()
    linear.build_model(model_type="Linear_model")
    linear.fit(X, y)
    linear.save_model(dir_path="tmp/linear_pole.pkl")

    linear2 = SKModel()
    linear2.load_model(dir_path="tmp/linear_pole.pkl")

    yhat0 = linear.predict(X)
    yhat = linear2.predict(X)

    assert np.array_equal(yhat, yhat0)


def test_gbr_train():

    if not pathlib.Path("tmp").exists():
        pathlib.Path("tmp").mkdir(parents=True, exist_ok=True)
    
    gbr = SKModel()
    gbr.build_model(model_type="GradientBoostingRegressor")
    gbr.fit(X, y)
    gbr.save_model(dir_path="tmp/gbr_pole.pkl")

    gbr2 = SKModel()
    gbr2.load_model(dir_path="tmp/gbr_pole.pkl")

    yhat0 = gbr.predict(X)
    yhat = gbr2.predict(X)

    assert np.array_equal(yhat, yhat0)
