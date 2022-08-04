from skmodels import SKModel
import numpy as np
import pathlib

skmodel = SKModel()
X, y, _, _ = skmodel.load_csv(
    dataset_path="csv_data/cartpole_st1_at.csv",
    max_rows=1000,
    augm_cols=["action_command", "config_length", "config_masspole"],
    test_perc=0.15,
)


def test_shape():

    assert X.shape[0] == 833 == y.shape[0]
    assert X.shape[1] == skmodel.input_dim
    assert y.shape[1] == skmodel.output_dim


def test_svm_train():

    if not pathlib.Path("tmp").exists():
        pathlib.Path("tmp").mkdir(parents=True, exist_ok=True)
    lsvm = SKModel()
    lsvm.build_model(model_type="SVR", fit_separate=True)
    lsvm.fit(X, y)
    lsvm.save_model(filename="tmp/lsvm_pole")

    lsvm2 = SKModel()
    lsvm2.load_model(filename="tmp/lsvm_pole", separate_models=True)

    yhat0 = lsvm.predict(X)
    yhat = lsvm2.predict(X)

    assert np.array_equal(yhat, yhat0)


def test_linear_train():

    if not pathlib.Path("tmp").exists():
        pathlib.Path("tmp").mkdir(parents=True, exist_ok=True)
    linear = SKModel()
    linear.build_model(model_type="linear_model")
    linear.fit(X, y)
    linear.save_model(filename="tmp/linear_pole.pkl")

    linear2 = SKModel()
    linear2.load_model(filename="tmp/linear_pole.pkl")

    yhat0 = linear.predict(X)
    yhat = linear2.predict(X)

    assert np.array_equal(yhat, yhat0)


def test_gbr_train():

    if not pathlib.Path("tmp").exists():
        pathlib.Path("tmp").mkdir(parents=True, exist_ok=True)

    gbr = SKModel()
    gbr.build_model(model_type="GradientBoostingRegressor", fit_separate=True)
    gbr.fit(X, y)
    gbr.save_model(filename="tmp/gbr_pole")

    gbr2 = SKModel()
    gbr2.load_model(filename="tmp/gbr_pole", separate_models=True)

    yhat0 = gbr.predict(X)
    yhat = gbr2.predict(X)

    assert np.array_equal(yhat, yhat0)
