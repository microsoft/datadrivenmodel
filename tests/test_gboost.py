from gboost_models import GBoostModel

xgboost_model = GBoostModel()

X, y = xgboost_model.load_csv(
    dataset_path="csv_data/cartpole-log.csv",
    max_rows=1000,
    augm_cols=["action_command", "config_length", "config_masspole"],
)


def test_shape():

    assert X.shape[0] == 980 == y.shape[0]
    assert X.shape[1] == xgboost_model.input_dim
    assert y.shape[1] == xgboost_model.output_dim
