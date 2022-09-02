import os
import pandas as pd
from base import BaseModel
from math import ceil, floor
from typing import Dict, Optional

from darts.models.forecasting.nhits import NHiTSModel
from darts.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
from darts.models.forecasting.tft_model import TFTModel
from darts.models.forecasting.varima import VARIMA
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing

# create a class for timeseries data from darts
# where each episode is a single observation
# over multiple time horizons
# need a separate class for model fitting

# input_chunk_length should be set to <= the min number of past_covariates

# designing the data for prediction:
# we will typically only take a `output_chunk_length` to be 1
# and then continuously feed the predicted value into past_covariates

darts_models = {
    "nhits": NHiTSModel,
    "tftmodel": TFTModel,
    "varima": VARIMA,
    "ets": ExponentialSmoothing,
    "sfarima": StatsForecastAutoARIMA,
}


class TimeSeriesDarts(BaseModel):
    def load_from_csv(
        self,
        dataset_path,
        episode_col,
        iteration_col,
        label_cols,
        feature_cols,
        test_perc: float = 0.2,
        return_ts: bool = True,
    ):

        self.episode_col = episode_col
        self.iteration_col = iteration_col
        self.label_cols = label_cols
        self.feature_cols = feature_cols

        pandas_df = pd.read_csv(dataset_path)
        test_ep_start = pandas_df[episode_col].max() - floor(
            pandas_df[episode_col].max() * test_perc
        )
        train_df = pandas_df[pandas_df[episode_col] < test_ep_start]
        test_df = pandas_df[pandas_df[episode_col] >= test_ep_start]

        if return_ts:

            train_features, train_labels = self._ts_group(train_df)
            test_features, test_labels = self._ts_group(test_df)

            return train_features, train_labels, test_features, test_labels
        else:
            return train_df, test_df

    def _ts_group(self, df):

        from darts import timeseries as ts

        features = ts.TimeSeries.from_group_dataframe(
            df[[self.episode_col] + [self.iteration_col] + self.feature_cols],
            group_cols=self.episode_col,
            time_col=self.iteration_col,
        )

        labels = ts.TimeSeries.from_group_dataframe(
            df[[self.episode_col] + [self.iteration_col] + self.label_cols],
            group_cols=self.episode_col,
            time_col=self.iteration_col,
        )

        return features, labels

    def build_model(
        self,
        scale_data: bool = False,
        halt_model: bool = False,
        model_type: str = "nhits",
        build_params: Dict = {},
    ):

        self.scale_data = scale_data
        self.halt_model = halt_model

        model = darts_models[model_type](**build_params)

        self.model = model

    def fit(
        self, df, fit_params: Dict = {},
    ):

        if not self.model:
            raise ValueError("Build model first before fit")

        train_features, train_labels = self._ts_group(df)

        self.model.fit(
            series=train_labels, past_covariates=train_features, **fit_params
        )

    def predict(
        self, df, predict_params: Dict = {},
    ):

        if not self.model:
            raise ValueError("Model not yet fitted")

        covariates, series = self._ts_group(df)

        yhat = self.model.predict(
            series=series, past_covariates=covariates, **predict_params
        )

        return yhat

    def predict_sequentially(self, df, predict_params: Dict = {}, horizon: int = 50):

        # For multi-step prediction, we can either predict a single-step ahead
        # using the predict method and feed the value back into the covariates
        # and predict again, or, create a forward looking matrix for past_covariates
        # and predict until the end of the series.
        # in the case of RL, this is doable when your actions are known ahead of time
        # e.g., if you have a fixed-policy that is independent of states

        covariates, series = self._ts_group(df)

        if not self.model:
            raise ValueError("Model not yet fitted")

        iteration = 0
        yhat_list = []
        while iteration < horizon:
            yhat = self.model.predict(
                series=series, past_covariates=covariates, **predict_params
            )
            yhat_list.append(yhat)
            series = yhat.values()
            iteration += 1


if __name__ == "__main__":

    dataset_path = os.path.join("csv_data", "cartpole_at_st.csv")
    data_df = pd.read_csv(dataset_path)

    feature_cols = ["action_command", "config_length", "config_masspole"]
    label_cols = [
        "state_x_position",
        "state_x_velocity",
        "state_angle_position",
        "state_angle_velocity",
    ]
    episode_col = "episode"
    iteration_col = "iteration"

    darts_model = TimeSeriesDarts()

    train_df, test_df = darts_model.load_from_csv(
        dataset_path,
        episode_col,
        iteration_col,
        label_cols,
        feature_cols,
        0.2,
        return_ts=False,
    )

    nhits_params = {
        "input_chunk_length": 1,
        "output_chunk_length": 1,
        # "pl_trainer_kwargs": {"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True}
    }
    fit_params = {"epochs": 1}
    predict_params = {"n": 1}

    darts_model.build_model(model_type="nhits", build_params=nhits_params)
    darts_model.fit(train_df, fit_params)
    yhat = darts_model.predict(test_df, predict_params)

    # train_df = data_df[data_df["episode"] < 70]
    # test_df = data_df[data_df["episode"] >= 70]

    # from darts import timeseries as ts

    # train_features = ts.TimeSeries.from_group_dataframe(
    #     train_df[[episode_col] + [iteration_col] + feature_cols],
    #     group_cols=episode_col,
    #     time_col=iteration_col,
    # )

    # train_labels = ts.TimeSeries.from_group_dataframe(
    #     train_df[[episode_col] + [iteration_col] + label_cols],
    #     group_cols=episode_col,
    #     time_col=iteration_col,
    # )

    # test_features = ts.TimeSeries.from_group_dataframe(
    #     test_df[[episode_col] + [iteration_col] + feature_cols],
    #     group_cols=episode_col,
    #     time_col=iteration_col,
    # )

    # test_labels = ts.TimeSeries.from_group_dataframe(
    #     test_df[[episode_col] + [iteration_col] + label_cols],
    #     group_cols=episode_col,
    #     time_col=iteration_col,
    # )

    # from darts.models.forecasting.nhits import NHiTSModel

    # nhits_model = NHiTSModel(input_chunk_length=1, output_chunk_length=1)
    # nhits_model.fit(series=train_labels, past_covariates=train_features)
    # yhat = nhits_model.predict(series=test_labels, past_covariates=test_features, n=1)

