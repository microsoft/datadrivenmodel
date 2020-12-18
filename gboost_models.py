import os
import pathlib
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from tune_sklearn import TuneGridSearchCV, TuneSearchCV
from xgboost import XGBRegressor

from base import BaseModel

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GBoostModel(BaseModel):
    def build_model(self, model_type: str = "xgboost", scale_data: bool = False):

        self.scale_data = scale_data
        if model_type == "xgboost":
            self.single_model = XGBRegressor(objective="reg:squarederror")
        elif model_type == "lightgbm":
            self.single_model = LGBMRegressor()
        else:
            raise NotImplementedError("Unknown model selected")

        self.model = MultiOutputRegressor(self.single_model)
        self.model_type = model_type

    def fit(self, X, y, fit_separate: bool = False):

        if self.scale_data:
            X, y = self.scalar(X, y)

        self.separate_models = fit_separate

        if self.separate_models:
            self.models = []
            for i in range(y.shape[1]):

                if self.model_type == "xgboost":
                    boost_model = XGBRegressor(objective="reg:squarederror")
                elif self.model_type == "lightgbm":
                    boost_model = LGBMRegressor()
                else:
                    raise ValueError("Unknown model type")

                logger.info(f"Fitting model {i+1} of {y.shape[1]}")
                self.models.append(boost_model.fit(X, y[:, i]))
        else:
            self.model.fit(X, y)

    def predict(self, X):

        if self.separate_models:
            pred = []
            if self.scale_data:
                X = self.xscalar.transform(X)
            for i in range(len(self.models)):
                logger.debug(f"Predicting model {i} of {len(self.models)}")
                pred.append(self.models[i].predict(X))

            preds = np.array(pred).transpose()
        else:
            preds = self.model.predict(X)
        if self.scale_data:
            preds = self.yscalar.inverse_transform(preds)

        # preds_df = pd.DataFrame(preds)
        # preds_df.columns = label_col_names

        return preds

    def save_model(self, dir_path):

        if self.separate_models:
            if not pathlib.Path(dir_path).exists():
                pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
            # pickle.dump(self.models, open(filename, "wb"))
            for i in range(len(self.models)):
                pickle.dump(
                    self.models[i], open(os.path.join(dir_path, f"model{i}.pkl"), "wb")
                )
        else:
            pickle.dump(self.model, open(dir_path, "wb"))

    def load_model(
        self, dir_path: str, scale_data: bool = False, separate_models: bool = False
    ):

        self.separate_models = separate_models
        if self.separate_models:
            all_models = os.listdir(dir_path)
            all_models.sort()
            num_models = len(all_models)
            models = []
            for i in range(num_models):
                models.append(
                    pickle.load(open(os.path.join(dir_path, all_models[i]), "rb"))
                )
            self.models = models
        else:
            self.model = pickle.load(open(dir_path, "rb"))

        self.scale_data = scale_data

    def sweep(self, params: Dict, X, y):

        tune_search = TuneSearchCV(
            self.model,
            param_distributions=params,
            n_trials=3,
            # early_stopping=True,
            # use_gpu=True
        )

        tune_search.fit(X, y)

        return tune_search


if __name__ == "__main__":

    xgm = GBoostModel()
    X, y = xgm.load_csv(
        dataset_path="csv_data/cartpole-log.csv",
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
    )

    xgm.build_model(model_type="xgboost")
    xgm.fit(X, y, fit_separate=False)
    yhat = xgm.predict(X)

    # xgm.save_model(dir_path="models/xgbm_pole_multi.pkl")
