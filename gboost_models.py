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

    def fit(self, X, y):

        if self.scale_data:
            X, y = self.scalar(X, y)

        self.models = []
        for i in range(y.shape[1]):
            print(f"Fitting model {i} of {y.shape[1]}")
            self.models.append(
                XGBRegressor(objective="reg:squarederror").fit(X, y[:, i])
            )

    def predict(self, X, label_col_names: str = None):

        pred = []
        if self.scale_data:
            X = self.xscalar.transform(X)
        for i in range(len(self.models)):
            logger.debug(f"Predicting model {i} of {len(self.models)}")
            pred.append(self.models[i].predict(X))

        preds = np.array(pred).transpose()
        if self.scale_data:
            preds = self.yscalar.inverse_transform(preds)

        # preds_df = pd.DataFrame(preds)
        # preds_df.columns = label_col_names

        return preds

    def save_model(self, filename):

        if not pathlib.Path(filename).parent.exists():
            pathlib.Path(filename).parent.mkdir(parents=True)
        # pickle.dump(self.models, open(filename, "wb"))
        pickle.dump(self, open(filename, "wb"))

    def load_model(self, filename: str):

        self.models = pickle.load(open(filename, "rb"))

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
    X, y = xgm.load_numpy("/home/alizaidi/bonsai/repsol/data/scenario1")

    xgm.build_model(model_type="lightgbm")

    xgm.fit(X, y)

    param_dists = {
        "n_estimators": [400, 700, 1000],
        "colsample_bytree": [0.7, 0.8],
        "max_depth": [15, 20, 25],
        "num_leaves": [50, 100, 200],
        "reg_alpha": [1.1, 1.2, 1.3],
        "reg_lambda": [1.1, 1.2, 1.3],
        "min_split_gain": [0.3, 0.4],
        "subsample": [0.7, 0.8, 0.9],
        "subsample_freq": [20],
    }

    xgm.sweep(param_dists, X, y)
