import numpy as np
from typing import Tuple, Dict

from base import BaseModel
from tune_sklearn import TuneGridSearchCV, TuneSearchCV
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class GBoostModel(BaseModel):
    def build_model(self, model_type: str = "xgboost"):

        if model_type == "xgboost":
            single_model = XGBRegressor(objective="reg:squarederror")
        elif model_type == "lightgbm":
            single_model = LGBMRegressor()
        else:
            raise NotImplementedError("Unknown model selected")

        self.model = MultiOutputRegressor(single_model)

    def sweep(self, params: Dict, X, y):

        tune_search = TuneSearchCV(
            self.model,
            param_distributions=params,
            n_trials=3,
            # early_stopping=True,
            # use_gpu=True  # Commented out for testing on travis,
            # but this is how you would use gpu
        )

        tune_search.fit(X, y)

        return tune_search


if __name__ == "__main__":

    xgm = BoostModel()
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
