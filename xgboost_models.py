import numpy as np
from typing import Tuple, Dict

from base import BaseModel
from tune_sklearn import TuneGridSearchCV, TuneSearchCV
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor


class XGBoostModel(BaseModel):
    def build_model(self):

        single_model = XGBRegressor(objective="reg:squarederror")

        self.model = MultiOutputRegressor(single_model)

    def sweep(self, params: Dict, X, y):

        xg_search = TuneSearchCV(
            self.model,
            param_distributions=params,
            n_trials=3,
            early_stopping=True,
            # use_gpu=True  # Commented out for testing on travis,
            # but this is how you would use gpu
        )

        xg_search.fit(X, y)

        return xg_search


if __name__ == "__main__":

    xgm = XGBoostModel()
    X, y = xgm.load_numpy("/home/alizaidi/bonsai/repsol/data/scenario1")

    xgm.build_model()

    xgm.fit(X, y)

