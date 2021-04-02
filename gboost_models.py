import os
import pathlib
import pickle
from typing import Dict, Tuple
from natsort import natsorted

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import NotFittedError
from tune_sklearn import TuneGridSearchCV, TuneSearchCV
from xgboost import XGBRegressor, XGBClassifier

from base import BaseModel

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GBoostModel(BaseModel):
    def build_model(
        self,
        model_type: str = "xgboost",
        scale_data: bool = False,
        halt_model: bool = False,
        objective: str = "reg:squarederror",
        num_trees: int = 50,
        step_size: float = 0.3,
        device: str = "cpu",
        batch_size: int = 128,
        gamma: int = 0,
        max_bin: int = 256,
    ):

        self.scale_data = scale_data
        if model_type == "xgboost":
            self.single_model = XGBRegressor(objective=objective)
        elif model_type == "lightgbm":
            self.single_model = LGBMRegressor()
        else:
            raise NotImplementedError("Unknown model selected")

        if halt_model:
            logger.info(
                f"Halt model specified, using same model_type for halt classifier: {model_type}"
            )
            if model_type == "xgboost":
                self.halt_model = XGBClassifier()
            elif model_type == "lightgbm":
                self.halt_model = LGBMClassifier()

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

    def partial_fit(self, X, y):

        if not self.model:
            raise NotFittedError("No model found")
        else:
            self.model.partial_fit(X, y)

    def predict(self, X: np.ndarray):

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        if self.scale_data:
            X = self.xscalar.transform(X)

        if self.separate_models:
            pred = []
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

    def save_model(self, filename):

        if self.scale_data:
            logger.info(f"Scale transformations used, saving to {filename}")
            if not self.separate_models:
                if not any([s in filename for s in [".pkl", ".pickle"]]):
                    filename += ".pkl"
                parent_dir = pathlib.Path(filename).parent
                if not parent_dir.exists():
                    parent_dir.mkdir(parents=True, exist_ok=True)
                path_name = str(parent_dir)
            else:
                file_dir = pathlib.Path(filename)
                if not file_dir.exists():
                    logger.info(f"Creating new directories at {file_dir}")
                    file_dir.mkdir(parents=True, exist_ok=True)
                path_name = filename
            pickle.dump(
                self.xscalar, open(os.path.join(path_name, "xscalar.pkl"), "wb")
            )
            pickle.dump(
                self.yscalar, open(os.path.join(path_name, "yscalar.pkl"), "wb")
            )

        if self.separate_models:
            if not pathlib.Path(filename).exists():
                pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
            for i in range(len(self.models)):
                pickle.dump(
                    self.models[i], open(os.path.join(filename, f"model{i}.pkl"), "wb")
                )
        else:
            parent_dir = pathlib.Path(filename).parent
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.model, open(filename, "wb"))

    # def load_model(
    #     self, filename: str, scale_data: bool = False, separate_models: bool = False
    # ):

    #     self.scale_data = scale_data
    #     self.separate_models = separate_models
    #     if self.separate_models:
    #         all_models = os.listdir(filename)
    #         all_models = natsorted(all_models)
    #         if self.scale_data:
    #             all_models = all_models[:-2]
    #         num_models = len(all_models)
    #         models = []
    #         for i in range(num_models):
    #             models.append(
    #                 pickle.load(open(os.path.join(filename, all_models[i]), "rb"))
    #             )
    #         self.models = models
    #     else:
    #         if not any([s in filename for s in [".pkl", ".pickle"]]):
    #             filename += ".pkl"
    #         self.model = pickle.load(open(filename, "rb"))

    #     if scale_data:
    #         if not separate_models:
    #             path_name = str(pathlib.Path(filename).parent)
    #         else:
    #             path_name = filename
    #         self.xscalar = pickle.load(
    #             open(os.path.join(path_name, "xscalar.pkl"), "rb")
    #         )
    #         self.yscalar = pickle.load(
    #             open(os.path.join(path_name, "yscalar.pkl"), "rb")
    #         )

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
        


    def get_feat_importance(self, X = None):

        if not self.model:
            raise Exception("No model found, please run fit first")

        feats = self.features
        outputs = self.labels

        
        feat_importance_d = dict()

        if not self.separate_models:

            if self.model_type in ["xgboost", "lightgbm"]:
                for i,(feat_name,estimator) in enumerate(zip(outputs, self.model.estimators_)):
                    feat_importance_d[feat_name] = estimator.feature_importances_
            
            else:
                raise ValueError("Unknown model type")

        else:
            
            if self.model_type in ["xgboost", "lightgbm"]:
                for i,feat_name in enumerate(outputs):
                    feat_importance_d[feat_name] = self.models[i].feature_importances_
            else:
                raise ValueError("Unknown model type")
            

        # Normalize vector of importances
        for f_name in feat_importance_d.keys():
            feat_importances = np.abs(feat_importance_d[f_name]).tolist()
            importance_sum = sum(feat_importances)
            feat_importances_norm = [imp/importance_sum for imp in feat_importances]
            feat_importance_d[f_name] = feat_importances_norm

        return feat_importance_d



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

    xgm.save_model(filename="models/xgbm_pole_multi.pkl")
