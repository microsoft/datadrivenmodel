import os
import pathlib
import pickle
from typing import Dict, Tuple

import numpy as np
import copy

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from natsort import natsorted

from tune_sklearn import TuneSearchCV
from tune_sklearn import TuneGridSearchCV
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

from ray.tune.sklearn import TuneGridSearchCV, TuneSearchCV

from base import BaseModel
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# param_grid = {"learning_rate": (0.01, 0.1), "n_estimators": (25, 250), "subsample": [False, True]}


class SKModel(BaseModel):
    def build_model(
        self,
        model_type: str = "linear_model",
        scale_data: bool = False,
        fit_separate: bool = False,
    ):
        self.scale_data = scale_data
        self.model_type = model_type
        self.fit_separate = fit_separate
        if model_type == "linear_model":
            self.model = linear_model.LinearRegression()
        elif model_type == "SVR":
            self.model = SVR(C=1.0, epsilon=0.2)
        elif model_type == "GradientBoostingRegressor":
            self.model = GradientBoostingRegressor()
        else:
            raise NotImplementedError("unknown model selected")

    def fit(self, X, y):

        if self.scale_data:
            X, y = self.scalar(X, y)

        if (
            self.model_type == "GradientBoostingRegressor"
            and self.fit_separate == False
        ):
            fit_separate = True
            logger.warn(
                "Note: fit_separate must be set toTrue for GradientBoostingRegressor, but False was provided. Changing to True"
            )

        if self.model_type == "SVR" and self.fit_separate == False:
            fit_separate = True
            logger.warn(
                "Note: fit_separate must be set to True for SVR, but False was provided. Changing to True"
            )

        self.separate_models = self.fit_separate

        if self.separate_models:
            self.models = []
            for i in range(y.shape[1]):
                logger.info(f"Fitting model {i+1} of {y.shape[1]}")
                # ensure model doesn't change in between per-output models
                aux_model = copy.deepcopy(self.model.fit(X, y[:, i]))
                self.models.append(aux_model)
        else:
            try:
                self.model.fit(X, y)
            except ValueError:
                logger.info(
                    f"fit separate should be True for model type of {self.model_type}"
                )

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
                path_name = os.path.splitext(filename)[0]
                if not pathlib.Path(path_name).exists():
                    pathlib.Path(path_name).mkdir(parents=True, exist_ok=True)

            pickle.dump(
                self.xscalar, open(os.path.join(path_name, "xscalar.pkl"), "wb")
            )
            pickle.dump(
                self.yscalar, open(os.path.join(path_name, "yscalar.pkl"), "wb")
            )

        if self.separate_models:
            path_name = os.path.splitext(filename)[0]
            if not pathlib.Path(path_name).exists():
                pathlib.Path(path_name).mkdir(parents=True, exist_ok=True)
            for i in range(len(self.models)):
                pickle.dump(
                    self.models[i], open(os.path.join(path_name, f"model{i}.pkl"), "wb")
                )
        else:
            parent_dir = pathlib.Path(filename).parent
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.model, open(filename, "wb"))

    # def load_model(
    #     self, filename: str, scale_data: bool = False, separate_models: bool = False
    # ):

    #     self.separate_models = separate_models
    #     if self.separate_models:
    #         all_models = os.listdir(filename)
    #         all_models = natsorted(all_models)
    #         num_models = len(all_models)
    #         models = []
    #         for i in range(num_models):
    #             models.append(
    #                 pickle.load(open(os.path.join(filename, all_models[i]), "rb"))
    #             )
    #         self.models = models
    #     else:
    #         self.model = pickle.load(open(filename, "rb"))

    #     self.scale_data = scale_data

    def sweep(self, X, y, params: Dict = None):
        if not params:
            raise NotImplementedError

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

        if self.model_type == "SVR":
            # Note, .coef_ is only available for linear models
            return BaseModel.get_feat_importance(self, X)

        if self.model_type in ["linear_model"] and not self.scale_data:
            # Note, coefficients do NOT match feature importance for NON scaled data (self.scale_data == False)
            return BaseModel.get_feat_importance(self, X)

        feats = self.features
        outputs = self.labels
        
        feat_importance_d = dict()

        if not self.separate_models:

            if self.model_type in ["GradientBoostingRegressor"]:
                for i,(feat_name,estimator) in enumerate(zip(outputs, self.model.estimators_)):
                    feat_importance_d[feat_name] = estimator.feature_importances_
            
            elif self.model_type in ["linear_model"]:
                print("self.model.coef_", self.model.coef_)
                for i,(feat_name,coefficients) in enumerate(zip(outputs, self.model.coef_)):
                    feat_importance_d[feat_name] = coefficients
            
            else:
                raise ValueError("Unknown model type")

        else:
            
            if self.model_type in ["GradientBoostingRegressor"]:
                for i,feat_name in enumerate(outputs):
                    feat_importance_d[feat_name] = self.models[i].feature_importances_
            elif self.model_type in ["linear_model"]:
                for i,feat_name in enumerate(outputs):
                    feat_importance_d[feat_name] = self.models[i].coef_
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

    """Example using an sklearn Pipeline with TuneGridSearchCV.

    Example taken and modified from
    https://scikit-learn.org/stable/auto_examples/compose/
    plot_compare_reduction.html
    """

    skm = SKModel()
    X, y = skm.load_csv(
        dataset_path="csv_data/cartpole-log.csv",
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
    )

    skm.build_model(model_type="linear_model")
    skm.fit(X, y, fit_separate=False)
    logger.info(X)
    yhat = skm.predict(X)

    skm.save_model(dir_path="models/linear_pole_multi.pkl")

    skm = SKModel()
    X, y = skm.load_csv(
        dataset_path="csv_data/cartpole-log.csv",
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
    )

    skm.build_model(model_type="SVR")
    skm.fit(X, y, fit_separate=False)
    logger.info(X)
    yhat = skm.predict(X)

    skm.save_model(dir_path="models/lsvc_pole_multi.pkl")

    skm.build_model(model_type="GradientBoostingRegressor")
    skm.fit(X, y, fit_separate=False)
    logger.info(X)
    yhat = skm.predict(X)

    skm.save_model(dir_path="models/gbr_pole_multi.pkl")

    # from sklearn.model_selection import GridSearchCV
    # from sklearn.datasets import load_digits
    # from sklearn.pipeline import Pipeline
    # from sklearn.svm import LinearSVC
    # from sklearn.decomposition import PCA, NMF
    # from sklearn.feature_selection import SelectKBest, chi2

    # from tune_sklearn import TuneSearchCV
    # from tune_sklearn import TuneGridSearchCV

    # pipe = Pipeline(
    #     [
    #         # the reduce_dim stage is populated by the param_grid
    #         ("reduce_dim", "passthrough"),
    #         ("classify", LinearSVC(dual=False, max_iter=10000)),
    #     ]
    # )

    # N_FEATURES_OPTIONS = [2, 4, 8]
    # C_OPTIONS = [1, 10]
    # param_grid = [
    #     {
    #         "reduce_dim": [PCA(iterated_power=7), NMF()],
    #         "reduce_dim__n_components": N_FEATURES_OPTIONS,
    #         "classify__C": C_OPTIONS,
    #     },
    #     {
    #         "reduce_dim": [SelectKBest(chi2)],
    #         "reduce_dim__k": N_FEATURES_OPTIONS,
    #         "classify__C": C_OPTIONS,
    #     },
    # ]

    # random = TuneSearchCV(pipe, param_grid, search_optimization="random")
    # X, y = load_digits(return_X_y=True)
    # random.fit(X, y)
    # logger.info(random.cv_results_)

    # grid = TuneGridSearchCV(pipe, param_grid=param_grid)
    # grid.fit(X, y)
    # logger.info(grid.cv_results_)

