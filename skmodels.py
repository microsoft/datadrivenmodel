import copy
import logging
import os
import pathlib
import pickle

import numpy as np
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from base import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SKModel(BaseModel):
    def build_model(
        self,
        model_type: str = "linear_model",
        scale_data: bool = False,
        fit_separate: bool = False,
    ):
        self.scale_data = scale_data
        self.model_type = model_type
        self.separate_models = fit_separate
        if model_type == "linear_model":
            self.model = linear_model.LinearRegression()
        elif model_type == "SVR":
            self.model = SVR(C=1.0, epsilon=0.2)
        elif model_type == "GradientBoostingRegressor":
            self.model = GradientBoostingRegressor()
        elif model_type.lower() == "sgdregressor":
            self.model = SGDRegressor()
        else:
            raise NotImplementedError("unknown model selected")
        if not self.separate_models:
            self.single_model = self.model
            self.model = MultiOutputRegressor(self.single_model)

    def fit(self, X, y):

        if self.scale_data:
            X, y = self.scalar(X, y)

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
                logger.info(f"Unable to fit model of type {type(self.model)}")

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
    skm.fit(X, y)
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
    skm.fit(X, y)
    logger.info(X)
    yhat = skm.predict(X)

    skm.save_model(dir_path="models/lsvc_pole_multi.pkl")

    skm.build_model(model_type="GradientBoostingRegressor")
    skm.fit(X, y)
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
