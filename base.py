import abc
import copy
import logging
import os
import pathlib
import pickle
from collections import OrderedDict
from typing import Dict, List, Tuple, Union
from omegaconf.listconfig import ListConfig

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    RandomizedSearchCV,
    TimeSeriesSplit,
    PredefinedSplit,
)
from sklearn.preprocessing import StandardScaler

from loaders import CsvReader

logger = logging.getLogger(__name__)
matplotlib.rcParams["figure.figsize"] = [12, 10]


class BaseModel(abc.ABC):
    def __init__(self, log_dirs: str = "logs", model=None):

        self.logs_dir = log_dirs
        self.model = model
        self.halt_model = None

    def load_csv(
        self,
        dataset_path: str,
        input_cols: Union[str, List[str]] = "state",
        augm_cols: Union[str, List[str]] = ["action_command"],
        output_cols: Union[str, List[str]] = "state",
        iteration_order: int = -1,
        episode_col: str = "episode",
        iteration_col: str = "iteration",
        drop_nulls: bool = True,
        max_rows: Union[int, None] = None,
        diff_state: bool = False,
        # calc_config_stats: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read CSV data into two datasets for modeling

        Parameters
        ----------
        dataset_path : str
            path to csv dataset
        input_cols : Union[str, List[str]], optional
            list of columns represent the inputs to the dynamical system in the raw dataset. Can either be a string which is then matched for all columns in the dataset, or a list of strings with exact matches, by default "state"
        augm_cols : Union[str, List[str]], optional
            Exact match of additional columns to use for modeling, such as the actions of the current iteration and any scenario/config parameters, by default ["action_command"]
        output_col : Union[str, List[str]], optional
            output columns of the dynamical system. Can either be a string which is then matched for any columns or a list of exact matches, by default "state"
        iteration_order : int, optional
            in the order of the raw dataset, what is the lag between action t and state t, by default -1, which means each row is
            a_t,s_{t+1}
        max_rows : Union[int, None], optional
            max rows to read for a large dataset, by default None
        diff_state : bool, default False
            If enabled, calculate differential between current output_cols and past output_cols

        Returns
        -------
        Tuple[np.array, np.array]
            Features and labels for modeling


        Raises
        ------
        ValueError
            Data not found
        """

        csv_reader = CsvReader()
        if not os.path.exists(dataset_path):
            raise ValueError(f"No data found at {dataset_path}")
        else:
            if max_rows < 0:
                max_rows = None
            df = pd.read_csv(dataset_path, nrows=max_rows)
            if drop_nulls:
                df = df[~df.isnull().any(axis=1)]
            if type(input_cols) == str:
                base_features = [str(col) for col in df if col.startswith(input_cols)]
            elif isinstance(input_cols, (list, ListConfig)):
                base_features = input_cols
            else:
                raise TypeError(
                    f"input_cols expected type List[str] or str but received type {type(input_cols)}"
                )
            if not augm_cols:
                logging.debug(f"No augmented columns...")
                augm_features = []
            elif type(augm_cols) == str:
                augm_features = [str(col) for col in df if col.startswith(augm_cols)]
            elif isinstance(augm_cols, (list, ListConfig)):
                augm_features = augm_cols
            else:
                raise TypeError(
                    f"augm_cols expected type List[str] or str but received type {type(augm_cols)}"
                )

            if augm_cols:
                features = base_features + augm_features
            else:
                features = base_features
            self.features = features
            logging.info(f"Using {features} as the features for modeling DDM")

            if type(output_cols) == str:
                labels = [col for col in df if col.startswith(output_cols)]
            elif isinstance(output_cols, (list, ListConfig)):
                labels = output_cols
            else:
                raise TypeError(
                    f"output_cols expected type List[str] but received type {type(output_cols)}"
                )
            self.labels = labels
            logging.info(f"Using {labels} as the labels for modeling DDM")

            df = csv_reader.read(
                df,
                iteration_order=iteration_order,
                feature_cols=features,
                label_cols=labels,
                episode_col=episode_col,
                iteration_col=iteration_col,
                augmented_cols=augm_features,
            )
            # TODO: calcualte config summary stats and save somewhere
            # if calc_config_stats:
            #     config_df = df[csv_reader.feature_cols]
            X = df[csv_reader.feature_cols].values
            y = df[csv_reader.label_cols].values
            # store episode_id to group_per_episode
            self.episode_ids = df[episode_col].values

        self.diff_state = diff_state
        if diff_state == True:
            logging.info(
                "delta states enabled, calculating differential between input and output values"
            )
            y = y - X[:, : y.shape[1]]  # s_t+1 - s_t

        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        return X, y

    def load_numpy(
        self, dataset_path: str, X_path: str = "x_set.npy", y_path: str = "y_set.npy"
    ) -> Tuple:

        X = np.load(os.path.join(dataset_path, X_path))
        y = np.load(os.path.join(dataset_path, y_path))
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        return X, y

    def load_pickle_data(self, x_path: str, y_path: str):

        X = pickle.load(open(x_path, "rb"))
        y = pickle.load(open(y_path, "rb"))

        return X, y

    def scalar(self, X, y):

        self.xscalar = StandardScaler()
        self.yscalar = StandardScaler()

        X_scaled = self.xscalar.fit_transform(X)
        y_scaled = self.yscalar.fit_transform(y)

        return X_scaled, y_scaled

    def build_model(self, scale_data: bool = False, halt_model: bool = False):

        self.scale_data = scale_data
        self.halt_model = halt_model

    def fit(self, X, y):

        if not self.model:
            raise ValueError("Please build or load the model first")

        if self.scale_data:
            X, y = self.scalar(X, y)
        self.model.fit(X, y)

    def fit_halt_classifier(self, X, y):

        if not self.halt_model:
            raise ValueError("Please build or load the halted model first")
        if self.scale_data:
            X = self.xscalar.transform(X)
        self.halt_model.fit(X, y)

    def predict(self, X, label_col_names: List[str] = None):

        if not self.model:
            raise ValueError("Please build or load the model first")
        else:
            if self.scale_data:
                X = self.xscalar.transform(X)
            preds = self.model.predict(X)
            if self.scale_data:
                preds = self.yscalar.inverse_transform(preds)

            preds_df = pd.DataFrame(preds)
            preds_df.columns = label_col_names

            return preds_df

    def predict_sequentially_all(self, X, label_col_names: List[str] = None):

        if not self.model:
            raise ValueError("Please build or load the model first")
        else:
            # scaling is performed within "predict" (self.scale_data)

            if label_col_names is None:
                label_col_names = self.labels
                if label_col_names is None:
                    # If None provided, and None stored in self.labels, we ask user to provide as input
                    # - Currently needed to match outputs to inputs when running the model forward -
                    raise ValueError(
                        "Please provide a list of predicted output labels ('label_col_names')"
                    )

            # prepare features & a list of predictions that are feats too (often all)
            feats = self.features
            preds_that_are_feats = [
                f_name for f_name in feats if f_name in label_col_names
            ]
            # initialize feat_dict to first row & pred_dict to match first row too
            feat_dict = OrderedDict(zip(feats, list(X[0])))
            if not self.diff_state:
                pred_dict = dict(
                    [
                        (k, v)
                        for (k, v) in feat_dict.items()
                        if k in preds_that_are_feats
                    ]
                )
            else:
                pred_dict = dict(
                    [(k, 0) for k in feat_dict.keys() if k in preds_that_are_feats]
                )

            # sequentially iterate retriving next prediction based on previous prediction
            preds = []
            for i in range(len(X)):
                for (f_name, x_value) in zip(feats, list(X[i])):
                    if f_name in preds_that_are_feats:
                        if not self.diff_state:
                            feat_dict[f_name] = pred_dict[f_name]
                        else:
                            feat_dict[f_name] += pred_dict[f_name]
                    else:
                        feat_dict[f_name] = x_value
                # get next prediction
                pred = self.predict(np.array([list(feat_dict.values())]))
                preds.append(pred[0])
                # update prediction dictionary (for next iteration)
                pred_dict = OrderedDict(list(zip(label_col_names, list(pred[0]))))

            preds = np.array(preds)  # .transpose()

            # preds_df = pd.DataFrame(preds)
            # preds_df.columns = label_col_names

            return preds  # preds_df

    def predict_sequentially(
        self,
        X,
        label_col_names: List[str] = None,
        it_per_episode: int = None,
        episode_ids: Union[np.ndarray, list, None] = None,
    ):

        if not self.model:
            raise ValueError("Please build or load the model first")
        else:
            # note, scaling is performed in inner function "predict_sequentially_all" > "predict" (self.scale_data)

            if label_col_names is None:
                label_col_names = self.labels
                if label_col_names is None:
                    # If None provided, and None stored in self.labels, we ask user to provide as input
                    # - Currently needed to match outputs to inputs when running the model forward -
                    raise ValueError(
                        "Please provide a list of predicted output labels ('label_col_names')"
                    )

            # group data into episodes
            if episode_ids is not None:
                assert len(X) == len(
                    episode_ids
                ), f"X length ({len(X)}) is different than episode_ids ({len(episode_ids)}) length, when it should be the same"
            X_grouped, _ = self.group_per_episode(X, episode_ids=episode_ids)

            # initialize predictions
            preds = []

            if not it_per_episode:
                # when grouped, X has shape {episode_count, iteration_count, feature_count}
                it_per_episode = np.inf

            num_of_episodes = int(np.shape(X_grouped)[0])

            # iterate per as many episodes as selected
            for i in range(num_of_episodes):

                n_iterations = len(X_grouped[i])
                if it_per_episode >= n_iterations:
                    preds_aux = self.predict_sequentially_all(
                        X_grouped[i], label_col_names
                    )

                else:
                    # split episodes into subepisodes when it_per_episode < episode length
                    n_subepisodes = int(np.ceil(n_iterations / it_per_episode))
                    preds_aux_array = []
                    for j in range(n_subepisodes):
                        preds_aux = self.predict_sequentially_all(
                            X_grouped[i][j * it_per_episode : (j + 1) * it_per_episode],
                            label_col_names,
                        )
                        preds_aux_array.append(copy.deepcopy(preds_aux))

                    preds_aux = np.concatenate(preds_aux_array, axis=0)

                # append to predictions before getting into next episode
                preds.extend(copy.deepcopy(preds_aux))

            preds = np.array(preds)

            # preds_df = pd.DataFrame(preds)
            # preds_df.columns = label_col_names

            return preds  # preds_df

    def predict_halt_classifier(self, X):

        if not self.halt_model:
            raise ValueError("Please build or load the model first")
        else:
            if self.scale_data:
                X = self.xscalar.transform(X)
            halts = self.halt_model.predict(X)

        return halts

    def save_model(self, filename):

        if not any([s in filename for s in [".pkl", ".pickle"]]):
            filename += ".pkl"
        parent_dir = pathlib.Path(filename).parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        if self.scale_data:
            logging.info(f"Scale transformations used, saving to {filename}")
            pickle.dump(
                self.xscalar, open(os.path.join(str(parent_dir), "xscalar.pkl"), "wb")
            )
            pickle.dump(
                self.yscalar, open(os.path.join(str(parent_dir), "yscalar.pkl"), "wb")
            )

        pickle.dump(self.model, open(filename, "wb"))

    def save_halt_model(self, dir_path: str = "models"):

        filename = os.path.join(dir_path, "halted_classifier.pkl")
        if not pathlib.Path(filename).parent.exists():
            pathlib.Path(filename).parent.mkdir(parents=True)
        pickle.dump(self.halt_model, open(filename, "wb"))

    def load_model(
        self, filename: str, scale_data: bool = False, separate_models: bool = False
    ):

        self.separate_models = separate_models
        self.scale_data = scale_data

        if scale_data:
            if not self.separate_models:
                path_name = str(pathlib.Path(filename).parent)
            else:
                path_name = filename
            self.xscalar = pickle.load(
                open(os.path.join(path_name, "xscalar.pkl"), "rb")
            )
            self.yscalar = pickle.load(
                open(os.path.join(path_name, "yscalar.pkl"), "rb")
            )
        if separate_models:
            self._load_multimodels(filename, scale_data)
        else:
            if not any([s in filename for s in [".pkl", ".pickle"]]):
                filename += ".pkl"
            self.model = pickle.load(open(filename, "rb"))

    def _load_multimodels(self, filename: str, scale_data: bool):

        all_models = os.listdir(filename)
        all_models = natsorted(all_models)
        if self.scale_data:
            all_models = all_models[:-2]
        num_models = len(all_models)
        models = []
        for i in range(num_models):
            models.append(
                pickle.load(open(os.path.join(filename, all_models[i]), "rb"))
            )
        self.models = models

    def load_halt_classifier(self, filename: str):

        self.halt_model = pickle.load(open(filename, "rb"))

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, metric, marginal: bool = False
    ):

        if not self.model:
            raise Exception("No model found, please run fit first")
        else:
            if not marginal:
                y_hat = self.predict(X_test)
                return metric(y_test, y_hat)
            else:
                results_df = self.evaluate_margins(X_test, y_test, metric, False)
                return results_df

    def evaluate_sequentially(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metric,
        marginal: bool = False,
        it_per_episode=100,
    ):

        if not self.model:
            raise Exception("No model found, please run fit first")
        else:

            if not marginal:
                y_hat = self.predict_sequentially(X_test, it_per_episode=it_per_episode)
                y_hat_len = np.shape(y_hat)[0]
                y_test = y_test[:y_hat_len]
                return metric(y_test, y_hat)
            else:
                results_df = self.evaluate_margins_sequentially(
                    X_test, y_test, metric, False, it_per_episode=it_per_episode
                )
                return results_df

    def evaluate_margins(
        self, X_test: np.ndarray, y_test: np.ndarray, metric, verbose: bool = False
    ):

        y_pred = self.predict(X_test)
        idx = 0
        results = {}
        for var in self.labels:
            scores = metric(y_test[:, idx], y_pred[:, idx])
            if verbose:
                print(f"Score for var {var}: {scores}")
            results[var] = scores
            idx += 1
        return pd.DataFrame(results.items(), columns=["var", "score"])

    def evaluate_margins_sequentially(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metric,
        verbose: bool = False,
        it_per_episode: int = 100,
    ):

        # Extract prediction and remove any tail reminder from int(len(X_test)/it_per_episode)
        y_pred = self.predict_sequentially(X_test, it_per_episode=it_per_episode)
        y_pred_len = np.shape(y_pred)[0]
        y_test = y_test[:y_pred_len]

        idx = 0
        results = {}
        for var in self.labels:
            scores = metric(y_test[:, idx], y_pred[:, idx])
            if verbose:
                print(f"Score for var {var}: {scores}")
            results[var] = scores
            idx += 1
        return pd.DataFrame(results.items(), columns=["var", "score"])

    def plot_roc_auc(self, halt_x: np.ndarray, halt_y: np.ndarray):

        test_halt_preds = self.predict_halt_classifier(halt_x)
        halt_fpr, halt_tpr, _ = roc_curve(halt_y, test_halt_preds)
        roc_auc = auc(halt_fpr, halt_tpr)

        lw = 2
        plt.figure()
        plt.plot(
            halt_fpr,
            halt_tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (AUC = %0.2f)" % roc_auc,
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC for Recycle Predictions")
        plt.legend(loc="lower right")

    def group_per_episode(self, X, y=None, episode_ids=None):
        """groups the X, y data into independent episodes using episode_ids as reference,
        an array of same length than X/y with a unique id per independent episode
        """

        if not episode_ids.any():
            episode_ids = self.episode_ids

        assert (
            np.shape(X)[0] == np.shape(episode_ids)[0]
        ), f"X length ({len(X)}) is different than episode_ids ({len(episode_ids)}) length, when it should be the same"

        if len(episode_ids) < 1:
            return X, y

        # [TODO] validate if episode_ids are consecutive, and have required length
        #  (possibly best at loaders.py? to also check if iterations are consecutive)
        X_grouped = []
        y_grouped = []
        prev_ep_index = 0
        for i, ep_id in enumerate(episode_ids[1:]):
            if ep_id != episode_ids[i]:
                X_grouped.append(X[prev_ep_index : i + 1])
                if y is not None:
                    y_grouped.append(y[prev_ep_index : i + 1])
                prev_ep_index = i + 1

        # add last episode to array
        X_grouped.append(X[prev_ep_index:])
        if y is not None:
            y_grouped.append(y[prev_ep_index:])

        return X_grouped, y_grouped

    def sweep(
        self,
        params: Dict,
        X,
        y,
        search_algorithm: str = "bayesian",
        num_trials: int = 3,
        scoring_func: str = "r2",
        early_stopping: bool = False,
        results_csv_path: str = "outputs/results.csv",
        splitting_criteria: str = "CV",
        test_indices: Union[None, List[int]] = None,
        num_splits: int = 5,
    ) -> pd.DataFrame:

        if self.scale_data:
            X, y = self.scalar(X, y)

        if splitting_criteria.lower() == "cv":
            cv = None
        elif splitting_criteria.lower() == "timeseries":
            cv = TimeSeriesSplit(n_splits=num_splits)
        elif splitting_criteria.lower() == "grouped":
            cv = GroupShuffleSplit(n_splits=num_splits)
        elif splitting_criteria.lower() == "fixed":
            if type(test_indices) != list:
                raise ValueError("fixed split used but no test-indices provided...")
            cv = PredefinedSplit(test_fold=test_indices)
        else:
            raise ValueError(
                "Unknowing splitting criteria provided: {splitting_criteria}, should be one of [cv, timeseries, grouped]"
            )

        # early stopping only supported for learners that have a
        # `partial_fit` method
        from tune_sklearn import TuneSearchCV
        import mlflow
        import time

        mlflow.set_tracking_uri(os.path.join("file:/",os.getcwd(),"outputs"))


        # start mlflow auto-logging
        # mlflow.sklearn.autolog()

        if search_algorithm.lower() == "bohb":
            early_stopping = True

        if any(
            [search_algorithm.lower() in ["bohb", "bayesian", "hyperopt", "optuna"]]
        ):
            search = TuneSearchCV(
                self.model,
                params,
                search_optimization=search_algorithm,
                cv=cv,
                n_trials=num_trials,
                early_stopping=early_stopping,
                scoring=scoring_func,
                loggers=["csv", "tensorboard"],
                verbose=1
            )
        elif search_algorithm == "grid":
            search = GridSearchCV(
                self.model,
                param_grid=params,
                refit=True,
                cv=cv,
                scoring=scoring_func,
                verbose=1
            )
        elif search_algorithm == "random":
            search = RandomizedSearchCV(
                self.model,
                param_distributions=params,
                refit=True,
                cv=cv,
                scoring=scoring_func,
                verbose=1
            )
        else:
            raise NotImplementedError(
                "Search algorithm should be one of grid, hyperopt, bohb, optuna, bayesian, or random"
            )

        
        #with mlflow.start_run() as run:
        search.fit(X, y)
        self.model = search.best_estimator_
        results_df = pd.DataFrame(search.cv_results_)
        if not pathlib.Path(results_csv_path).parent.exists():
            pathlib.Path(results_csv_path).parent.mkdir(exist_ok=True, parents=True)
        final_path = results_csv_path[:-4]+"_"+time.strftime("%Y%m%d-%H%M%S")+".csv"
        logger.info(f"Saving sweeping results to {final_path}")
        results_df.to_csv(final_path)
        logger.info(f"Best hyperparams: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_}")

        return results_df


def plot_parallel_coords(results_df: pd.DataFrame):

    import plotly.express as px

    cols_keep = [col for col in results_df if "param_" in col]
    cols_keep += ["mean_test_score"]

    results_df = results_df[cols_keep]
    # want to convert object columns to type float
    results_df = results_df.apply(pd.to_numeric, errors="ignore", downcast="float")

    fig = px.parallel_coordinates(
        results_df,
        color="mean_test_score",
        labels=dict(
            zip(
                list(results_df.columns),
                list(["_".join(i.split("_")[1:]) for i in results_df.columns]),
            )
        ),
        color_continuous_scale=px.colors.diverging.Earth,
        # color_continuous_midpoint=27,
    )

    fig.show()


if __name__ == "__main__":

    base_model = BaseModel()
    x, y = base_model.load_csv(
        dataset_path="csv_data/cartpole-log.csv",
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
    )
