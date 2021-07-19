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

from dataclass import DataClass

logger = logging.getLogger(__name__)
matplotlib.rcParams["figure.figsize"] = [12, 10]


class BaseModel(abc.ABC):
    def __init__(self, log_dirs: str = "logs", model=None):

        self.logs_dir = log_dirs
        self.model = model
        self.halt_model = None
        self.dataclass_obj = DataClass()

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
        test_perc: float = 0.15,
        diff_state: bool = False,
        concatenated_steps: int = 1,
        concatenated_zero_padding: bool = True,
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
            in the order of the raw dataset, what is the lag between iteration t and iteration t+1, by default -1
        max_rows : Union[int, None], optional
            max rows to read for a large dataset, by default None
        diff_state : bool, default False
            If enabled, calculate differential between current output_cols and past output_cols
        concatenated_steps : int, optional
            number of steps to concatenate as input to ddm (per inference run)
        concatenated_zero_padding : bool, optional
            true: initial state padding made with zeroes
            false: initial state padding made copying initial sample 'concatenated_steps' times

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Features and labels for modeling


        Raises
        ------
        ValueError
            Data not found
        """

        X, y = self.dataclass_obj.load_csv(
            dataset_path=dataset_path,
            input_cols=input_cols,
            augm_cols=augm_cols,
            output_cols=output_cols,
            iteration_order=iteration_order,
            episode_col=episode_col,
            iteration_col=iteration_col,
            drop_nulls=drop_nulls,
            max_rows=max_rows,
            test_perc=test_perc,
            diff_state=diff_state,
            concatenated_steps=concatenated_steps,
            concatenated_zero_padding=concatenated_zero_padding,
        )

        # Transferring key features in between classes for easier access
        self.features = self.dataclass_obj.features
        self.labels = self.dataclass_obj.labels

        self.feature_cols = self.dataclass_obj.feature_cols
        self.label_cols = self.dataclass_obj.label_cols

        self.input_dim = self.dataclass_obj.input_dim
        self.output_dim = self.dataclass_obj.output_dim

        if hasattr(self, "original_features"):
            self.original_features = self.dataclass_obj.original_features
        if hasattr(self, "original_labels"):
            self.original_labels = self.dataclass_obj.original_labels

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

    def predict_sequentially_all(
        self,
        X: Union[None, np.ndarray] = None,
    ):
        """Make predictions sequentially for provided iterations. All iterations are run sequentially until the end.

        Parameters
        ----------
        X : np.ndarray
            Array of input values to the DDM for sequential prediction

        Returns
        -------
        np.ndarray
            Array of predicted values


        Raises
        ------
        ValueError
            Model or Data not found
        """

        if not self.model:
            raise ValueError("Please build or load the model first")
        else:

            if X is None:
                X, y_test = self.get_test_set(grouped_per_episode=False)

            assert (
                len(X) > 0
            ), "Predict sequentially requires at least one iteration provided, but none where given."

            # scaling is performed within "predict" (self.scale_data)

            # print(f"X (predict_sequentially_all) ---> {np.shape(X)}")
            self.dataclass_obj.sequential_inference_initialize(ini_X=X[0])

            # sequentially iterate retriving next prediction based on previous prediction
            next_X = X[0]
            preds_array = []
            for i in range(len(X)):
                # get next prediction
                preds = self.predict(np.array(next_X))
                preds_array.append(preds[0])
                # update prediction dictionary (for next iteration)
                next_X = self.dataclass_obj.sequential_inference(new_y=preds[0])

            preds_array = np.array(preds_array)  # .transpose()

            # preds_df = pd.DataFrame(preds)
            # preds_df.columns = label_col_names

            return preds_array  # preds_df

    def predict_sequentially(
        self,
        X_grouped: Union[None, List[np.ndarray]] = None,
        y_grouped: Union[None, List[np.ndarray]] = None,
        it_per_episode: Union[None, int] = None,
    ):
        """Make predictions sequentially for provided episodes. Each episode is compound of the iterations to be run sequentially.

        Parameters
        ----------
        X_test_grouped: List[np.ndarray]
            List of numpy arrays with input DDM features to be used for sequential prediction. Iterations grouped per episode.
            - If None is given, default test sets are used (generated when calling load_csv method).
        y_test_grouped: List[np.ndarray]
            List of numpy arrays with output labels to be used to assess sequential prediction. Iterations grouped per episode.
            - If None is given, default test sets are used (generated when calling load_csv method).
            - Note, we provide the labels here to ensure we can keep track of any skipped iterations.
        it_per_episode: int
            Number os iterations to cut episodes on. Disregarded if it_per_episode > len(X_grouped[i]).


        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Array of predicted values and provided labels.


        Raises
        ------
        ValueError
            Model not instanced.
        Assert
            X_grouped is empty.
            y_grouped has different episode length than X_grouped.
        """

        if not self.model:
            raise ValueError("Please build or load the model first")
        else:
            # note, scaling is performed in inner function "predict_sequentially_all" > "predict" (self.scale_data)

            if X_grouped is None:
                X_grouped, y_grouped = self.get_test_set(grouped_per_episode=True)

            # initialize predictions
            preds = []
            labels = []

            print(f"X (predict_sequentially) ---> {np.shape(X_grouped)}")

            if not it_per_episode:
                # when grouped, X has shape {episode_count, iteration_count, feature_count}
                it_per_episode = np.inf

            num_of_episodes = int(np.shape(X_grouped)[0])

            # iterate per as many episodes as selected
            for i in range(num_of_episodes):

                n_iterations = len(X_grouped[i])
                if it_per_episode >= n_iterations:
                    preds_aux = self.predict_sequentially_all(X_grouped[i])
                    if y_grouped is not None:
                        labels_aux = y_grouped[i]

                else:
                    # split episodes into subepisodes when it_per_episode < episode length
                    n_subepisodes = int(np.ceil(n_iterations / it_per_episode))
                    preds_aux_array = []
                    labels_aux_array = []
                    for j in range(n_subepisodes):
                        preds_aux = self.predict_sequentially_all(
                            X_grouped[i][j * it_per_episode : (j + 1) * it_per_episode],
                        )
                        preds_aux_array.append(copy.deepcopy(preds_aux))

                        if y_grouped is not None:
                            labels_aux = y_grouped[i][
                                j * it_per_episode : (j + 1) * it_per_episode
                            ]
                            labels_aux_array.append(copy.deepcopy(labels_aux))

                    preds_aux = np.concatenate(preds_aux_array, axis=0)
                    if y_grouped is not None:
                        labels_aux = np.concatenate(labels_aux_array, axis=0)

                # append to predictions before getting into next episode
                preds.extend(copy.deepcopy(preds_aux))
                if y_grouped is not None:
                    labels.extend(copy.deepcopy(labels_aux))

            preds = np.array(preds)
            labels = np.array(labels)

            # preds_df = pd.DataFrame(preds)
            # preds_df.columns = label_col_names

            return preds, labels  # preds_df

    def predict_halt_classifier(self, X: np.ndarray):

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
        self,
        metric,
        X_test: Union[None, np.ndarray] = None,
        y_test: Union[None, np.ndarray] = None,
        marginal: bool = False,
    ):

        if not self.model:
            raise Exception("No model found, please run fit first")
        else:

            if X_test is None:
                X_test, y_test = self.get_test_set(grouped_per_episode=False)

            assert (
                len(X_test) > 0
            ), "Predict sequentially requires at least one iteration provided, but none where given."

            if y_test is not None:
                assert len(X_test) == len(
                    y_test
                ), f"number of iterations for predictions ({len(X_test)}) and labels ({len(y_test)}) do not match."

            if not marginal:
                y_hat = self.predict(X_test)
                return metric(y_test, y_hat)
            else:
                results_df = self.evaluate_margins(metric, X_test, y_test, False)
                return results_df

    def evaluate_sequentially(
        self,
        metric,
        X_grouped: Union[None, List[np.ndarray]] = None,
        y_grouped: Union[None, List[np.ndarray]] = None,
        marginal: bool = False,
        it_per_episode=100,
    ):
        """Evaluate sequential prediction for provided episodes.

        Parameters
        ----------
        metric: function
            Retrieves the desired metric to be used to compare and assess difference between predictions and test labels.
        X_grouped: List[np.ndarray]
            List of numpy arrays with input DDM features to be used for sequential prediction and evaluation. Iterations grouped per episode.
            - If None is given, default test sets are used (generated when calling load_csv method).
        y_grouped: List[np.ndarray]
            List of numpy arrays with output labels to be used to assess sequential prediction and evaluation. Iterations grouped per episode.
            - If None is given, default test sets are used (generated when calling load_csv method).
            - Note, we provide the y_test here to ensure we can keep track of any skipped iterations.
        marginal: bool
            Retrieve per var computed error honoring "metric" function.
        it_per_episode: int
            Number os iterations to subdivide episodes on. Disregarded if it_per_episode > len(X_test_grouped[i]).


        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Array of predicted values and provided labels.


        Raises
        ------
        ValueError
            Model or Data not found.
        """

        if not self.model:
            raise Exception("No model found, please run fit first")
        else:

            if X_grouped is None:
                X_grouped, y_grouped = self.get_test_set(grouped_per_episode=True)

            if not marginal:
                y_hat, y_test = self.predict_sequentially(
                    X_grouped, y_grouped, it_per_episode=it_per_episode
                )
                return metric(y_test, y_hat)
            else:
                results_df = self.evaluate_margins_sequentially(
                    metric, X_grouped, y_grouped, False, it_per_episode=it_per_episode
                )
                return results_df

    def evaluate_margins(
        self,
        metric,
        X_test: Union[None, np.ndarray] = None,
        y_test: Union[None, np.ndarray] = None,
        verbose: bool = False,
    ):
        """Evaluate predictions for each var separately.

        Parameters
        ----------
        metric: function
            Retrieves the desired metric to be used to compare and assess difference between predictions and test labels.
        X_test: np.ndarray
            Numpy arrays with input DDM features to be used for per-iteration prediction and evaluation.
            - If None is given, default test sets are used (generated when calling load_csv method).
        y_test: np.ndarray
            Numpy arrays with output labels to be used to assess per-iteration prediction and evaluation.
            - If None is given, default test sets are used (generated when calling load_csv method).


        Returns
        -------
        Pandas Dataframe
            Dataframe with scores per label


        Raises
        ------
        ValueError
            Model or Data not found.
        """

        if not self.model:
            raise Exception("No model found, please run fit first")
        else:

            if X_test is None:
                X_test, y_test = self.get_test_set(grouped_per_episode=False)

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
        metric,
        X_grouped: Union[None, np.ndarray] = None,
        y_grouped: Union[None, np.ndarray] = None,
        verbose: bool = False,
        it_per_episode: int = 100,
    ):
        """Evaluate sequential prediction for provided episodes. Splitting prediction results per label variable.

        Parameters
        ----------
        metric: function
            Retrieves the desired metric to be used to compare and assess difference between predictions and test labels.
        X_grouped: List[np.ndarray]
            List of numpy arrays with input DDM features to be used for sequential prediction and evaluation. Iterations grouped per episode.
            - If None is given, default test sets are used (generated when calling load_csv method).
        y_grouped: List[np.ndarray]
            List of numpy arrays with output labels to be used to assess sequential prediction and evaluation. Iterations grouped per episode.
            - If None is given, default test sets are used (generated when calling load_csv method).
            - Note, we provide the y_test here to ensure we can keep track of any skipped iterations.
        marginal: bool
            Retrieve per var computed error honoring "metric" function.
        it_per_episode: int
            Number os iterations to subdivide episodes on. Disregarded if it_per_episode > len(X_test_grouped[i]).


        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Array of predicted values and provided labels.


        Raises
        ------
        ValueError
            Model or Data not found.
        """

        if X_grouped is None:
            X_grouped, y_grouped = self.get_test_set(grouped_per_episode=True)

        # Extract prediction and remove any tail reminder from int(len(X_test)/it_per_episode)
        y_pred, y_test = self.predict_sequentially(
            X_grouped, y_grouped, it_per_episode=it_per_episode
        )

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

    def get_test_set(self, grouped_per_episode=False):
        """Extracts test sets from dataclass_obj"""

        if grouped_per_episode:
            (
                X_test_grouped,
                y_test_grouped,
            ) = self.dataclass_obj.get_test_set_per_episode()

            assert (
                len(X_test_grouped) > 0
            ), "Predict sequentially requires at least one episode provided, but none where given."

            assert len(X_test_grouped) == len(
                y_test_grouped
            ), f"number of episodes for predictions ({len(X_test_grouped)}) and labels ({len(y_test_grouped)}) do not match."

            return X_test_grouped, y_test_grouped

        else:
            X_test, y_test = self.dataclass_obj.get_test_set()

            assert (
                len(X_test) > 0
            ), "At least one iteration must be provided, but none where extracted."

            assert len(X_test) == len(
                y_test
            ), f"number of iterations for predictions ({len(X_test)}) and labels ({len(y_test)}) do not match."

            return X_test, y_test

    def get_train_set(self, grouped_per_episode=False):
        """Extracts training sets from dataclass_obj"""

        if grouped_per_episode:
            (
                X_train_grouped,
                y_train_grouped,
            ) = self.dataclass_obj.get_train_set_per_episode()

            assert (
                len(X_train_grouped) > 0
            ), "Predict sequentially requires at least one episode provided, but none where given."

            assert len(X_train_grouped) == len(
                y_train_grouped
            ), f"number of episodes for predictions ({len(X_train_grouped)}) and labels ({len(y_train_grouped)}) do not match."

            return X_train_grouped, y_train_grouped

        else:
            X_train, y_train = self.dataclass_obj.get_train_set()

            assert (
                len(X_train) > 0
            ), "At least one iteration must be provided, but none where extracted."

            assert len(X_train) == len(
                y_train
            ), f"number of iterations for predictions ({len(X_train)}) and labels ({len(y_train)}) do not match."

            return X_train, y_train

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

        # start mlflow auto-logging
        mlflow.sklearn.autolog()

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
            )
        elif search_algorithm == "grid":
            search = GridSearchCV(
                self.model,
                param_grid=params,
                refit=True,
                cv=cv,
                scoring=scoring_func,
            )
        elif search_algorithm == "random":
            search = RandomizedSearchCV(
                self.model,
                param_distributions=params,
                refit=True,
                cv=cv,
                scoring=scoring_func,
            )
        else:
            raise NotImplementedError(
                "Search algorithm should be one of grid, hyperopt, bohb, optuna, bayesian, or random"
            )

        with mlflow.start_run() as run:
            search.fit(X, y)
        self.model = search.best_estimator_
        results_df = pd.DataFrame(search.cv_results_)
        if not pathlib.Path(results_csv_path).parent.exists():
            pathlib.Path(results_csv_path).parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving sweeping results to {results_csv_path}")
        results_df.to_csv(results_csv_path)
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
