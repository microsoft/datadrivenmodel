import abc
import logging
import os
import pathlib
import pickle
import sys
from typing import List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

from collections import OrderedDict

from loaders import CsvReader

matplotlib.rcParams["figure.figsize"] = [12, 10]

# Add stdout handler, with level INFO
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
formater = logging.Formatter("%(name)-13s: %(levelname)-8s %(message)s")
console.setFormatter(formater)
logging.getLogger("datamodeler").addHandler(console)

# TODO: add weighting to the model
# TODO: this should go into a metrics function?
# TODO: implement saving of scalar transformers


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
            elif type(input_cols) == list:
                base_features = input_cols
            else:
                raise TypeError(
                    f"input_cols expected type List[str] or str but received type {type(input_cols)}"
                )
            if not augm_cols:
                logging.debug(f"No augmented columns...")
            elif type(augm_cols) == str:
                augm_features = [str(col) for col in df if col.startswith(augm_cols)]
            elif type(augm_cols) == list:
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
            elif type(output_cols) == list:
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
            )
            X = df[csv_reader.feature_cols].values
            y = df[csv_reader.label_cols].values

        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        return X, y

    def load_numpy(self, dataset_path: str) -> Tuple:

        X = np.load(os.path.join(dataset_path, "x_set.npy"))
        y = np.load(os.path.join(dataset_path, "y_set.npy"))
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
            if self.scale_data:
                X = self.xscalar.transform(X)

            if label_col_names is None:
                label_col_names = self.labels
                if label_col_names is None:
                    # If None provided, and None stored in self.labels, we ask user to provide as input
                    # - Currently needed to match outputs to inputs when running the model forward -
                    raise ValueError("Please provide a list of predicted output labels ('label_col_names')")
            
            # prepare features & a list of predictions that are feats too (often all)
            feats = self.features
            preds_that_are_feats = [f_name for f_name in feats if f_name in label_col_names]
            # initialize feat_dict to first row & pred_dict to match first row too
            feat_dict = OrderedDict(list(zip(feats, X[0])))
            pred_dict = dict([(k,v) for (k,v) in feat_dict.items() if k in preds_that_are_feats])

            # sequentially iterate retriving next prediction based on previous prediction
            preds = []
            for i in range(len(X)):
                # extrac next row feats
                feat_dict = OrderedDict(list(zip(feats, list(X[i]))))
                # update feats with previous prediction
                for f_name in preds_that_are_feats:
                    feat_dict[f_name] = pred_dict[f_name]
                # get next prediction
                pred = self.predict(np.array([list(feat_dict.values())]))
                preds.append(pred[0])
                # update prediction dictionary (for next iteration)
                pred_dict = OrderedDict(list(zip(label_col_names, pred.tolist()[0])))

            preds = np.array(preds) #.transpose()

            if self.scale_data:
                preds = self.yscalar.inverse_transform(preds)

            #preds_df = pd.DataFrame(preds)
            #preds_df.columns = label_col_names

            return preds #preds_df

    
    def predict_sequentially(self, X, label_col_names: List[str] = None, it_per_episode: int = None):

        if not self.model:
            raise ValueError("Please build or load the model first")
        else:
            if self.scale_data:
                X = self.xscalar.transform(X)
            
            if label_col_names is None:
                label_col_names = self.labels
                if label_col_names is None:
                    # If None provided, and None stored in self.labels, we ask user to provide as input
                    # - Currently needed to match outputs to inputs when running the model forward -
                    raise ValueError("Please provide a list of predicted output labels ('label_col_names')")

            # initialize predictions
            preds = []

            if not it_per_episode:
                it_per_episode = np.shape(X)[0]

            num_of_episodes = int(np.shape(X)[0]/it_per_episode)

            # iterate per as many episodes as selected
            for i in range(num_of_episodes):

                X_aux = X[i*it_per_episode:(i+1)*it_per_episode]

                preds_aux = self.predict_sequentially_all(X_aux, label_col_names)
                preds.extend(preds_aux)

            preds = np.array(preds)

            #preds_df = pd.DataFrame(preds)
            #preds_df.columns = label_col_names

            return preds #preds_df

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
        self, X_test: np.ndarray, y_test: np.ndarray, metric, marginal: bool = False, it_per_episode = 100
    ):

        if not self.model:
            raise Exception("No model found, please run fit first")
        else:
            
            if not marginal:
                y_hat = self.predict_sequentially(X_test, it_per_episode = it_per_episode)
                y_hat_len = np.shape(y_hat)[0]
                y_test = y_test[:y_hat_len]
                return metric(y_test, y_hat)
            else:
                results_df = self.evaluate_margins_sequentially(X_test, y_test, metric, False, it_per_episode=it_per_episode)
                return results_df
                

    def evaluate_margins(
        self, X_test: np.ndarray, y_test: np.ndarray, metric, verbose: bool = False
    ):

        y_pred = self.predict(X_test)
        idx = 0
        results = {}
        for var in X_test.shape[1]:
            scores = metric(y_test[:, idx], y_pred[:, idx])
            if verbose:
                print(f"Score for var {var}: {scores}")
            results[var] = scores
            idx += 1
        return pd.DataFrame(results.items(), columns=["var", "score"])

    def evaluate_margins_sequentially(
        self, X_test: np.ndarray, y_test: np.ndarray, metric, verbose: bool = False, it_per_episode: int = 100
    ):

        # Extract prediction and remove any tail reminder from int(len(X_test)/it_per_episode)
        y_pred = self.predict_sequentially(X_test, it_per_episode = it_per_episode)
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


if __name__ == "__main__":

    base_model = BaseModel()
    x, y = base_model.load_csv(
        dataset_path="csv_data/cartpole-log.csv",
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
    )
