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
                results_df = self.evaluate_margins()
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
        


    def get_feat_importance(self):

        if not self.model:
            raise Exception("No model found, please run fit first")

        feats = self.features
        outputs = self.labels

        if self.model_type == "SVR":
            # Note, .coef_ is only available for linear models
            raise ValueError("Feat importance hasn't been configured for 'SVR'")

        if self.model_type in ["linear_model", "SVR"] and not self.scale_data:
            # Note, coefficients do NOT match feature importance for NON scaled data (self.scale_data == False)
            raise ValueError("Feat importance hasn't been configured for non-scaled data in 'linear_model'")

        
        feat_importance_d = dict()

        if not self.separate_models:

            if self.model_type in ["GradientBoostingRegressor", "xgboost", "lightgbm"]:
                for i,(feat_name,estimator) in enumerate(zip(outputs, self.model.estimators_)):
                    feat_importance_d[feat_name] = estimator.feature_importances_
            
            elif self.model_type in ["linear_model"]:
                print("self.model.coef_", self.model.coef_)
                for i,(feat_name,coefficients) in enumerate(zip(outputs, self.model.coef_)):
                    feat_importance_d[feat_name] = coefficients
            
            else:
                raise ValueError("Unknown model type")

        else:
            
            if self.model_type in ["GradientBoostingRegressor", "xgboost", "lightgbm"]:
                for i,feat_name in enumerate(outputs):
                    feat_importance_d[feat_name] = self.models[i].feature_importances_
            elif self.model_type in ["linear_model"]:
                for i,feat_name in enumerate(outputs):
                    feat_importance_d[feat_name] = self.models[i].coef_
            else:
                raise ValueError("Unknown model type")


        return feat_importance_d

    

    def plot_feature_importance(self,
                                feature_data = None,
                                in_feats_dim = None,
                                out_labels = None,
                                total_width = 1.0):

        if not self.model:
            raise Exception("No model found, please run fit first")
        
        if feature_data is None:
            feature_data = self.get_feat_importance()
        if in_feats_dim is None:
            in_feats_dim = self.input_dim
        if out_labels is None:
            out_labels = self.features
        
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  
        n_bars = len(feature_data) + 1  # plus 1 to add a space in between predictors
        bar_width = total_width / n_bars # width of single bar
        bars = []
        for i, (name, values) in enumerate(feature_data.items()):
            x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
            for x, y in enumerate(values):
                bar = ax.bar(x + x_offset, y, width=bar_width , color=colors[i % len(colors)])
            bars.append(bar[0])


        ax.legend(bars, feature_data.keys())
        plt.title('Feature Importance / Feature Multipliers', fontsize=18)
        plt.xlabel('Feature Column Names', fontsize=18)
        plt.ylabel('Feature Importance', fontsize=18)

        plt.xticks(ticks=range(in_feats_dim), labels=out_labels)
        ax.tick_params(labelrotation=90)
        plt.show()


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
