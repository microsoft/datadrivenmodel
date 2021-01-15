import abc
import logging
import os
import pathlib
import pickle
import sys
import numpy as np
import pandas as pd

from typing import Tuple, List, Union
from sklearn.preprocessing import StandardScaler
from loaders import CsvReader


# Add stdout handler, with level INFO
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
formater = logging.Formatter("%(name)-13s: %(levelname)-8s %(message)s")
console.setFormatter(formater)
logging.getLogger(__name__).addHandler(console)

# TODO: add weighting to the model
# TODO: this should go into a metrics function?
# TODO: implement saving of scalar transformers


class BaseModel(abc.ABC):
    def __init__(self, log_dirs: str = "logs"):

        self.logs_dir = log_dirs
        self.model = None

    def load_csv(
        self,
        dataset_path: str,
        input_cols: Union[str, List[str]] = "state",
        augm_cols: Union[str, List[str]] = ["action_command"],
        output_cols: Union[str, List[str]] = "state",
        iteration_order: int = -1,
        episode_col: str = "episode",
        iteration_col: str = "iteration",
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
            if type(input_cols) == str:
                base_features = [str(col) for col in df if col.startswith(input_cols)]
            elif type(input_cols) == list:
                base_features = input_cols
            else:
                raise TypeError(
                    f"input_cols expected type List[str] or str but received type {type(input_cols)}"
                )
            if type(augm_cols) == str:
                augm_features = [str(col) for col in df if col.startswith(augm_cols)]
            elif type(augm_cols) == list:
                augm_features = augm_cols
            else:
                raise TypeError(
                    f"augm_cols expected type List[str] or str but received type {type(augm_cols)}"
                )

            features = base_features + augm_features
            self.features = features

            if type(output_cols) == str:
                labels = [col for col in df if col.startswith(output_cols)]
            elif type(output_cols) == list:
                labels = output_cols
            else:
                raise TypeError(
                    f"output_cols expected type List[str] but received type {type(output_cols)}"
                )
            self.labels = labels

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

    def build_model(self, scale_data: bool = False):

        self.scale_data = scale_data

    def fit(self, X, y):

        if not self.model:
            raise ValueError("Please build or load the model first")

        if self.scale_data:
            X, y = self.scalar(X, y)
        self.model.fit(X, y)

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

    def load_model(self, filename: str, scale_data: bool = False):

        self.scale_data = scale_data
        self.model = pickle.load(open(filename, "rb"))

    def evaluate(self, test_data: np.ndarray):

        if not self.model:
            raise Exception("No model found, please run fit first")


if __name__ == "__main__":

    base_model = BaseModel()
    x, y = base_model.load_csv(
        dataset_path="csv_data/cartpole-log.csv",
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
    )
