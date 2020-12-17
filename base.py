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


class BaseModel(abc.ABC):
    def __init__(self, log_dirs: str = "logs"):

        self.logs_dir = log_dirs
        self.model = None

    def load_csv(
        self,
        dataset_path: str,
        input_cols_read: Union[str, List[str]] = "state",
        augm_cols: Union[str, List[str]] = ["action_command"],
        output_col: Union[str, List[str]] = "state",
        timelag: int = -1,
        max_rows: Union[int, None] = None,
    ) -> Tuple[np.array, np.array]:
        """Read CSV data into two datasets for modeling

        Parameters
        ----------
        dataset_path : str
            path to csv dataset
        input_cols_read : Union[str, List[str]], optional
            list of columns represent the inputs to the dynamical system in the raw dataset. Can either be a string which is then matched for all columns in the dataset, or a list of strings with exact matches, by default "state"
        augm_cols : Union[str, List[str]], optional
            Exact match of additional columns to use for modeling, such as the actions of the current iteration and any scenario/config parameters, by default ["action_command"]
        output_col : Union[str, List[str]], optional
            output columns of the dynamical system. Can either be a string which is then matched for any columns or a list of exact matches, by default "state"
        timelag : int, optional
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
            df = csv_reader.read(
                dataset_path,
                timelag=timelag,
                feature_cols=input_cols_read,
                max_rows=max_rows,
            )
            features = csv_reader.feature_cols + augm_cols
            output_cols = [col for col in df if col.startswith(output_col)]
            X = df[features].values
            y = df[output_cols].values

        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        return X, y

    def load_numpy(self, dataset_path: str) -> Tuple:

        X = np.load(os.path.join(dataset_path, "x_set.npy"))
        y = np.load(os.path.join(dataset_path, "y_set.npy"))
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        return X, y

    def scalar(self, X, y):

        self.xscalar = StandardScaler()
        self.yscalar = StandardScaler()

        X_scaled = self.xscalar.fit_transform(X)
        y_scaled = self.yscalar.fit_transform(y)

        return X_scaled, y_scaled

    def build_model(self, scale_data: bool = False):

        self.scale_data = scale_data
        raise NotImplementedError

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

        if not pathlib.Path(filename).parent.exists():
            pathlib.Path(filename).parent.mkdir(parents=True)
        pickle.dump(self.model, open(filename, "wb"))

    def load_model(self, filename: str, scale_data: bool = False):

        self.scale_data = scale_data
        self.model = pickle.load(open(filename, "rb"))

    def evaluate(self, test_data: np.array):

        if not self.model:
            raise Exception("No model found, please run fit first")


if __name__ == "__main__":

    base_model = BaseModel()
    base_model.load_csv(
        dataset_path="csv_data/cartpole-log.csv",
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
    )
