import os
import pandas as pd
from typing import List, Tuple, Union
import logging
import numpy as np
import copy as copy
import random
from collections import OrderedDict
from typing import Dict, List, Tuple, Union
from omegaconf.listconfig import ListConfig

logger = logging.getLogger("data_class")
logger.setLevel(logging.INFO)


class DataClass(object):
    def __init__(self):
        # Required fields that need to be filled every time we call load_csv as well.

        # feature df original col naming, saved when renaming is performed --> concatenation
        self.original_features = None
        # label df original col naming, saved when renaming is performed --> diff
        self.original_labels = None

        # training variables - df
        self.train_df_array = None
        # training variables - numpy.array
        self._X = None
        self._y = None
        # training variables - List[numpy.array]
        self._X_train_seq = None
        self._y_train_seq = None

        # testing variables - df
        self.test_df_array = None
        # testing variables - numpy.array
        self._X_test = None
        self._y_test = None
        # testing variables - List[numpy.array]
        self._X_test_seq = None
        self._y_test_seq = None

        # sequential inference
        self.last_X_d = None
        self.new_y_d = None

    def split(
        self,
        df,
        iteration_col,
        episode_col,
        iteration_order,
        lagger_str,
        current_row,
        feature_cols,
        label_cols,
        augmented_cols,
    ):
        """Split the dataset by features and labels

        Parameters
        ----------
        df : [type]
            [description]
        iteration_col : [type]
            [description]
        episode_col : [type]
            [description]
        iteration_order : [type]
            [description]
        lagger_str : [type]
            [description]
        current_row : [type]
            [description]
        feature_cols : [type]
            [description]
        label_cols : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        logger.info(
            f"Iteration order set to {iteration_order} so using {current_row} from {lagger_str} {iteration_order} row"
        )

        # We group by episode and iteration indices to make dataset episodic
        df = df.sort_values(by=[episode_col, iteration_col])
        # Create a lagged dataframe for capturing inputs and outputs
        # when iteration_order < 0, this will consist of the features
        # since we are doing a shift-backwards
        # when iteration_order > 0, this will consist of labels
        # since we are doing a shift-forward
        lagged_df = df.groupby(by=episode_col, as_index=False).shift(
            iteration_order * -1
        )
        lagged_df = lagged_df.drop([iteration_col], axis=1)

        # if iteration order is less than 1
        # then the actions, configs should not be lagged
        # only states should be lagged
        # features = lagged_df[states] + df[actions, configs]
        # labels = df[states]
        if iteration_order < 0:
            features_df = lagged_df[feature_cols]
            features_df[augmented_cols] = df[augmented_cols]
        # if iteration order is greater than 1
        # then features = states, actions, configs from current row (df)
        # labels = states from next row (lagged_df)
        else:
            features_df = df[feature_cols]
            # TODO: check, is this always redundant?
            # i.e., is feature_cols is supset of augmented_cols
            features_df[augmented_cols] = df[augmented_cols]

        # eventually we will join the labels_df with the features_df
        # if any columns are matching then rename them
        if bool(set(feature_cols) & set(label_cols)):
            features_df = features_df.rename(
                columns=lambda x: "prev_" + x if x in label_cols else x
            )

        self.feature_cols = list(features_df.columns.values)
        self.label_cols = list(label_cols)
        logger.info(f"Feature columns are: {self.feature_cols}")
        logger.info(f"Label columns are: {self.label_cols}")
        # joined_df = df.join(features_df)
        vars_to_keep = (
            [episode_col, iteration_col] + self.feature_cols + self.label_cols
        )
        if iteration_order < 0:
            labels_df = df[[episode_col, iteration_col] + label_cols]
        else:
            labels_df = df[[episode_col, iteration_col]].join(lagged_df[label_cols])
        return labels_df.join(features_df)[vars_to_keep]

    def read(
        self,
        df: pd.DataFrame,
        iteration_order: int = -1,
        episode_col: str = "episode",
        iteration_col: str = "iteration",
        feature_cols: List[str] = ["state_x_position"],
        label_cols: List[str] = ["state_x_position"],
        augmented_cols: List[str] = ["action_command"],
    ):
        """Read episodic data where each row contains either inputs and its preceding output output or the causal inputs/outputs relationship

        Parameters
        ----------
        df : pdf.DataFrame
            [description]
        iteration_order : int, optional
            [description], by default -1
        episode_col : str, optional
            [description], by default "episode"
        iteration_col : str, optional
            [description], by default "iteration"
        feature_cols : Union[List, str], optional
            [description], by default "state_"

        Returns
        -------
        [type]
            [description]
        """

        # CASE 1: rows are of the form {st+1, at}
        # Append st into next row
        # if iteration_order < 0 then drop the iteration - iteration_order iteration from each episode
        # and append previous state columns into each row: {st+1, at} -> {st, at, st+1}
        if all([episode_col, iteration_col, iteration_order < 0]):
            lagger_str = "previous"
            current_row = "inputs"

            joined_df = self.split(
                df,
                iteration_col,
                episode_col,
                iteration_order,
                lagger_str,
                current_row,
                feature_cols,
                label_cols,
                augmented_cols,
            )

            # skip the first row of each episode since we do not have its st
            joined_df = (
                joined_df.groupby(by=episode_col, as_index=False)
                .apply(lambda x: x.iloc[iteration_order * -1 :])
                .reset_index()
            )
            return joined_df.drop(["level_0", "level_1"], axis=1)

        # CASE 2: rows of the form {st, at}
        # Append st+1 from next row into current row {st, at, st+1}
        elif all([episode_col, iteration_col, iteration_order > 0]):
            lagger_str = "next"
            current_row = "outputs"

            joined_df = self.split(
                df,
                iteration_col,
                episode_col,
                iteration_order,
                lagger_str,
                current_row,
                feature_cols,
                label_cols,
                augmented_cols,
            )
            # truncate before the end of iteration_order for complete observations only
            joined_df = (
                joined_df.groupby(by=episode_col, as_index=False)
                .apply(lambda x: x.iloc[: iteration_order * -1])
                .reset_index()
            )
            return joined_df.drop(["level_0", "level_1"], axis=1)
        else:
            return df

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

        # define require fields
        self.__init__()

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

            self.iteration_order = iteration_order

            df = self.read(
                df,
                iteration_order=iteration_order,
                feature_cols=features,
                label_cols=labels,
                episode_col=episode_col,
                iteration_col=iteration_col,
                augmented_cols=augm_features,
            )

            # store episode_id to group_per_episode
            self.episode_col = episode_col
            self.iteration_col = iteration_col
            self.episode_ids = df[episode_col].values

            # Trim episodes
            df_per_episode = self.trim_episodes(df)

            # Diff generation
            self.diff_state = diff_state
            if self.diff_state:
                self.df_per_episode = []
                for df in df_per_episode:
                    aux_df = self.df_diff_predictions(df)
                    if aux_df is not None:
                        self.df_per_episode.append(aux_df)

            # Input-state concatenation
            self.concatenated_steps = concatenated_steps
            self.concatenated_zero_padding = concatenated_zero_padding
            if self.concatenated_steps > 1:
                df_per_episode = copy.deepcopy(self.df_per_episode)
                self.df_per_episode = []
                for df in df_per_episode:
                    aux_df = self.df_concatenate_inputs(df)
                    if aux_df is not None:
                        self.df_per_episode.append(aux_df)

            # Splitting datasets
            self.split_train_and_test_samples(test_perc=test_perc)

            return self.get_train_set()

        return None, None

    def trim_episodes(self, df):
        """Split the array into episodes using iteration/episode provided in dataframe.

        Parameters
        ----------
        df: pd.Dataframe
            Takes the set of dataframes and splits into a list of DFs per episode.


        Returns
        -------
        List[pd.Dataframe]
            List of dataframes, grouped by episode.

        """

        # We group by episode and iteration indices to make dataset episodic
        df = df.sort_values(by=[self.episode_col, self.iteration_col])
        # Create a lagged dataframe for capturing inputs and outputs
        df_per_episode = df.groupby(by=self.episode_col, as_index=False)
        # Remove indexer, and get only df list
        df_per_episode = list(map(lambda x: x[1], df_per_episode))

        self.df_per_episode = df_per_episode

        logger.info(
            f"trimmed dataframe across episodes. found ({len(self.df_per_episode)}) episodes."
        )

        return df_per_episode

    def split_train_and_test_samples(self, test_perc=0.15):
        """Takes care of splitting test and train sets. The dataset is split without breaking episodes, but making the split at an iteration level.

        Parameters
        ----------
        split: float [0, 1]
            Percentage of samples to keep for training.

        """

        # Get train split - but ensuring we do not divide by zero later
        split = min(max(1 - test_perc, 0.01), 0.99)

        episodes_len = []

        self.test_len = 0
        self.train_len = 0
        self.test_df_array = []
        self.train_df_array = []
        for df in self.df_per_episode:
            ep_len = len(df)
            episodes_len.append(ep_len)

            if self.train_len / split <= self.test_len / (1 - split):
                self.train_len += ep_len
                self.train_df_array.append(df)
            else:
                self.test_len += ep_len
                self.test_df_array.append(df)

        # shuffle training and testing data
        random.shuffle(self.train_df_array)
        random.shuffle(self.test_df_array)

        # Extract episode length (mean & std dev)
        self.mean_episode_len = np.mean(episodes_len)
        self.std_episode_len = np.std(episodes_len)
        logger.info(
            f"divided train & test set with ({self.train_len}) and ({self.test_len}) iterations, respectively. Chosen split == {split*100}%.\
            \n   >> Average episode length: ({self.mean_episode_len}). Average std dev: ({self.std_episode_len})"
        )

        return

    @property
    def X(self):
        if self._X is None:
            self.get_train_set()
        # return value extracted on previous run
        return self._X

    @property
    def y(self):
        if self._y is None:
            self.get_train_set()
        # return value extracted on previous run
        return self._y

    def get_train_set(self):
        # Prepares X and y training dataset, and retrieves after aggregation
        if self._X is None or self._y is None:
            self._X = []
            self._y = []
            for df in self.train_df_array:
                self._X.extend(df[self.feature_cols].values)
                self._y.extend(df[self.label_cols].values)

            self._X = np.array(self._X)
            self._y = np.array(self._y)

            self.input_dim = self._X.shape[1]
            self.output_dim = self._y.shape[1]

        return self._X, self._y

    def get_train_set_per_episode(self):
        # Prepares X and y training dataset, and retrieves after aggregation
        if not self._X_train_seq or not self._y_train_seq:
            self._X_train_seq = []
            self._y_train_seq = []
            for df in self.train_df_array:
                X_episode = np.array(df[self.feature_cols].values)
                y_episode = np.array(df[self.label_cols].values)
                self._X_train_seq.append(X_episode)
                self._y_train_seq.append(y_episode)

            assert (
                self.input_dim == self._X_train_seq[0].shape[1]
            ), "input dimension has changed between train ({self.input_dim}) and current train grouped-per-episode set ({self._X_train_seq[0].shape[1]})."
            assert (
                self.output_dim == self._y_train_seq[0].shape[1]
            ), "output dimension has changed between train ({self.output_dim}) and current train grouped-per-episode set ({self._y_train_seq[0].shape[1]})."

        return self._X_train_seq, self._y_train_seq

    @property
    def X_test(self):
        if self._X_test is None:
            self.get_test_set()
        # return value extracted on previous run
        return self._X_test

    @property
    def y_test(self):
        if self._y_test is None:
            self.get_test_set()
        # return value extracted on previous run
        return self._y_test

    def get_test_set(self):
        # Prepares X and y training dataset, and retrieves after aggregation
        if self._X_test is None or self._y_test is None:
            self._X_test = []
            self._y_test = []
            for df in self.test_df_array:
                self._X_test.extend(df[self.feature_cols].values)
                self._y_test.extend(df[self.label_cols].values)

            self._X_test = np.array(self._X_test)
            self._y_test = np.array(self._y_test)

            assert (
                self.input_dim == self._X_test.shape[1]
            ), "input dimension has changed between train ({self.input_dim}) and current test set ({self._X_test.shape[1]})."
            assert (
                self.output_dim == self._y_test.shape[1]
            ), "output dimension has changed between train ({self.output_dim}) and current test set ({self._y_test.shape[1]})."

        return self._X_test, self._y_test

    def get_test_set_per_episode(self):
        # Prepares X and y training dataset, and retrieves after aggregation
        if not self._X_test_seq or not self._y_test_seq:
            self._X_test_seq = []
            self._y_test_seq = []
            for df in self.test_df_array:
                X_episode = np.array(df[self.feature_cols].values)
                y_episode = np.array(df[self.label_cols].values)
                self._X_test_seq.append(X_episode)
                self._y_test_seq.append(y_episode)

            assert (
                self.input_dim == self._X_test_seq[0].shape[1]
            ), "input dimension has changed between train ({self.input_dim}) and current test set ({self._X_test_seq[0].shape[1]})."
            assert (
                self.output_dim == self._y_test_seq[0].shape[1]
            ), "output dimension has changed between train ({self.output_dim}) and current test set ({self._y_test_seq[0].shape[1]})."

        return self._X_test_seq, self._y_test_seq

    def sequential_inference_initialize(self, ini_X: np.ndarray):
        """Takes care of initializing the features to the model for sequential prediction.

        Parameters
        ----------
        ini_X: np.ndarray
            Set of initial features to store for subsequent updates.

        """

        self.last_X_d = OrderedDict(zip(self.feature_cols, list(ini_X)))
        return None

    def sequential_inference(self, new_y: np.ndarray):
        """Takes care of processing the predicted outputs, and insert them on top of the previous step for sequential prediction.
        At the moment we keep the input features static in between runs, only overwritting the labels that the model predicts sequentially.
        - Note, "sequential_inference_initialize" needs to be called first.

        Parameters
        ----------
        new_y: np.ndarray
            Predictions made by DDM, to be used to overwrite initial X.
            .

        Returns
        -------
        np.ndarray
            Array of next features, when receiving new_y for feature update.


        Raises
        ------
        Exception
            When method "sequential_inference_initialize" has not been called prior to stepping.
        """

        if self.last_X_d is None:
            raise Exception(
                "Method 'sequential_inference_initialize' must be called prior to sequential prediction."
            )

        assert len(self.label_cols) == len(
            new_y
        ), "new_y should have same length than labels provided during load_csv method."

        self.new_y_d = OrderedDict(zip(self.label_cols, list(new_y)))

        # if new_state is not None:

        if self.concatenated_steps > 1:
            feats_list = self.original_features
        else:
            feats_list = self.feature_cols

        for feat in feats_list:

            if self.concatenated_steps > 1:
                for i in range(1, self.concatenated_steps):
                    concat_feat = feat + f"_{i}"
                    next_concat_feat = feat + f"_{i+1}"
                    self.last_X_d[next_concat_feat] = self.last_X_d[concat_feat]

                target_feat = feat + "_1"
            else:
                target_feat = feat

            target_label = None
            for label in self.label_cols:
                if self.iteration_order > 0:
                    if label in feat:
                        target_label = label
                        break
                elif self.iteration_order < 0:
                    if label[5:] in feat:
                        target_label = label
                        break
                else:
                    raise Exception(
                        "iteration_order == 0 has not been configured for sequential inference."
                    )

            if target_label is None:
                continue

            if self.diff_state:
                target_label = "diff_" + target_label
                self.last_X_d[target_feat] += self.new_y_d[target_label]
            else:
                self.last_X_d[target_feat] = self.new_y_d[target_label]

        return np.array(list(self.last_X_d.values()))

    def df_diff_predictions(self, df):
        """Take the dataframe and modify labels to be differential states.

        Parameters
        ----------
        df: pd.Dataframe
            Dataframe with labels and features.
            .

        Returns
        -------
        pd.Dataframe
            Dataframe with diff labels added on new column.

        """

        labels_matched_to_feats = True

        if not self.original_labels:
            self.original_labels = copy.deepcopy(self.label_cols)

        if len(self.original_labels) > len(self.label_cols):
            # Re-start defining new cols if we previously skipped a df without copying all "diff" vars
            self.label_cols = []

        for label in self.original_labels:

            diff_label = "diff_" + label
            diff_values = None

            if len(self.original_labels) > len(self.label_cols):
                self.label_cols.append(diff_label)

            # Iterate to find match for 'label' within feature columns
            for feat in self.feature_cols:

                if self.iteration_order > 0:
                    if label in feat:
                        diff_values = df[label].values - df[feat].values
                        break
                if self.iteration_order < 0:
                    if label[5:] in feat:
                        diff_values = df[label].values - df[feat].values
                        break

            if diff_values is None:

                if len(df) < 2:
                    logger.warn(
                        "not enough rows to provide diff on (minimum 2), or at least a matching feature column. df is skipped"
                    )
                    return None

                labels_matched_to_feats = False
                diff_values = df[label].values[1:] - df[label].values[:-1]
                diff_values = np.append([0], diff_values)

            df[diff_label] = diff_values

        if labels_matched_to_feats:
            logger.debug(
                "delta states enabled, calculating differential between input and output values. note, no rows have been lost."
            )
        else:
            # drop last zeroed row
            df.drop(df.head(1).index, axis=0, inplace=True)
            logger.debug(
                "delta states enabled, calculating differential between input and output values. note, first row has been lost."
            )

        # y = y - X[:, : y.shape[1]]  # s_t+1 - s_t
        return df

    def df_concatenate_inputs(self, df):
        """Take the dataframe and concatenate as many steps as defined.
        Uses 'self.concatenated_steps' and 'self.concatenated_zero_padding', parsed during 'load_csv' method.

        Parameters
        ----------
        df: pd.Dataframe
            Dataframe with labels and features.
            .

        Returns
        -------
        pd.Dataframe
            List of dataframes with concatenated steps.

        """

        concatenated_steps = self.concatenated_steps
        zero_padding = self.concatenated_zero_padding

        # Drop episode if number of iterations is lower than number of desired concatenated steps.
        # - Dropped no matter if zero_padding is enabled or disabled -
        if len(df) < concatenated_steps:
            logger.debug(
                f"concatenated inputs enabled, concatenating {concatenated_steps} steps. zero_padding: {zero_padding}.\
                \n   >> We drop df, since df length ({len(df)}) is lower than number of steps to concatenate ({concatenated_steps})."
            )
            return None

        # Redefine input states to ensure input state names are unique
        # - Note, state names are used on predict_sequentially_all method (and possibly others)

        if not self.original_features:
            self.original_features = copy.deepcopy(self.feature_cols)
            # Note, naming convention needs to honor the way it is done in the subsequent loop
            self.feature_cols = [
                feat + f"_{i}"
                for i in range(1, concatenated_steps + 1)
                for feat in self.original_features
            ]

        if not hasattr(self, "aux_concat_index"):
            self.aux_concat_index = 0
        self.aux_concat_index += 1

        for feat in self.original_features:
            for i in range(1, concatenated_steps + 1):
                concat_feat = feat + f"_{i}"

                # Concatenate steps >> i == 1: has the newest value; i == concatenated_steps: has the oldest value
                if i == 1:
                    feat_array = df[feat].values
                else:
                    feat_array = df[feat].values[: -i + 1]
                    # pad with zeros by default (remove later if undesired)
                    feat_array = np.array(list(np.zeros(i - 1)) + list(feat_array))
                df[concat_feat] = feat_array

        # Removing zero padded tows, if padding with zeros is disabled.
        if not zero_padding:
            df.drop(df.head(concatenated_steps - 1).index, axis=0, inplace=True)

        # Store information on transformation performed on debugger.
        if zero_padding:
            logger.debug(
                f"concatenated inputs enabled, concatenating {concatenated_steps} steps. zero_padding: {zero_padding}. no rows have been lost."
            )
        else:
            logger.debug(
                f"concatenated inputs enabled, concatenating {concatenated_steps} steps. zero_padding: {zero_padding}. initial ({concatenated_steps-1}) rows are dropped."
            )

        return df


if __name__ == "__main__":

    data_dir = "csv_data"
    logger.info(f"Using data saved in directory {data_dir}")

    data_class = DataClass()
    df = pd.read_csv(os.path.join(data_dir, "cartpole-log.csv"), nrows=1000)
    df = data_class.read(df, iteration_order=-1)
    df2 = pd.read_csv(os.path.join(data_dir, "cartpole_at_st.csv"), nrows=1000)
    df2 = data_class.read(df2, iteration_order=1)
