import os
import pandas as pd
from typing import List
import logging

logger = logging.getLogger("data_loaders")
logger.setLevel(logging.INFO)


class CsvReader(object):
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
            labels_df = df[[episode_col, iteration_col] + self.label_cols]
        else:
            labels_df = df[[episode_col, iteration_col]].join(
                lagged_df[self.label_cols]
            )
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


if __name__ == "__main__":

    data_dir = "csv_data"
    logger.info(f"Using data saved in directory {data_dir}")

    csv_reader = CsvReader()
    # df = pd.read_csv(os.path.join(data_dir, "cartpole-log.csv"), nrows=1000)
    df = pd.read_csv(
        os.path.join(data_dir, "cartpole-mixed-brain-rand-term.csv"), nrows=1000
    )
    df = csv_reader.read(df, iteration_order=-1)
    # df2 = pd.read_csv(os.path.join(data_dir, "cartpole_at_st.csv"), nrows=1000)
    # df2 = csv_reader.read(df2, iteration_order=1)
