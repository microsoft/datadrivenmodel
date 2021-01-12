import os
import pandas as pd
from typing import List, Tuple, Union
import logging

logger = logging.getLogger("data_loaders")
logger.setLevel(logging.INFO)


class CsvReader(object):
    def order_by_time(self, time_col: str = "timestamp"):

        pass

    def read(
        self,
        filename: str,
        iteration_order: int = -1,
        episode_col: str = "episode",
        iteration_col: str = "iteration",
        feature_cols: Union[List, str] = "state_",
        max_rows: Union[int, None] = None,
    ):
        """Read episodic data where each row contains either inputs and its preceding output output or the causal inputs/outputs relationship

        Parameters
        ----------
        filename : str
            [description]
        iteration_order : int, optional
            [description], by default -1
        episode_col : str, optional
            [description], by default "episode"
        iteration_col : str, optional
            [description], by default "iteration"
        feature_cols : Union[List, str], optional
            [description], by default "state_"
        max_rows : Union[int, None], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """

        logger.info(f"Reading data from {filename}")
        df = pd.read_csv(filename, nrows=max_rows)

        # CASE 1: rows are of the form {st+1, at}
        # Append st into next row
        # if iteration_order < 0 then drop the iteration - iteration_order iteration from each episode
        # and append previous state columns into each row: {st+1, at} -> {st, at, st+1}
        if all([episode_col, iteration_col, iteration_order < 0]):
            logger.info(
                f"Iteration order set to {iteration_order} so using inputs from previous {iteration_order} row"
            )
            df = df.sort_values(by=[episode_col, iteration_col])
            lagged_df = df.groupby(by=episode_col, as_index=False).shift(
                iteration_order * -1
            )
            lagged_df = lagged_df.drop([iteration_col], axis=1)
            if type(feature_cols) == list:
                self.feature_cols = feature_cols
                lagged_df = lagged_df[feature_cols]
            else:
                self.feature_cols = [
                    col for col in lagged_df if col.startswith(feature_cols)
                ]
                lagged_df = lagged_df[self.feature_cols]
                lagged_df = lagged_df.rename(columns=lambda x: "prev_" + x)
                logger.info(
                    f"Previous states are being added to same row with prefix prev_"
                )
                self.feature_cols = list(lagged_df.columns.values)
            logger.info(f"Feature columns are: {self.feature_cols}")
            joined_df = df.join(lagged_df)
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
            logger.info(
                f"Iteration order set to {iteration_order} so using outputs from next {iteration_order} row"
            )
            df = df.sort_values(by=[episode_col, iteration_col])
            lagged_df = df.groupby(by=episode_col, as_index=False).shift(
                iteration_order * -1
            )
            lagged_df = lagged_df.drop([iteration_col], axis=1)
            if type(feature_cols) == list:
                lagged_df = lagged_df[feature_cols]
            else:
                self.feature_cols = [
                    col for col in lagged_df if col.startswith(feature_cols)
                ]
                lagged_df = lagged_df[self.feature_cols]
                lagged_df = lagged_df.rename(columns=lambda x: "next_" + x)
                self.feature_cols = list(lagged_df.columns.values)
                logger.info(
                    f"Next states are being added to same row with prefix next_"
                )
            joined_df = df.join(lagged_df)
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
    df = csv_reader.read(
        os.path.join(data_dir, "cartpole-log.csv"), iteration_order=-1, max_rows=1000
    )
    df2 = csv_reader.read(
        os.path.join(data_dir, "cartpole_at_st.csv"), iteration_order=1, max_rows=1000
    )
