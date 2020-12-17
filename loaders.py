import os
import pandas as pd
from typing import List, Tuple, Union
import logging

FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]")
logger = logging.getLogger("data_loader")
data_dir = "csv_data"
logger.info(f"Using data saved in directory {data_dir}")


class CsvReader(object):
    def read(
        self,
        filename: str,
        timelag: int = -1,
        episode_col: Union[str, None] = "episode",
        iteration_col: Union[str, None] = "iteration",
        feature_cols: Union[List, str] = "state_",
        max_rows: Union[int, None] = None,
    ):

        logger.info(f"Reading data from {filename}")
        df = pd.read_csv(filename, nrows=max_rows)

        # CASE 1: rows are of the form {st+1, at}
        # Append st into next row
        # if timelag < 0 then drop the iteration - timelag iteration from each episode
        # and append previous state columns into each row: {st+1, at} -> {st, at, st+1}
        if all([episode_col, iteration_col, timelag < 0]):
            logger.info(
                f"Timelag set {timelag} so using features from {timelag} previous row"
            )
            df = df.sort_values(by=[episode_col, iteration_col])
            lagged_df = df.groupby(by=episode_col, as_index=False).shift(timelag * -1)
            lagged_df = lagged_df.drop([iteration_col], axis=1)
            if type(feature_cols) == list:
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
            joined_df = df.join(lagged_df)
            # skip the first row of each episode since we do not have its st
            joined_df = (
                joined_df.groupby(by=episode_col, as_index=False)
                .apply(lambda x: x.iloc[timelag * -1 :])
                .reset_index()
            )
            return joined_df.drop(["level_0", "level_1"], axis=1)
        # CASE 2: rows of the form {st, at}
        # Append st+1 from next row into current row {st, at, st+1}
        elif all([episode_col, iteration_col, timelag > 0]):
            logger.info(
                f"Timelag set {timelag} so using ouputs from {timelag} next row"
            )
            df = df.sort_values(by=[episode_col, iteration_col])
            lagged_df = df.groupby(by=episode_col, as_index=False).shift(timelag * -1)
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
            # truncate before the end of timelag for complete observations only
            joined_df = (
                joined_df.groupby(by=episode_col, as_index=False)
                .apply(lambda x: x.iloc[: timelag * -1])
                .reset_index()
            )
            return joined_df.drop(["level_0", "level_1"], axis=1)
        else:
            return df


if __name__ == "__main__":

    csv_reader = CsvReader()
    df = csv_reader.read(
        os.path.join(data_dir, "cartpole-log.csv"), timelag=-1, max_rows=1000
    )
    df2 = csv_reader.read(
        os.path.join(data_dir, "cartpole_at_st.csv"), timelag=1, max_rows=1000
    )
