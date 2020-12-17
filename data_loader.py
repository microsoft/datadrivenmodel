import os
import pandas as pd
from typing import List, Tuple, Union
import logging

FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]")
logger = logging.getLogger("data_loader")


def csv_reader(
    filename: str,
    timelag: int = 1,
    episode_col: Union[str, None] = "episode",
    iteration_col: Union[str, None] = "iteration",
):

    df = pd.read_csv(filename)
    # if timelag != 0 then drop the last_episode - timelag iteration from each episode
    # and append next state columns into each row: {row_t, row_{t+timelag}} -> {st, at, st+1}
    if all([episode_col, iteration_col, timelag != 0]):
        df = df.sort_values(by=[episode_col, iteration_col])
        neg_lag = timelag * -1
        lagged_df = df.groupby(by=episode_col, as_index=False).shift(neg_lag)
        lagged_df = lagged_df.drop([iteration_col], axis=1)
        joined_df = df.join(lagged_df.rename(columns=lambda x: "lag_" + x))
        # truncate before the end of timelag for complete observations only
        joined_df = (
            joined_df.groupby(by=episode_col, as_index=False)
            .apply(lambda x: x.iloc[:neg_lag])
            .reset_index()
        )
        return joined_df.drop(["level_0", "level_1"], axis=1)
    else:
        return df


if __name__ == "__main__":

    data_dir = "csv_data"
    df = csv_reader(os.path.join(data_dir, "cartpole-log.csv"))
