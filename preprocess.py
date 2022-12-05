""""
Add your preprocessing functions here
These will get called in ddm_trainer.py and ddm_predictor.py

Define as many functions as you want, and chain them together in the pipeline function
"""


def calculate_deltas(df):

    df["action.CDW_SWS_RWT_Delta"] = df["state.CDW_RWT"] - df["action.CDW_SWS"]
    df["action.CDW_SWS_WBT_Delta"] = df["state.WBT"] - df["action.CDW_SWS"]
    df["action.CHW_SWS_RWT_Delta"] = df["state.CHW_RWT"] - df["action.CHW_SWS"]
    df["action.CHW_SWS_OAT_Delta"] = df["state.OAT"] - df["action.CHW_SWS"]

    return df


def pipeline(df):

    return calculate_deltas(df)
