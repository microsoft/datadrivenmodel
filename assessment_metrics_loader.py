"""
Any desired assessment metric should be added here

"""
# DEVELOPER TODO: Add your favorite assessing metrics
from sklearn.metrics import mean_squared_error, r2_score


def root_mean_squared_error(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average", squared=False
):
    return mean_squared_error(y_true, y_pred, sample_weight, multioutput, squared)


available_metrics = {
    "mean_squared_error": mean_squared_error,
    "root_mean_squared_error": root_mean_squared_error,
    "r2_score": r2_score,
}
