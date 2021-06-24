"""
Any newly developed model should be added here as imporrted model and then also instantiated inside
the following function

"""
# DEVELOPER TODO: Add your favorite models bases on abstract class here and also add then to available_models
from torch_models import PyTorchModel
from skmodels import SKModel
from gboost_models import GBoostModel

available_models = {
    "pytorch": PyTorchModel,
    "linear_model": SKModel,
    "SVR": SKModel,
    "GradientBoostingRegressor": SKModel,
    "sklearn": SKModel,
    "SGDRegressor": SKModel,
    "gboost": GBoostModel,
    "xgboost": GBoostModel,
    "lightgbm": GBoostModel,
}
