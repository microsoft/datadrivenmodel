# from torch_models import PyTorchModel
# from gboost_models import GBoostModel
import typer
import os
import logging

import model_loader
from model_loader import available_models

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("datamodeler")

# available_models = {"pytorch": PyTorchModel, "gboost": GBoostModel}


with open("config/config_model.yml") as cmfile:
        config = yaml.full_load(cmfile)

config_model = config["MODEL"]["type"]

def main(
    dataset_path: str,
    input_cols="state",
    model: str = config_model,
    augm_cols=["action_command", "config_length", "config_masspole"],
    output_col="state",
    save_file_path="saved_models",
):
    """Train a Data-Driven Simulator Using Logged Dataset of Interactions with a Fixed Environment
    """

    logger.info(f"Using model type {model}")
    model_class = available_models[model]()
    X, y = model_class.load_csv(
        dataset_path=dataset_path,
        input_cols_read=input_cols,
        output_col=output_col,
        augm_cols=augm_cols,
    )
    model_class.build_model()
    model_class.fit(X, y)
    model_class.save_model(filename=os.path.join(save_file_path, "trained_model.pkl"))


if __name__ == "__main__":

    typer.run(main)
