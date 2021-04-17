import logging
import os

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("datamodeler")

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from model_loader import available_models

dir_path = os.path.dirname(os.path.realpath(__file__))


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    logger.info("Configuration: ")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    input_cols = cfg["data"]["inputs"]
    output_cols = cfg["data"]["outputs"]
    augmented_cols = cfg["data"]["augmented_cols"]
    iteration_order = cfg["data"]["iteration_order"]
    episode_col = cfg["data"]["episode_col"]
    iteration_col = cfg["data"]["iteration_col"]
    dataset_path = cfg["data"]["path"]
    max_rows = cfg["data"]["max_rows"]
    save_path = cfg["model"]["saver"]["filename"]
    model_name = cfg["model"]["name"]
    delta_state = cfg["data"]["diff_state"]
    run_sweep = cfg["model"]["sweep"]["run"]

    Model = available_models[model_name]

    if cfg["data"]["full_or_relative"] == "relative":
        dataset_path = os.path.join(dir_path, dataset_path)

    save_path = os.path.join(dir_path, save_path + ".pkl")

    if type(input_cols) == ListConfig:
        input_cols = list(input_cols)
    if type(output_cols) == ListConfig:
        output_cols = list(output_cols)
    if type(augmented_cols) == ListConfig:
        augmented_cols = list(augmented_cols)

    model = Model()
    X, y = model.load_csv(
        input_cols=input_cols,
        output_cols=output_cols,
        augm_cols=augmented_cols,
        dataset_path=dataset_path,
        iteration_order=iteration_order,
        episode_col=episode_col,
        iteration_col=iteration_col,
        max_rows=max_rows,
        diff_state=delta_state,
    )
    logger.info("Building model...")
    model.build_model(**cfg["model"]["build_params"])

    if run_sweep:
        params = OmegaConf.to_container(cfg["model"]["sweep"]["params"])
        logger.info(f"Sweeping with parameters: {params}")

        model.sweep(
            params=params,
            X=X,
            y=y,
            search_algorithm=cfg["model"]["sweep"]["search_algorithm"],
            num_trials=cfg["model"]["sweep"]["num_trials"],
            scoring_func=cfg["model"]["sweep"]["scoring_func"],
        )
    else:
        logger.info("Fitting model...")
        model.fit(X, y)

    logger.info(f"Saving model to {save_path}")
    model.save_model(filename=save_path)


if __name__ == "__main__":

    main()
