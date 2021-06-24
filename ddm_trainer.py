import logging
import os
import pathlib
import hydra
import numpy as np
from math import floor
from omegaconf import DictConfig, ListConfig, OmegaConf

from model_loader import available_models

logger = logging.getLogger("datamodeler")
dir_path = os.path.dirname(os.path.realpath(__file__))


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    logger.info("Configuration: ")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # for readability, read common data args into variables
    input_cols = cfg["data"]["inputs"]
    output_cols = cfg["data"]["outputs"]
    augmented_cols = cfg["data"]["augmented_cols"]
    iteration_order = cfg["data"]["iteration_order"]
    episode_col = cfg["data"]["episode_col"]
    iteration_col = cfg["data"]["iteration_col"]
    dataset_path = cfg["data"]["path"]
    max_rows = cfg["data"]["max_rows"]
    test_perc = cfg["data"]["test_perc"]
    scale_data = cfg["data"]["scale_data"]
    delta_state = cfg["data"]["diff_state"]
    concatenated_steps = cfg["data"]["concatenated_steps"]
    concatenated_zero_padding = cfg["data"]["concatenated_zero_padding"]

    # common model args
    save_path = cfg["model"]["saver"]["filename"]
    model_name = cfg["model"]["name"]
    run_sweep = cfg["model"]["sweep"]["run"]
    split_strategy = cfg["model"]["sweep"]["split_strategy"]
    results_csv_path = cfg["model"]["sweep"]["results_csv_path"]

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
        train_split=1 - test_perc,
        diff_state=delta_state,
        concatenated_steps=concatenated_steps,
        concatenated_zero_padding=concatenated_zero_padding,
    )

    logger.info(
        f"Saving last {test_perc * 100}% for test, using first {(1 - test_perc) * 100}% for training/sweeping"
    )
    train_id_end = floor(X.shape[0] * (1 - test_perc))
    X_train, y_train = (
        X[:train_id_end,],
        y[:train_id_end,],
    )
    X_test, y_test = (
        X[train_id_end:,],
        y[train_id_end:,],
    )

    # save training and test sets
    save_data_path = os.path.join(os.getcwd(), "data")
    if not os.path.exists(save_data_path):
        pathlib.Path(save_data_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving data to {os.path.abspath(save_data_path)}")
    np.save(os.path.join(save_data_path, "x_train.npy"), X_train)
    np.save(os.path.join(save_data_path, "y_train.npy"), y_train)
    np.save(os.path.join(save_data_path, "x_test.npy"), X_test)
    np.save(os.path.join(save_data_path, "y_test.npy"), y_test)

    logger.info("Building model...")
    model.build_model(**cfg["model"]["build_params"])

    if run_sweep:
        params = OmegaConf.to_container(cfg["model"]["sweep"]["params"])
        logger.info(f"Sweeping with parameters: {params}")

        sweep_df = model.sweep(
            params=params,
            X=X_train,
            y=y_train,
            search_algorithm=cfg["model"]["sweep"]["search_algorithm"],
            num_trials=cfg["model"]["sweep"]["num_trials"],
            scoring_func=cfg["model"]["sweep"]["scoring_func"],
            results_csv_path=results_csv_path,
            splitting_criteria=split_strategy,
        )
        logger.info(f"Sweep results: {sweep_df}")
    else:
        logger.info("Fitting model...")
        model.fit(X_train, y_train)

    logger.info(f"Saving model to {save_path}")
    model.save_model(filename=save_path)


if __name__ == "__main__":

    main()
