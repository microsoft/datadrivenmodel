import logging
import os
import pathlib
import hydra
import numpy as np
import pandas as pd
from math import floor
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.metrics import r2_score

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

    if model_name.lower() == "pytorch":
        from all_models import available_models
    else:
        from model_loader import available_models

    Model = available_models[model_name]

    # TODO, decide whether to always save to outputs directory
    if cfg["data"]["full_or_relative"] == "relative":
        dataset_path = os.path.join(dir_path, dataset_path)

    save_path = os.path.join(dir_path, save_path)

    if type(input_cols) == ListConfig:
        input_cols = list(input_cols)
    if type(output_cols) == ListConfig:
        output_cols = list(output_cols)
    if type(augmented_cols) == ListConfig:
        augmented_cols = list(augmented_cols)

    model = Model()
    # Add extra preprocessing step inside load_csv
    # should be done before concatenate_steps
    X_train, y_train, X_test, y_test = model.load_csv(
        dataset_path=dataset_path,
        input_cols=input_cols,
        augm_cols=augmented_cols,
        output_cols=output_cols,
        iteration_order=iteration_order,
        episode_col=episode_col,
        iteration_col=iteration_col,
        # drop_nulls: bool = True,
        max_rows=max_rows,
        test_perc=test_perc,
        diff_state=delta_state,
        concatenated_steps=concatenated_steps,
        concatenated_zero_padding=concatenated_zero_padding,
    )

    logger.info(
        f"From the full dataset, {test_perc * 100}% will be used for test, while {(1 - test_perc) * 100}% for training/sweeping"
    )
    # X_train, y_train = model.get_train_set(grouped_per_episode=False)
    # X_test, y_test = model.get_test_set(grouped_per_episode=False)

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

    y_pred = model.predict(X_test)
    logger.info(f"R^2 score is {r2_score(y_test,y_pred)} for test set.")
    logger.info(f"Saving model to {save_path}")
    model.save_model(filename=save_path)

    ## save datasets
    pd.DataFrame(X_train, columns=model.feature_cols).to_csv(
        os.path.join(save_data_path, "x_train.csv")
    )
    pd.DataFrame(X_test, columns=model.feature_cols).to_csv(
        os.path.join(save_data_path, "x_test.csv")
    )
    pd.DataFrame(y_train, columns=model.label_cols).to_csv(
        os.path.join(save_data_path, "y_train.csv")
    )
    pd.DataFrame(y_test, columns=model.label_cols).to_csv(
        os.path.join(save_data_path, "y_test.csv")
    )


if __name__ == "__main__":

    main()
