import logging
import os
import pathlib
import hydra
import numpy as np
from math import floor
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.metrics import r2_score, mean_squared_error

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

    # common model args
    save_path = cfg["model"]["saver"]["filename"]
    model_name = cfg["model"]["name"]
    delta_state = cfg["data"]["diff_state"]
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
    path = os.path.join(dir_path, "csv_data/expert_19k/")
    with open(path + 'X.npy', 'rb') as f:
        X = np.load(f)

    with open(path + 'y.npy', 'rb') as f:
        y = np.load(f)

    if delta_state == True:
        logging.info(
            "delta states enabled, calculating differential between input and output values"
        )
        y = y - X[:, : y.shape[1]] # s_t+1 - s_t

    # X, y = model.load_csv(
    #     input_cols=input_cols,
    #     output_cols=output_cols,
    #     augm_cols=augmented_cols,
    #     dataset_path=dataset_path,
    #     iteration_order=iteration_order,
    #     episode_col=episode_col,
    #     iteration_col=iteration_col,
    #     max_rows=max_rows,
    #     diff_state=delta_state,
    # )

    logger.info(
        f"Saving last {test_perc * 100}% for test, using first {(1 - test_perc) * 100}% for training/sweeping"
    )
    train_id_end = floor(X.shape[0] * (1 - test_perc))
    X_train, y_train = (
        X[
            :train_id_end,
        ],
        y[
            :train_id_end,
        ],
    )
    X_test, y_test = (
        X[
            train_id_end:,
        ],
        y[
            train_id_end:,
        ],
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

    if delta_state:
        y_pred = model.predict(X_train)
        y_pred += X_train[:, : y_pred.shape[1]]
        y_train += X_train[:, : y_pred.shape[1]]
    else:
        y_pred = model.predict(X_train)
    logger.info(f"R^2 score, samples: {y_pred.shape[0]} is {r2_score(y_train, y_pred)} for train set. "
                f"MSE: {mean_squared_error(y_train, y_pred)}")

    if delta_state:
        y_pred = model.predict(X_test)
        y_pred += X_test[:, : y_pred.shape[1]]
        y_test += X_test[:, : y_pred.shape[1]]
    else:
        y_pred = model.predict(X_test)
    logger.info(f"R^2 score, samples: {y_pred.shape[0]} - {test_perc*100}% is {r2_score(y_test,y_pred)} for test set. "
                f"MSE: {mean_squared_error(y_test, y_pred)}")


    logger.info(f"Saving model to {save_path}")
    model.save_model(filename=save_path)


if __name__ == "__main__":

    main()
