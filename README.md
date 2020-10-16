# Data driven model creation for simulators to train brains on Bonsai

Tooling to simplify the creation and use of data driven simulators using supervised learning with the purpose of training brains with Project Bonsai. It digests data as csv and will generate simulation models which can then be directly used to train a reinforcement learning agent.

>🚩 Disclaimer: This is not an official Microsoft product. This application is considered an experimental addition to Microsoft Project Bonsai's software toolchain. It's primary goal is to reduce barriers of entry to use Project Bonsai's core Machine Teaching. Pull requests for fixes and small enhancements are welcome, but we do expect this to be replaced by out-of-the-box features of Project Bonsai in the near future.

## Dependencies

```bash
conda env update -f environment.yml
conda activate datadriven
```

## Main steps to follow

`Step 1.` Obtain historical or surrogate sim data in csv format.

- header names 
- a single row should be a slice in time
- the sequential rows should be within the desired cadence of control frequency
- ensure data ranges cover what we might set reinforcement learning to explore
- smooth noisy data, get rid of outliers
- remove NaN or SNA values. 

> Note: LSTM will require three dimensional data, [sample, timestep, feature]

Refer to later sections for help with checking data quality before using this tool.

`Step 2.` Change the `config_model.yml` file in the `config/` folder

Enter the csv file name. Define the names of the features as input to the simulator model you will create. The names should match the headers of the csv file you provide. Set the values as either `state` or `action`. Define the `output_name` matching the headers in your csv. 

Define the model type as either `gb, poly, nn, or lstm`. Depending on the specific model one chooses, alter the hyperparameters in this config file as well. 

```YAML
# Define csv file path to train a simulator with
DATA:
  path: example_data.csv
# Define the inputs and outputs of datadriven simulator
IO:
  feature_name:
    theta: state
    alpha: state
    theta_dot: state
    alpha_dot: state
    Vm: action
  output_name:
    - theta
    - alpha
    - theta_dot
    - alpha_dot
# Select the model type gb, poly, nn, or lstm
MODEL:
  type: gb
# Polynomial Regression hyperparameters
POLY:
  degree: 1
# Gradient Boost hyperparameters
GB:
  n_estimators: 100
  lr: 0.1
  max_depth: 3
# MLP Neural Network hyperparameters
NN:
  epochs: 100
  batch_size: 512
  activation: linear
  n_layer: 5
  n_neuron: 12
  lr: 0.00001
  decay: 0.0000003
  dropout: 0.5
# LSTM Neural Network hyperparameters
LSTM:
  epochs: 100
  batch_size: 512
  activation: linear
  num_hidden_layer: 5
  n_neuron: 12
  lr: 0.00001
  decay: 0.0000003
  dropout: 0.5
  markovian_order: 2
  num_lstm_units: 1
```

`Step 3.` Run the tool

```bash
python datamodeler.py
```

The tool will ingest your csv file as input and create a simulator model of the type you selected. THe resultant model will be placed into `models/`.

`Step 4.` Use the model directly 

An adaptor class is available for usage in the following way to make custom integrations. We've already done this for you in `Step 5`, but this provides a good understanding. Initialize the class with model type, which consists of either `'gb', 'poly', 'nn', or 'lstm'`. 

Specify a `noise_percentage` to optionally add to the states of the simulator, leaving it at zero will not add noise. Training a brain can benefit from adding noise to the states of an approximated simulator to promote robustness.

Define the `action_space_dimensions` and the `state_space_dimensions`. The `markovian_order` is needed when setting the sequence length of the features for an `LSTM`. 

```python
from predictor import ModelPredictor

predictor = ModelPredictor(
    modeltype="gb",
    noise_percentage=0,
    state_space_dim=4,
    action_space_dim=1,
    markovian_order=0
)
```

Calculate next state as a function of current state and current action. IMPORTANT: input state and action are arrays. You need to convert brain action, i.e. dictionary, to an array before feeding into the predictor class. 

```python
next_state = predictor.predict(action=action, state=state)
```

The thing to watch out for with datadriven simulators is one can trust the approximations when the feature inputs are not within the range it was trained on, i.e. you may get erroneous results. One can optionally evaluate if this occurs by using the `warn_limitation()` functionality. 

```python
features = np.concatenate([state, action]
predictor.warn_limitation(features)
```
> Sim should not be necessarily trusted since predicting with the feature Vm outside of range it was trained on, i.e. extrapolating.

`Step 5.` Train with Bonsai

Create a brain and write Inkling with type definitions that match what the simulator can provide, which you defined in `config_model.yml`. Run the `train_bonsai_main.py` file to register your newly created simulator. The integration is already done! Then connect the simulator to your brain.

Be sure to specify `noise_percentage` in your Inkling's scenario. Training a brain can benefit from adding noise to the states of an approximated simulator to promote robustness.

```javascript
lesson `Start Inverted` {
    scenario {
        theta: number<-1.4 .. 1.4>,
        alpha: number<-0.05 .. 0.05>,  # reset inverted
        theta_dot: number <-0.05 .. 0.05>,
        alpha_dot: number<-0.05 .. 0.05>,
        noise_percentage: 0.05,
    }
}
```

> Ensure the SimConfig in Inkling matches the names of the headers in the `config_model.yml` to allow `train_bonsai_main.py` to work.

```bash
python train_bonsai_main.py --workspace <workspace-id> --accesskey <accesskey>
```

## Data Cleaning

This helps in removing the Outliers and NaNs in the data. Outlier detection algorithm fits y_set as a function of x_set, and if the prediction by the model is far away from the actual value then we define the point as outlier

### Main steps to follow to detect outliers

Once the data is generated (in the same way as data generation process for datadriven modeling), we can detect outliers in the following way:

```bash 
>>> python checkDataQuality.py
>>> python checkDataQuality.py --thrhld 4
```

By default, outlier threshold is 3, it can be adjusted based on the data. 

## Feature Importance

This code helps in computing feature importance using gradient boosting trees.

### Main steps to follow to detect outliers

Once the data is generated (in the same way as data generation process for datadriven modeling), we can compute feature importance in the following way:

```bash 
>>> python featureImportance.py
```

## Contribute Code
This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Telemetry
The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

## Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.