"""
Aux script to track key metrics during job training on Azure ML.

"""
# Callback-related imports to track task progress in AML
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from azureml.core import Run
from ray.tune.logger import pretty_print


class CurriculumCallback(DefaultCallbacks):
    """A custom callback class that logs key training metrics to tensorboard and
    Azure ML.

    This class inherits from the DefaultCallbacks class provided by RLlib and overrides
    the on_train_result methods to access the episode object, and log key performance
    metrics to both tensorboard and Azure ML.
    """

    def __init__(self):
        self.run = Run.get_context()

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        print(
            "Algorithm.train() result: {} -> {} episodes".format(
                algorithm, result["episodes_this_iter"]
            )
        )
        # Print each episodic results
        pretty_print(result)

        # Log metrics to TensorBoard
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        # Filter the results dictionary to only log metrics with the substring "episode"
        to_log = {
            k: v for k, v in result.items() if "episode" in k and "media" not in k
        }
        # Log metrics to Azure ML
        for k, v in to_log.items():
            self.run.log(name=k, value=v)
