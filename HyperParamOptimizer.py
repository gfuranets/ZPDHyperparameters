from ControllerEvaluator import ControllerEvaluator
import ray
from ray import tune

from ai_evaluators.IEvaluator import IEvaluator
from ai_evaluators.RFCEvaluator import RFCEvaluator

from dataset_controllers.IDatasetController import IDatasetController
from dataset_controllers.IRISDatasetController import IRISDatasetController
from dataset_controllers.MNISTDatasetController import MNISTDatasetController

from typing import List
from pandas import DataFrame

RAYTUNE_SAMPLES = 30

# Number of CPU and GPU depends on system
USE_GPU = False
NUM_GPUS = 1.0
# CPU: 0 tells it to auto-allocate cpu resources
RAYTUNE_TRIAL_RESOURCES = {
    "CPU": 0 if USE_GPU else 0.1,
    "GPU": NUM_GPUS / RAYTUNE_SAMPLES if USE_GPU else 0.0
}

# Define hyperparameters and their arguments
hyperparameters = {
    "n_estimators": (10, 100),
    "max_depth": (1, 10)
}


def get_search_space(search_algo):
    result = {}

    # Apply search algorithm for hyperparameters
    for param, args in hyperparameters.items():
        # *args unpacks tuple for use as arguments in the function
        result[param] = search_algo(*args)
    return result


def objective(config):
    # The function to be minimized by hyperparameter optimization
    controller = ControllerEvaluator(config)
    return controller.evaluate()


def main():
    ray.init(num_gpus=NUM_GPUS)

    search_space = get_search_space(tune.randint)

    # Run hyperparameter search
    analysis = tune.run(
        objective,
        config=search_space,
        num_samples=RAYTUNE_SAMPLES,
        metric="mean_metric",
        mode="max",
        resources_per_trial=RAYTUNE_TRIAL_RESOURCES
    )
    
    # Print the best result
    print("Best hyperparameters found were: ", analysis.best_config)


def compare():
    evaluators: List[IEvaluator] = [RFCEvaluator()]
    dataset_controllers: List[IDatasetController] = [MNISTDatasetController()]
    # Hyperparameter search strategies
    strategies = [tune.grid_search]
    datasets: List[DataFrame] = []

    # Fetching and preparing datasets
    for dataset_controller in dataset_controllers:
        dataset_controller.prepare_dataset()
        datasets.append(dataset_controller.get_dataset())

    for evaluator in evaluators:
        for dataset in datasets:
            evaluator.set_dataset(dataset)
            for strategy in strategies:
                search_space = evaluator.get_search_space(strategy)
                analysis = tune.run(
                    evaluator.evaluate,
                    config=search_space,
                    metric="mean_metric",
                    mode="max",
                    num_samples=1,
                    resources_per_trial=RAYTUNE_TRIAL_RESOURCES
                )

                print("Best hyperparameters found were: ", analysis.best_config)
