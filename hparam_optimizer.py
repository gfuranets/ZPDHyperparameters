from ai_evaluators.RFCEvaluator import RFCEvaluator

from dataset_controllers.IDatasetController import IDatasetController
from dataset_controllers.BrainTumorController import BrainTumorDatasetController

import ray
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hebo import HEBOSearch

from typing import *

# Value of maximum score is acquired by running full grid search
# MAXIMUM_SCORE = 0.87540

# class ExperimentStopper(Stopper):
#     def __init__(self):
#         self.should_stop = False

#     def __call__(self, trial_id, result):
#         if result["f1_score"] >= MAXIMUM_SCORE:
#             self.should_stop = True
#         return self.should_stop

#     def stop_all(self):
#         return self.should_stop

def compare(training_count, strategy_position):
    evaluator = RFCEvaluator()
    dataset_controller: IDatasetController = BrainTumorDatasetController()

    strategies = [
        ("Random Search", BasicVariantGenerator(), tune.uniform, training_count),  # Random Search
        ("HEBO Search", HEBOSearch(), tune.uniform, training_count)  # HEBO Search
    ]

    evaluate_with_resources = tune.with_resources(evaluator.evaluate, {"cpu": 1})

    X_train, X_test, y_train, y_test = dataset_controller.get_sets()

    # Define the current algorithm
    strategy_name, strategy, search_space_algo, num_samples = strategies[strategy_position]
    print(f"\nStarting {strategy_name} with {training_count} iterations:")

    search_space = evaluator.get_search_space(search_space_algo)
    search_space["X_train_id"], search_space["X_test_id"] = ray.put(X_train), ray.put(X_test)
    search_space["y_train_id"], search_space["y_test_id"] = ray.put(y_train), ray.put(y_test)
    search_space["searched_params"] = {}

    # Set up Ray Tune with TuneConfig
    tuner = tune.Tuner(
        evaluate_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=strategy,
            mode="max",
            metric="f1_score",
            num_samples=num_samples
        )
        # Specify the stopping criterion and path for results here
        # trial_dirname_creator=lambda trial: f"trial_{trial.trial_id[:8]}",  # Shortened trial ID
    )
    results = tuner.fit()
    print(results.get_best_result(metric="f1_score", mode="max").config)
        
    ray.shutdown()
