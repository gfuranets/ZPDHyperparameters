from ai_evaluators.RFCEvaluator import RFCEvaluator
from ai_evaluators.SGDCEvaluator import SGDCEvaluator

from dataset_controllers.IDatasetController import IDatasetController
from dataset_controllers.BrainTumorController import BrainTumorController

import ray
from ray import tune
from ray.tune import Stopper
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hebo import HEBOSearch

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from typing import List
import numpy as np

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


def compare():
    evaluator = RFCEvaluator()
    dataset_controller: IDatasetController = BrainTumorController()
    training_count = 10

    strategies = [
        ("Random Search", BasicVariantGenerator(), tune.uniform, training_count),  # Random Search
        ("HEBO Search", HEBOSearch(), tune.uniform, training_count),  # HEBO Search
    ]

    X_train, X_test, y_train, y_test = dataset_controller.get_sets()

    for strategy_name, strategy, search_space_algo, num_samples in strategies:
        print(f"\nStarting {strategy_name}...")

        search_space = evaluator.get_search_space(search_space_algo)
        search_space["X_train_id"], search_space["X_test_id"] = ray.put(X_train), ray.put(X_test)
        search_space["y_train_id"], search_space["y_test_id"] = ray.put(y_train), ray.put(y_test)
        search_space["searched_params"] = {}

        def evaluate_with_metrics(config):
            # Model training and evaluation function
            model = evaluator.get_model(config)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            # Return metrics to Ray Tune
            return {
                "f1_score": f1,
                "accuracy": accuracy,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn
            }

        # Set up Ray Tune with TuneConfig
        tuner = tune.Tuner(
            evaluate_with_metrics,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=strategy,
                mode="max",
                metric="f1_score",
                num_samples=num_samples,
            ),
            # Specify the stopping criterion and path for results here
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id[:8]}",  # Shortened trial ID
            resources={"cpu": 1}  # Specify resource allocation inside resources parameter
        )
        results = tuner.fit()

        print(f"\nResults for {strategy_name}:")
        # Print metrics for each trial
        for trial in results:
            metrics = trial.metrics
            print(f"Trial ID: {trial.trial_id}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  True Positives (TP): {metrics['tp']}")
            print(f"  True Negatives (TN): {metrics['tn']}")
            print(f"  False Positives (FP): {metrics['fp']}")
            print(f"  False Negatives (FN): {metrics['fn']}")
            print("-" * 40)

    ray.shutdown()
