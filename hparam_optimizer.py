import ray
from ray import tune

from ai_evaluators.IEvaluator import IEvaluator
from ai_evaluators.RFCEvaluator import RFCEvaluator

from dataset_controllers.IDatasetController import IDatasetController
from dataset_controllers.MNISTDatasetController import MNISTDatasetController
from dataset_controllers.MNISTFashionDatasetController import MNISTFashionDatasetController

from typing import List


def compare():
    # Evaluators, or AI models that will be used in testing
    evaluators: List[IEvaluator] = [RFCEvaluator()]

    # DatasetController constructors also fetch the datasets 
    dataset_controllers: List[IDatasetController] = [MNISTDatasetController()]

    # Hyperparameter search strategies
    strategies = [tune.grid_search]

    for evaluator in evaluators:
        for dataset_cont in dataset_controllers:
            # Retrieving train set and data set from the dataset controller
            X_train, X_test, y_train, y_test = dataset_cont.get_sets()
            for strategy in strategies:
                # Constraining parallel tasks, by assigning minimum resources usage
                evaluate_with_resources = tune.with_resources(evaluator.evaluate, {"cpu": 1})

                # Getting a search space with specific strategy
                search_space = evaluator.get_search_space(strategy)

                # Passing dataset reference id's to evaluator via config
                search_space['X_train_id'], search_space['X_test_id'] = ray.put(X_train), ray.put(X_test)
                search_space['y_train_id'], search_space['y_test_id'] = ray.put(y_train), ray.put(y_test)

                tuner = tune.Tuner(
                    evaluate_with_resources,
                    param_space=search_space
                )
                results = tuner.fit()
                print(results.get_best_result(metric="f1_score", mode="max").config)
