from ai_evaluators.SVCEvaluator import SVCEvaluator
import ray
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.zoopt import ZOOptSearch
from ray.tune.search.hebo import HEBOSearch

from ai_evaluators.IEvaluator import IEvaluator
from ai_evaluators.RFCEvaluator import RFCEvaluator
from ai_evaluators.SGDCEvaluator import SGDCEvaluator

from dataset_controllers.IDatasetController import IDatasetController
from dataset_controllers.MNISTDatasetController import MNISTDatasetController
from dataset_controllers.MNISTFashionDatasetController import MNISTFashionDatasetController

from typing import List


def compare():
    # Evaluators, or AI models that will be used in testing
    evaluators: List[IEvaluator] = [SGDCEvaluator()]

    # DatasetController constructors also fetch the datasets 
    dataset_controllers: List[IDatasetController] = [MNISTDatasetController()]

    # Hyperparameter search strategies
    strategies = [
        # (BayesOptSearch(random_search_steps=4), tune.uniform, 32), # Bayesian Search
        # (ZOOptSearch(budget=16), tune.randint, 16), # Zeroth-order Optimization Search
        # (BasicVariantGenerator(), tune.grid_search, 1), # Grid Search
        # (BasicVariantGenerator(), tune.uniform, 2) # Random Search
        (HEBOSearch(), tune.uniform, 2), # HUAWEI Search
    ]

    for evaluator in evaluators:
        # Constraining parallel tasks, by assigning minimum resources usage
        evaluate_with_resources = tune.with_resources(evaluator.evaluate, {"cpu": 1, "gpu": 0.1})

        for dataset_cont in dataset_controllers:
            # Retrieving train set and data set from the dataset controller
            X_train, X_test, y_train, y_test = dataset_cont.get_sets()

            for strategy, search_space_algo, num_samples in strategies:
                # Getting a search space with specific strategy
                search_space = evaluator.get_search_space(search_space_algo)

                # Passing dataset reference id's to evaluator via config
                search_space['X_train_id'], search_space['X_test_id'] = ray.put(X_train), ray.put(X_test)
                search_space['y_train_id'], search_space['y_test_id'] = ray.put(y_train), ray.put(y_test)

                tuner = tune.Tuner(
                    evaluate_with_resources,
                    param_space=search_space,
                    tune_config=tune.TuneConfig(
                        search_alg=strategy,
                        mode="max",
                        metric="f1_score",
                        num_samples=num_samples
                    )
                )
                results = tuner.fit()
                print(results.get_best_result(metric="f1_score", mode="max").config)
