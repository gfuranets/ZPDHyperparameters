from abc import ABC, abstractmethod
from ray import tune
from pandas import DataFrame

TRAINING_RANDOM_STATE = 420


class IEvaluator(ABC):
    def __init__(self):
        # Key - hparam name
        # Value - list with hparam values
        # Value[0] - list of possible hparam values (used for grid search etc.)
        # Value[1] - tuple with two values, lower bound and upper bound (for bayesian opt. etc.)
        self.hyperparameter_config = {}

    def get_search_space(self, search_space_algo) -> dict:
        """
        Returns a search space with specific AI model hparams, for specific search space algo
        :param search_space_algo: grid_search, uniform etc.
        :return: dict, search space
        """
        search_space = {}

        # Hardcoded:
        # grid_search or choice - use sequence
        # uniform or randint - use range
        k = 0
        if search_space_algo == tune.uniform or search_space_algo == tune.randint:
            k = 1

        # Apply search algorithm for hyperparameters
        for param, args in self.hyperparameters.items():
            # *args unpacks tuple for use as arguments in the function
            search_space[param] = search_space_algo(args[k])

        return search_space

    @abstractmethod
    def evaluate(self, config, X_train, X_test, y_train, y_test):
        pass
