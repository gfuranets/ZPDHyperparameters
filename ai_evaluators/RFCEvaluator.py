from ai_evaluators.IEvaluator import IEvaluator, TRAINING_RANDOM_STATE
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone


class RFCEvaluator(IEvaluator):
    def __init__(self):
        super().__init__()
        # Will change list to tuple, just need to write a wrapper for grid_search function first
        self.hyperparameters = {
            "n_estimators": [10, 30, 100],
            "max_depth": [1, 10, 20]
        }

    # Will probably implement this and some other functions in the IEvaluator
    def set_dataset(self, dataset: DataFrame):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            dataset.data, dataset.target, test_size=0.2, random_state=420)

    def get_search_space(self, strategy) -> dict:
        search_space = {}

        # Apply search algorithm for hyperparameters
        for param, args in self.hyperparameters.items():
            # *args unpacks tuple for use as arguments in the function
            search_space[param] = strategy(args)

        return search_space

    def evaluate(self, config):
        clf = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=TRAINING_RANDOM_STATE
        )

        # skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        #
        # for train_index, test_index in skfolds.split(self.X_train, self.y_train):
        #     clone_clf = clone(clf)
        #     X_train_folds = self.X_train[train_index]
        #     y_train_folds = self.y_train[train_index]
        #     X_test_fold = self.X_train[test_index]
        #     y_test_fold = self.y_train[test_index]
        #
        #     clone_clf.fit(X_train_folds, y_train_folds)
        #     y_pred = clone_clf.predict(X_test_fold)
        #     n_correct = sum(y_pred == y_test_fold)
        #     print(n_correct / len(y_pred))

        clf.fit(self.X_train, self.y_train)
        predictions = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)

        print(accuracy)

        return {"mean_metric": accuracy}


