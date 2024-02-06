from ai_evaluators.IEvaluator import IEvaluator, TRAINING_RANDOM_STATE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import ray
from ray import train


class RFCEvaluator(IEvaluator):
    def __init__(self):
        super().__init__()
        self.hyperparameters = {
            "n_estimators": ([15, 45, 70, 85, 100], (10, 100)),
            "max_depth": ([5, 10, 15, 20, 25], (5, 30)),
            "min_samples_split": ([2, 3, 4, 5, 6], (2, 6)),
            "min_samples_leaf": ([1, 2, 3, 4, 6], (1, 6))
        }

    def evaluate(self, config):
        params = (
            int(config["n_estimators"]),
            int(config["max_depth"]),
            int(config["min_samples_split"]),
            int(config["min_samples_leaf"])
        )

        if params in config['searched_params']:
            return train.report({"f1_score": config['searched_params'][params]})

        clf = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            min_samples_split=int(config["min_samples_split"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            random_state=TRAINING_RANDOM_STATE
        )

        X_train = ray.get(config["X_train_id"])
        X_test = ray.get(config["X_test_id"])
        y_train = ray.get(config["y_train_id"])
        y_test = ray.get(config["y_test_id"])


        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        f1 = f1_score(y_test, predictions, average='micro')

        train.report({"f1_score": f1})
        config['searched_params'][params] = f1

