from ai_evaluators.IEvaluator import IEvaluator, TRAINING_RANDOM_STATE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import ray


class RFCEvaluator(IEvaluator):
    def __init__(self):
        super().__init__()
        self.hyperparameters = {
            "n_estimators": [[10, 30, 100], (10, 50)],
            "max_depth": [[1, 10, 20], (0, 20)],
            "min_samples_split": [[2, 3, 5], (2, 5)],
            "min_samples_leaf": [[1, 3, 5], (1, 5)]
        }

    def evaluate(self, config):
        clf = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min__samples_leaf"],
            random_state=TRAINING_RANDOM_STATE
        )

        X_train = ray.get(config["X_train_id"])
        X_test = ray.get(config["X_test_id"])
        y_train = ray.get(config["y_train_id"])
        y_test = ray.get(config["y_test_id"])


        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        f1 = f1_score(y_test, predictions, average='micro')

        return {"f1_score": f1}


