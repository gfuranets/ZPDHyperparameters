from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ray import tune

DATASET_TEST_SPLIT = 0.2
DATASET_SPLIT_RANDOM_STATE = 420
TRAINING_RANDOM_STATE = 420

class ControllerEvaluator(tune.Trainable):
    def __init__(self, config):
        self.config = config

    def evaluate(self):
        # Load the datasets
        iris = load_iris()
        california = fetch_california_housing()

        # Split the datasets into training and test sets
        iris_train, iris_test, iris_train_labels, iris_test_labels = train_test_split(
            iris.data, iris.target, test_size=DATASET_TEST_SPLIT, random_state=DATASET_SPLIT_RANDOM_STATE)
        california_train, california_test, california_train_labels, california_test_labels = train_test_split(
            california.data, california.target, test_size=DATASET_TEST_SPLIT, random_state=DATASET_SPLIT_RANDOM_STATE)

        # Train and evaluate a RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=self.config["n_estimators"], 
            max_depth=self.config["max_depth"], 
            random_state=TRAINING_RANDOM_STATE
        )
        clf.fit(iris_train, iris_train_labels)
        iris_predictions = clf.predict(iris_test)
        iris_accuracy = accuracy_score(iris_test_labels, iris_predictions)

        # Train and evaluate a RandomForestRegressor
        reg = RandomForestRegressor(
            n_estimators=self.config["n_estimators"], 
            max_depth=self.config["max_depth"], 
            random_state=TRAINING_RANDOM_STATE
        )
        reg.fit(california_train, california_train_labels)
        california_predictions = reg.predict(california_test)
        california_mse = mean_squared_error(california_test_labels, california_predictions)

        # Calculate the mean_metric
        mean_metric = (iris_accuracy + california_mse) / 2
        return {"mean_metric": mean_metric}