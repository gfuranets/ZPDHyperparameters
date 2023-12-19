from ControllerEvaluator import ControllerEvaluator
import ray
from ray import tune

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

if __name__ == "__main__":
    main()