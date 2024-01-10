from HyperParamOptimizer import compare

import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "./results/")

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    compare(RESULTS_DIR)
