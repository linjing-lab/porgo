import numpy as np
from bayes_opt import BayesianOptimization
def objective_function(x):
    return np.exp(-(x-2)**2) + np.exp(-(x-6)**2/10) + 1/(x**2+1)
pbounds = {'x': (-2, 10)}
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=1,
)
optimizer.maximize(
    init_points=5,
    n_iter=10,
)
print(optimizer.max)