# Copyright (c) 2024 linjing-lab
'''
tests folder, test.py, and test_.py are executed on Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz
'''
import porgo
import time

objective_function = lambda x: (-13 + x[0] + ((5 - x[1])*x[1] - 2)*x[1])**2 + (-29 + x[0] + ((x[1] + 1)*x[1] - 14)*x[1])**2 + (-13 + x[2] + ((5 - x[3])*x[3] - 2)*x[3])**2 + (-29 + x[2] + ((x[3] + 1)*x[3] - 14)*x[3])**2
bounds = [(-10, 10)] * 4 # best converged point: [5, 4, 5, 4]

test = porgo.glos(objective_function, bounds) # mutation=0.5, recombination=0.9

test.rand_pop(40)
start = time.time()
test.train_gen(500)
end = time.time()
test.result()
print(test.best, test.best_fit)
test.result(True)
print("time={}".format(end-start))