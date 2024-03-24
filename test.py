# Copyright (c) 2024 linjing-lab
'''
tests folder and test.py are executed on Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz
'''
import numpy
import porgo
import time

objective_function = lambda x: (-13 + x[0] + ((5 - x[1])*x[1] - 2)*x[1])**2 + (-29 + x[0] + ((x[1] + 1)*x[1] - 14)*x[1])**2 + (-13 + x[2] + ((5 - x[3])*x[3] - 2)*x[3])**2 + (-29 + x[2] + ((x[3] + 1)*x[3] - 14)*x[3])**2
bounds = [(-10, 10)] * 4 # best converged point: [5, 4, 5, 4]
# objective_function = lambda x: -20*numpy.exp(-0.2*numpy.sqrt(numpy.sum(x**2)/len(x))) - numpy.exp(numpy.sum(numpy.cos(numpy.pi*x))/len(x)) + 22.71282
# bounds = [(-32.768, 32.768)] * 3 # best converged point: [-2.53914642e-17  3.08702991e-17 -3.28495269e-16]

test = porgo.glos(objective_function, bounds) # mutation=0.5, recombination=0.9

test.rand_pop(40)
start = time.time()
test.train_gen(500)
end = time.time()
test.result()
print(test.best, test.best_fit)
test.result(True)
print("time={}".format(end-start))