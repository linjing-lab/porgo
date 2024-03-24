# Copyright (c) 2024 linjing-lab

import numpy
import porgo
import time

objective_function = lambda x: -20*numpy.exp(-0.2*numpy.sqrt(numpy.sum(x**2)/len(x))) - numpy.exp(numpy.sum(numpy.cos(numpy.pi*x))/len(x)) + 22.71282
bounds = [(-32.768, 32.768)] * 3 # best converged point: [-2.53914642e-17  3.08702991e-17 -3.28495269e-16]

test = porgo.glos(objective_function, bounds) # mutation=0.5, recombination=0.9

test.rand_pop(40)
start = time.time()
test.train_gen(500)
end = time.time()
test.result()
print(test.best, test.best_fit)
test.result(True)
print("time={}".format(end-start))