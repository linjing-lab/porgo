# Copyright (c) 2024 linjing-lab

import numpy
import porgo

objective_function = lambda x: -20*numpy.exp(-0.2*numpy.sqrt(numpy.sum(x**2)/len(x))) - numpy.exp(numpy.sum(numpy.cos(numpy.pi*x))/len(x)) + 22.71282
bounds = [(-32.768, 32.768)] * 3 # best converged point: [-2.53914642e-17  3.08702991e-17 -3.28495269e-16]

test = porgo.glos(objective_function, bounds) # mutation=0.5, recombination=0.9

test.rand_pop(40)
for i in range(10):
    test.train_gen(100)
    test.result() # must be executed
    print('{} epoch, minimum {}, medium {}, maximum {}'.format(i, test.fit_mini, test.fit_medi, test.fit_maxi))
print(test.mini, test.fit_mini) # equal to (test.best, test.best_fit) in v1.0.0 and v1.0.1
# test.result(True)