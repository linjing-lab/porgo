# Copyright (c) 2024 linjing-lab
'''
tests folder, test.py, and test_.py are executed on Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz
'''
import porgo

objective_function = lambda x: (-13 + x[0] + ((5 - x[1])*x[1] - 2)*x[1])**2 + (-29 + x[0] + ((x[1] + 1)*x[1] - 14)*x[1])**2 + (-13 + x[2] + ((5 - x[3])*x[3] - 2)*x[3])**2 + (-29 + x[2] + ((x[3] + 1)*x[3] - 14)*x[3])**2
bounds = [(-10, 10)] * 4 # best converged point: [5, 4, 5, 4]

test = porgo.glos(objective_function, bounds) # mutation=0.5, recombination=0.9

test.rand_pop(40)
for i in range(6):
    test.train_gen(100)
    test.result() # must be executed
    print('{} epoch, minimum {}, medium {}, maximum {}'.format(i, test.fit_mini, test.fit_medi, test.fit_maxi))
print(test.mini, test.fit_mini) # equal to (test.best, self.best_fit) in v1.0.0 and v1.0.1
# test.result(True)