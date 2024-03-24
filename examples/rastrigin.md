# rastrigin function

```python
import numpy
import porgo
rastrigin_function = lambda x: 10 * len(x) + numpy.sum(x**2 - 10*numpy.cos(2*numpy.pi*x)) # non-convex function
bounds = [(-10, 10)] * 10 # high-dimensional search interval
rastrigin = porgo.glos(rastrigin_function, bounds) # mutation=0.5, recombination=0.9
rastrigin.rand_pop(100) # init population
for i in range(3): # if need to execute 3 times
    rastrigin.train_gen(1000)
rastrigin.result() # must be executed
print(rastrigin.best, rastrigin.best_fit) # the best converged result

'''
(array([-2.39950764e-09,  2.89384641e-09, -1.82451894e-09, -3.00804257e-09, 2.06450184e-09, 
        -1.23446967e-09,  9.17498501e-10,  3.33980914e-10, -1.62505039e-09, -3.78447686e-09]), 0.0)
'''
```

refer to [rastrigin.ipynb](../tests/rastrigin.ipynb).