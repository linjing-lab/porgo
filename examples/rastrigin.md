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
    rastrigin.result(minimum=True) # must be executed
    # print('{} epoch, minimum {}, medium {}, maximum {}'.format(i, rastrigin.fit_mini, rastrigin.fit_medi, rastrigin.fit_maxi))
```

refer to [rastrigin.ipynb](../tests/rastrigin.ipynb).