# multimodal function

```python
import numpy
import porgo
multimodal_function = lambda x: numpy.sum(x**2 - 10 * numpy.cos(2 * numpy.pi * x)) + 10 * len(x)
bounds = [(-10, 10)] * 10 # high-dimensional search interval
multimodal = porgo.glos(multimodal_function, bounds) # mutation=0.5, recombination=0.9
multimodal.rand_pop(100) # init population
for i in range(3): # if need to execute 3 times
    multimodal.train_gen(1000)
    multimodal.result() # must be executed
    print(multimodal.mini, multimodal.fit_mini) # the best converged result
    # print('{} epoch, minimum {}, medium {}, maximum {}'.format(i, multimodal.fit_mini, multimodal.fit_medi, multimodal.fit_maxi))
'''
(array([1.74453074e-09,  3.97957495e-09, -1.02597317e-09,  2.87045889e-09, -2.40535489e-09, 
        1.93068978e-09, -2.73757802e-09,  1.10677787e-09, -4.19505055e-09,  1.19590320e-09]), 0.0)
'''
```

refer to [multimodal.ipynb](../tests/multimodal.ipynb).