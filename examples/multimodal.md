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
    multimodal.result(minimum=True)
    # print('{} epoch, minimum {}, medium {}, maximum {}'.format(i, multimodal.fit_mini, multimodal.fit_medi, multimodal.fit_maxi))
```

refer to [multimodal.ipynb](../tests/multimodal.ipynb).