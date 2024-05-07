# porgo

When I was researching a function without given all local minima, like the underlined function:

$$
f(x)=\sum_{i=1}^{n/2}(-13+x_{2i-1}+((5-x_{2i})x_{2i}-2)x_{2i})^2+(-29+x_{2i-1}+((x_{2i}+1)x_{2i}-14)x_{2i})^2.
$$

I used `optimtool.unconstrain` to search local minima, got an efficient experience about searching the nearest minimum point. Add a mechanism to jump out of the local area would increase the runtime of the whole script, so `porgo` is a new progam to accelerate to search global minima.

refer to test.py and the global minima of 4-dimensional $f(x)$ is (5, 4, 5, 4).

## glos

glos is the main runtime to serve as a global search class, users can run train_gen module with given cycles at any times until the function searching process converged.

init:
- objective_function: *Callable*, a high-dimensional function with convex, non-convex, and many local minima.
- bounds: *List[List[float]] | List[Tuple[float]]*, changes this value makes a significant influence of mini and fit_mini.
- mutation: *float=0.5*, increase this value makes the search radius larger.
- recombination: *float=0.9*, increase this value allows larger number of mutation.

rand_pop:
- population_size: *int=50*, randomly init the population (or called initial points) with shape at (population, dimension).
- verbose: *bool=False*, whether to output initial population when manually replace the random generated rule.

train_gen:
- cycles: *int=1000*, try to run several times (until converged) when give a smaller cycle number if search bounds is in large space.

result:
- verbose: *bool=False*, whether to output console information after search populations were updated (check self.mini and self.fit_mini, the top3 updated results are (self.mini, self.fit_mini) < (self.medi, self.fit_medi) < (self.maxi, self.fit_maxi)).

## reference

Storn, R and Price, K, Differential Evolution - a Simple and Efficient Heuristic for Global Optimization over Continuous Spaces, Journal of Global Optimization, 1997, 11, 341 - 359.

## LICENSE

[MIT LICENSE](./LICENSE)