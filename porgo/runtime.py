# Copyright (c) 2024 linjing-lab

import numpy
from ._typing import Callable, Union, List, Tuple, NUMPY_NDARRAY

class glos:
    '''
    The solver program for global search between bounds.
    :param objective_function: Callable. lambda function or function with x-like value inputed.
    :param bounds: List[List[float]] | List[Tuple[float]]. search interval given by manually set.
    :param mutation: float. increase the search radius by increasing the mutation constant. default: 0.5.
    :param recombination: float. allows a larger number of mutants by increasing this value. default: 0.9.
    '''
    def __init__(self, objective_function: Callable,
                       bounds: Union[List[Union[List[float], Tuple[float]]]],
                       mutation: float=0.5,
                       recombination: float=0.9) -> None:
        assert mutation > 0.4 and mutation < 1
        assert recombination > 0 and recombination < 1
        self.obj = objective_function
        self.scale = len(bounds) # the dimension of objective_function
        assert self.scale > 0
        assert all(map(lambda x: isinstance(x, (List, Tuple)), bounds))
        if not isinstance(bounds, NUMPY_NDARRAY):
            self.bounds = numpy.array(bounds)
        assert numpy.all(self.bounds[:,0] < self.bounds[:,1])
        self.F = mutation
        self.CR = recombination

    def _tri_cr(self, pop_ind: int) -> None:
        '''
        Update target population by `self.CR`.
        :param pop_ind: int. the sequential population number.
        '''
        mask = numpy.random.rand(self.scale) < self.CR
        if not numpy.any(mask):
            mask[numpy.random.randint(0, self.scale)] = True
        trial = numpy.where(mask, self.mut, self.uniform[pop_ind])
        if self.obj(trial) < self.obj(self.uniform[pop_ind]):
            self.uniform[pop_ind] = trial

    def _rand_mut(self) -> None:
        '''
        Use `numpy.random.choice` to randomly select the values of `self.uniform`.
        '''
        ma, mb, mc = self.uniform[numpy.random.choice(self.population_size, 3, replace=False)]
        self.mut = ma + self.F * (mb - mc) # the portable solution
        self.mut = numpy.clip(self.mut, self.bounds[:,0], self.bounds[:,1])

    def _evo_pop(self) -> None:
        '''
        Update best population values from `self.uniform`.
        '''
        for pop_ind in range(self.population_size):
            self._rand_mut()
            self._tri_cr(pop_ind)

    def rand_pop(self, population_size: int=50, verbose: bool=False) -> None:
        '''
        Use `random` module to generate the range distribution of population.
        :param population_size: int. the size of population. default: 50.
        :param verbose: bool. whether to output initial population. default: False.
        '''
        assert population_size > 0
        self.population_size: int = population_size
        self.uniform = numpy.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.population_size, self.scale)) # replace with another initial pattern
        if verbose:
            print(self.uniform)

    def train_gen(self, cycles: int=1000):
        '''
        Configure the numbers of generations to accelerate convergence.
        :param cycles: int. set the times of train_gen. default: 1000.
        '''
        assert cycles > 0
        for _ in range(cycles): 
            self._evo_pop()

    def result(self, minimum: bool=False, verbose: bool=False) -> None:
        '''
        Find the top 3 updated results or output console information with `verbose=True`.
        :param minimum: bool. whether to output first if struct with mini and fit_mini. default: False.
        :param verbose: bool. whether to output console information with 'index point value'. default: False.
        '''
        if not verbose:
            top3_index = numpy.argsort([self.obj(updated) for updated in self.uniform])[:3]
            self.mini, self.medi, self.maxi = self.uniform[top3_index]
            self.fit_mini, self.fit_medi, self.fit_maxi = self.obj(self.mini), self.obj(self.medi), self.obj(self.maxi)
            if minimum:
                print("{}, {}".format(self.mini, self.fit_mini))
        else:
            for i, updated in enumerate(self.uniform):
                print('{}\t{}\t{}'.format(i, updated, self.obj(updated)))
