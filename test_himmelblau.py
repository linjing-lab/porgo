# Copyright (c) 2025 linjing-lab

import porgo

himmelblau = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
def solver(bounds, pop: int=40, obs: int=6, gen: int=100):
    test = porgo.glos(himmelblau, bounds) # mutation=0.5, recombination=0.9
    test.rand_pop(pop)
    for i in range(obs):
        test.train_gen(gen)
        test.result()
        # print('{} epoch, minimum {}, medium {}, maximum {}'.format(i, test.fit_mini, test.fit_medi, test.fit_maxi))
    print(test.mini, test.fit_mini) # equal to (test.best, self.best_fit) in v1.0.0 and v1.0.1

pp = [(0, 10), (0, 10)] # positive, positive
nn = [(-10, 0), (-10, 0)] # negative, negative
pn = [(0, 10), (-10, 0)] # positive, negative
np = [(-10, 0), (0, 10)] # negative, positive
solver(pp) # [3. 2.] 0.0
solver(nn) # [-3.77931025 -3.28318599] 0.0
solver(pn) # [ 3.58442834 -1.84812653] 0.0
solver(np) # [-2.80511809  3.13131252] 0.0