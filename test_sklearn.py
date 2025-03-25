from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import porgo
data, targets = make_classification(
        n_samples=1000,
        n_features=45,
        n_informative=12,
        n_redundant=7,
        random_state=0,
    )
def objective_function(x):
    '''
    x contains expC and expGamma
    '''
    C, gamma = 10 ** x[0], 10 ** x[1]
    estimator = SVC(C=C, gamma=gamma, random_state=0)
    cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=3)
    return 1 - cval.mean()
bounds = [[-3, 2], [-4, -1]]

test = porgo.glos(objective_function, bounds) # mutation=0.5, recombination=0.9

test.rand_pop(5)
for i in range(2):
    test.train_gen(10)
    test.result(minimum=True)
print('expC({}), expGamma {}, cval.mean() {}'.format(test.mini[0], test.mini[1], 1-test.fit_mini))