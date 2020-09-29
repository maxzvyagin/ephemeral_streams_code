#### Test integration of Hyperspace package with Ray Tune in order to optimize hyperparameters in Pytorch Lightning ####

from ray.tune.suggest import Searcher

### Definition of custom search algorithm for Ray Tune
class HyperSearch(Searcher):
    def __init__(self, metric='avg_test_loss', mode='min', **kwargs):
        super(HyperSearch, self).__init__(metric=metrix, mode=mode, **kwargs)
        self.configurations = {}

    def suggest(self, trial_id):
        ### return a new set of parameters to try
        pass

    def on_trial_complete(self, trial_id, result, **kwargs):
        ## update the optimizer with the returned value
        pass


### hyperspace is a collection of scikit optimize Space objects with overlapping parameters
# generate the search space, it should output a list of the parameters to try
from hyperspace.space import create_hyperspace

hyperparameters = [(0.00000001, 0.1),  # learning_rate
                   (0.0, 0.9),  # dropout
                   (0.00000001, 0.1),  # weight decay
                   (1, 6)]  # encoder depth
space = create_hyperspace(hyperparameters)

first = space[0]

print(type(first.bounds))
print(first.bounds)

