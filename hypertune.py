#### Test integration of Hyperspace package with Ray Tune in order to optimize hyperparameters in Pytorch Lightning ####

from ray.tune.suggest import Searcher

### dig into hyperdrive implementation and pull out the section where new values are suggested

# generate the search space, it should output a list of the parameters to try
from hyperspace.space import create_hyperspace

hyperparameters = [(0.00000001, 0.1),  # learning_rate
                   (0.0, 0.9),  # dropout
                   (0.00000001, 0.1),  # weight decay
                   (1, 6)]  # encoder depth
space = create_hyperspace(hyperparameters)

print(type(space))
