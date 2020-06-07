"""Helper class to support passing a
list of dictionaries for hyperparameters
    -- Anthony Yu and Michael Chau
"""

from ray.tune.suggest.suggestion import Searcher
from sklearn.model_selection import ParameterGrid
import random


class ListSearcher(Searcher):
    """Custom search algorithm to support passing in a list of
    dictionaries to TuneGridSearchCV

    """

    def __init__(self, param_grid):
        self._configurations = list(ParameterGrid(param_grid))
        Searcher.__init__(self)

    def suggest(self, trial_id):
        if self._configurations:
            return self._configurations.pop(0)

    def on_trial_complete(self, **kwargs):
        pass


class RandomListSearcher(Searcher):
    """Custom search algorithm to support passing in a list of
    dictionaries to TuneSearchCV for randomized search

    """

    def __init__(self, param_grid):
        self._configurations = param_grid
        Searcher.__init__(self)

    def suggest(self, trial_id):
        selected_dict = self._configurations[random.randint(
            0,
            len(self._configurations) - 1)]
        generated_config = {}

        for key, distribution in selected_dict.items():
            if isinstance(distribution, list):
                generated_config[key] = distribution[random.randint(
                    0,
                    len(distribution) - 1)]
            else:
                generated_config[key] = distribution.rvs(1)[0]

        return generated_config

    def on_trial_complete(self, **kwargs):
        pass
