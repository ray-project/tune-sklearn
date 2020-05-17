from ray.tune.suggest import Searcher


class ListSearcher(Searcher):
    def __init__(self, list_of_params):
        self._configurations = list_of_params
        Searcher.__init__(self)

    def suggest(self, trial_id):
        if self._configurations:
            return self._configurations.pop(0)
        else:
            self.set_finished()

    def on_trial_complete(self, **kwargs):
        pass
