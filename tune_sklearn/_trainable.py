""" Helper class to train models using Ray backend
"""

import ray
from ray.tune import Trainable
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.utils.metaestimators import _safe_split
from lightgbm import LGBMModel
import numpy as np
import os
from pickle import PicklingError
import ray.cloudpickle as cpickle
import warnings


def try_import_xgboost():
    try:
        import xgboost  # ignore: F401
        return True
    except ImportError:
        return False


class _Trainable(Trainable):
    """Class to be passed in as the first argument of tune.run to train models.

    Overrides Ray Tune's Trainable class to specify the setup, train, save,
    and restore routines.

    """
    estimator_list = None

    def _can_partial_fit(self):
        return hasattr(self.main_estimator, "partial_fit")

    @property
    def main_estimator(self):
        return self.estimator_list[0]

    @property
    def is_xgb(self):
        if try_import_xgboost():
            from xgboost.sklearn import XGBModel
            return isinstance(self.main_estimator, XGBModel)
        return False

    def setup(self, config):
        # forward-compatbility
        self._setup(config)

    def _setup(self, config):
        """Sets up Trainable attributes during initialization.

        Also sets up parameters for the sklearn estimator passed in.

        Args:
            config (dict): contains necessary parameters to complete the `fit`
                routine for the estimator. Also includes parameters for early
                stopping if it is set to true.

        """
        self.estimator_list = clone(config.pop("estimator_list"))
        self.early_stopping = config.pop("early_stopping")
        X_id = config.pop("X_id")
        self.X = ray.get(X_id)

        y_id = config.pop("y_id")
        self.y = ray.get(y_id)
        self.groups = config.pop("groups")
        self.fit_params = config.pop("fit_params")
        self.scoring = config.pop("scoring")
        self.max_iters = config.pop("max_iters")
        self.cv = config.pop("cv")
        self.return_train_score = config.pop("return_train_score")
        self.n_jobs = config.pop("n_jobs")
        self.estimator_config = config
        self.pickled = False

        if self.early_stopping:
            n_splits = self.cv.get_n_splits(self.X, self.y)
            self.fold_scores = np.zeros(n_splits)
            self.fold_train_scores = np.zeros(n_splits)
            if not self._can_partial_fit():
                # max_iter here is different than the max_iters the user sets.
                # max_iter is to make sklearn only fit for one epoch,
                # while max_iters (which the user can set) is the usual max
                # number of calls to _trainable.
                self.estimator_config["warm_start"] = True
                self.estimator_config["max_iter"] = 1
            for i in range(n_splits):
                self.estimator_list[i].set_params(**self.estimator_config)

            if self.is_xgb:
                self.saved_models = [None for _ in range(n_splits)]
        else:
            self.main_estimator.set_params(**self.estimator_config)

    def step(self):
        # forward-compatbility
        return self._train()

    def _train(self):
        """Trains one iteration of the model called when ``tune.run`` is called.

        Different routines are run depending on if the ``early_stopping``
        attribute is True or not.

        If ``self.early_stopping`` is not None, each fold is fit with
        `partial_fit`, which stops training the model if the validation
        score is not improving for a particular fold.

        Otherwise, run the full cross-validation procedure.

        In both cases, the average test accuracy is returned over all folds,
        as well as the individual folds' accuracies as a dictionary.

        Returns:
            ret (:obj:`dict): Dictionary of results as a basis for
                ``cv_results_`` for one of the cross-validation interfaces.

        """
        if self.early_stopping:
            for i, (train, test) in enumerate(self.cv.split(self.X, self.y)):
                X_train, y_train = _safe_split(self.estimator_list[i], self.X,
                                               self.y, train)
                X_test, y_test = _safe_split(
                    self.estimator_list[i],
                    self.X,
                    self.y,
                    test,
                    train_indices=train)
                if self._can_partial_fit():
                    if self.is_xgb:
                        self.estimator_list[i].fit(
                            X_train, y_train, xgb_model=self.saved_models[i])
                        self.saved_models[i] = self.estimator_list[i].get_booster()
                    else:
                        self.estimator_list[i].partial_fit(X_train, y_train,
                                                      np.unique(self.y))
                else:
                    self.estimator_list[i].fit(X_train, y_train)

                if self.return_train_score:
                    self.fold_train_scores[i] = self.scoring(
                        self.estimator_list[i], X_train, y_train)
                self.fold_scores[i] = self.scoring(self.estimator_list[i],
                                                   X_test, y_test)

            ret = {}
            total = 0
            for i, score in enumerate(self.fold_scores):
                total += score
                key_str = f"split{i}_test_score"
                ret[key_str] = score
            self.mean_score = total / len(self.fold_scores)
            ret["average_test_score"] = self.mean_score

            if self.return_train_score:
                total = 0
                for i, score in enumerate(self.fold_train_scores):
                    total += score
                    key_str = f"split{i}_train_score"
                    ret[key_str] = score
                self.mean_train_score = total / len(self.fold_train_scores)
                ret["average_train_score"] = self.mean_train_score

            return ret
        else:
            try:
                scores = cross_validate(
                    self.main_estimator,
                    self.X,
                    self.y,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    fit_params=self.fit_params,
                    groups=self.groups,
                    scoring=self.scoring,
                    return_train_score=self.return_train_score,
                )
            except PicklingError:
                warnings.warn("An error occurred in parallelizing the cross "
                              "validation. Proceeding to cross validate with "
                              "one core.")
                scores = cross_validate(
                    self.main_estimator,
                    self.X,
                    self.y,
                    cv=self.cv,
                    fit_params=self.fit_params,
                    groups=self.groups,
                    scoring=self.scoring,
                    return_train_score=self.return_train_score,
                )

            ret = {}
            for i, score in enumerate(scores["test_score"]):
                key_str = f"split{i}_test_score"
                ret[key_str] = score
            self.test_accuracy = sum(scores["test_score"]) / len(
                scores["test_score"])
            ret["average_test_score"] = self.test_accuracy

            if self.return_train_score:
                for i, score in enumerate(scores["train_score"]):
                    key_str = f"split{i}_train_score"
                    ret[key_str] = score
                self.train_accuracy = sum(scores["train_score"]) / len(
                    scores["train_score"])
                ret["average_train_score"] = self.train_accuracy

            return ret

    def save_checkpoint(self, checkpoint_dir):
        # forward-compatbility
        return self._save(checkpoint_dir)

    def _save(self, checkpoint_dir):
        """Creates a checkpoint in ``checkpoint_dir``, creating a pickle file.

        Args:
            checkpoint_dir (str): file path to store pickle checkpoint.

        Returns:
            path (str): file path to the pickled checkpoint file.

        """
        path = os.path.join(checkpoint_dir, "checkpoint")
        try:
            with open(path, "wb") as f:
                cpickle.dump(self.estimator_list, f)
        except Exception:
            warnings.warn("Unable to save estimator.")
        return path

    def load_checkpoint(self, checkpoint):
        # forward-compatbility
        return self._restore(checkpoint)

    def _restore(self, checkpoint):
        """Loads a checkpoint created from `save`.

        Args:
            checkpoint (str): file path to pickled checkpoint file.

        """
        try:
            with open(checkpoint, "rb") as f:
                self.estimator_list = cpickle.load(f)
        except Exception:
            warnings.warn("No estimator restored")

    def reset_config(self, new_config):
        self.config = new_config
        self._setup(new_config)
        return True
