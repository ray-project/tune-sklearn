""" Helper class to train models using Ray backend
"""

import ray
from ray.tune import Trainable
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.model_selection import _aggregate_score_dicts
from sklearn.utils.metaestimators import _safe_split
import numpy as np
import os
from pickle import PicklingError
import ray.cloudpickle as cpickle
import warnings


class _Trainable(Trainable):
    """Class to be passed in as the first argument of tune.run to train models.

    Overrides Ray Tune's Trainable class to specify the setup, train, save,
    and restore routines.

    """

    def _setup(self, config):
        """Sets up Trainable attributes during initialization.

        Also sets up parameters for the sklearn estimator passed in.

        Args:
            config (dict): contains necessary parameters to complete the `fit`
                routine for the estimator. Also includes parameters for early
                stopping if it is set to true.

        """
        self.estimator = clone(config.pop("estimator"))
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

        if self.early_stopping:
            n_splits = self.cv.get_n_splits(self.X, self.y)
            self.fold_scores = np.zeros(n_splits)
            self.fold_train_scores = np.zeros(n_splits)
            for i in range(n_splits):
                self.estimator[i].set_params(**self.estimator_config)
        else:
            self.estimator.set_params(**self.estimator_config)

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
                X_train, y_train = _safe_split(self.estimator[i], self.X,
                                               self.y, train)
                X_test, y_test = _safe_split(
                    self.estimator[i],
                    self.X,
                    self.y,
                    test,
                    train_indices=train)
                self.estimator[i].partial_fit(X_train, y_train,
                                              np.unique(self.y))
                if self.return_train_score:
                    self.fold_train_scores[i] = {name: score(self.estimator[i], X_train, y_train) for name, score in self.scoring.items()}
                self.fold_scores[i] = {name: score(self.estimator[i], X_test,
                                                   y_test) for name, score in self.scoring.items()}

            ret = {}
            agg_fold_scores = _aggregate_score_dicts(self.fold_scores)
            for name, scores in agg_fold_scores:
                total = 0
                for i, score in enumerate(scores):
                    total += score
                    key_str = f"split{i}_test_%s" % name
                    ret[key_str] = score
                self.mean_score = total / len(scores)
                ret["average_test_%s" % name] = self.mean_score

            if self.return_train_score:
                agg_fold_train_scores = _aggregate_score_dicts(self.fold_train_scores)
                for name, scores in agg_fold_train_scores:
                    total = 0
                    for i, score in enumerate(scores):
                        total += score
                        key_str = f"split{i}_train_%s" % name
                        ret[key_str] = score
                    self.mean_train_score = total / len(scores)
                    ret["average_train_%s" % name] = self.mean_train_score

            return ret
        else:
            try:
                scores = cross_validate(
                    self.estimator,
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
                    self.estimator,
                    self.X,
                    self.y,
                    cv=self.cv,
                    fit_params=self.fit_params,
                    groups=self.groups,
                    scoring=self.scoring,
                    return_train_score=self.return_train_score,
                )

            ret = {}
            for name in self.scoring:
                for i, score in enumerate(scores["test_%s" % name]):
                    key_str = f"split{i}_test_%s" % name
                    ret[key_str] = score
                self.test_accuracy = sum(scores["test_%s" % name]) / len(
                    scores["test_%s" % name])
                ret["average_test_%s" % name] = self.test_accuracy

            if self.return_train_score:
                for name in self.scoring:
                    for i, score in enumerate(scores["train_%s" % name]):
                        key_str = f"split{i}_train_%s" % name
                        ret[key_str] = score
                    self.train_accuracy = sum(scores["train_%s" % name]) / len(
                        scores["train_%s" % name])
                    ret["average_train_%s" % name] = self.train_accuracy

            return ret

    def _save(self, checkpoint_dir):
        """Creates a checkpoint in ``checkpoint_dir``, creating a pickle file.

        Args:
            checkpoint_dir (str): file path to store pickle checkpoint.

        Returns:
            path (str): file path to the pickled checkpoint file.

        """
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "wb") as f:
            try:
                cpickle.dump(self.estimator, f)
                self.pickled = True
            except PicklingError:
                self.pickled = False
                warnings.warn("{} could not be pickled. "
                              "Restoring estimators may run into issues."
                              .format(self.estimator))
        return path

    def _restore(self, checkpoint):
        """Loads a checkpoint created from `save`.

        Args:
            checkpoint (str): file path to pickled checkpoint file.

        """
        if self.pickled:
            with open(checkpoint, "rb") as f:
                self.estimator = cpickle.load(f)
        else:
            warnings.warn("No estimator restored")

    def reset_config(self, new_config):
        return False
