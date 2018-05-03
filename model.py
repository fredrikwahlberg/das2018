# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.gpc import _BinaryGaussianProcessClassifierLaplace as BinaryGPC
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C

__all__ = ['SharedKernelClassifier']


class SharedKernelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=100, kernel='rbf', ard=True, ardinit=True,
                 n_restarts=0, model_batch_size=None, verbose=False):
        # Check and store parameters
        assert n_iter > 0
        assert n_restarts >= 0
        assert kernel in ['rbf', 'matern52', 'matern32']
        assert type(ard) is bool
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.kernel = kernel
        self.ard = ard
        self.ardinit = ardinit
        self.verbose = verbose
        self.model_batch_size = model_batch_size
        # Container for the sub models
        self.models_ = dict()
        # Stores likelihoods of optimizations
        self.convergence_ = list()

    @property
    def classes_(self):
        return list(self.models_.keys())

    @property
    def log_likelihood_(self):
        likelihood = list()
        for m in self.models_.values():
            likelihood.append(m.log_marginal_likelihood())
        return np.mean(likelihood)

    def _kernel_factory(self, X, theta):
        """Factory for creating a kernel"""
        n_samples, n_features = X.shape
        if self.ard:
            lengthscale = np.ones(n_features)
        else:
            lengthscale = 1.0
        if self.kernel == 'rbf':
            k = C(1.0) * RBF(length_scale=lengthscale)
        elif self.kernel == 'matern32':
            k = C(1.0) * Matern(nu=1.5, length_scale=lengthscale)
        elif self.kernel == 'matern52':
            k = C(1.0) * Matern(nu=2.5, length_scale=lengthscale)
        else:
            raise RuntimeError("Unknown kernel")
        if theta is not None:
            theta = np.asarray(theta).copy()
            assert theta.shape == k.theta.shape
            k.theta = theta
        return k

    def _estimator_factory(self, X, y, theta):
        """Factory for creating a binary estimator"""
        k = self._kernel_factory(X, theta)
        estimator = BinaryGPC(kernel=k, optimizer=None, copy_X_train=False)
        # copy_X_train=False saves memory by not copying the feature data
        # optimizer=None only initializes the model without training
        estimator.fit(X, y)
        return estimator

    def _oneVsAllSplit(self, y):
        """Perform one-vs-all binary vector encoding"""
        one_vs_all_y = dict()
        for c in np.unique(y):
            one_vs_all_y[c] = np.vstack(np.asarray(y == c, dtype=np.int)).ravel()
        return one_vs_all_y

    def _init_sub_models(self, X, y, theta=None):
        # Encode y as one-vs-all binary vectors
        one_vs_all_y = self._oneVsAllSplit(y)
        # Add or replace models for classes in y
        for c in one_vs_all_y.keys():
            self.models_[c] = self._estimator_factory(X, one_vs_all_y[c], theta)

    def fit(self, X, y):
        # Save a reference to the training data
        self._X = X
#        self._y = y
        # Initialize the models
        self._init_sub_models(X, y)
        assert self.model_batch_size is None or self.model_batch_size <= len(self.models_.keys())
        assert len(self.classes_) > 0
        # Run optimization with restarts
        for restart in range(1 + self.n_restarts):
            if restart>0 and self.verbose:
                print("restarting optimization")
            # Randomize initial hyperparameters
            k = self._kernel_factory(X, theta=None)
            
            x0 = np.random.uniform(
                            low=k.bounds[:, 0], 
                            high=k.bounds[:, 1], 
                            size=k.theta.shape)
#            x0 = np.log(np.random.uniform(
#                            low=np.exp(k.bounds[:, 0]), 
#                            high=np.exp(k.bounds[:, 1]), 
#                            size=k.theta.shape))
            if self.ard and self.ardinit:
                if self.verbose:
                    print("initializing ard")
                from copy import deepcopy
                modelcopy = deepcopy(self)
                modelcopy.ard = False
                modelcopy.ardinit = False
                modelcopy.n_iter = 5
                modelcopy.n_restarts = 0
                modelcopy.fit(X, y)
                if self.verbose:
                    print("ard init hyper ", modelcopy.hyperparameters_)
                x0[:] = modelcopy.hyperparameters_[1]
                x0[0] = modelcopy.hyperparameters_[0]
            # Define optimization bounds
            theta_bounds = [tuple(k.bounds[i, :]) for i in range(k.bounds.shape[0])]
            # Create list for storing converngence information
            log_likelihood_convergence = list()
            # Define objective function to _minimize_
            self._optimizer_iteration = 0
            def inc_optimizer_iteration(theta):
                self._optimizer_iteration += 1
            def f(theta):
                likelihood = list()
                gradient = list()
                if self.model_batch_size is not None:
                    keys = random.sample(self.models_.keys(), 
                                         self.model_batch_size)
                else:
                    keys = self.models_.keys()
                for k in keys:
                    lml, grad = self.models_[k].log_marginal_likelihood(theta, 
                                                          eval_gradient=True)
                    likelihood.append(np.exp(lml))
                    gradient.append(grad)
                likelihood = np.log(np.mean(likelihood))
                log_likelihood_convergence.append(likelihood)
                gradient = np.mean(np.stack(gradient), axis=0)
                if self.verbose:
                    print("%i| log likelihood: %.6f" % (self._optimizer_iteration, likelihood))
                return -likelihood, -gradient
            theta, likelihood, flags = fmin_l_bfgs_b(f, x0,
                                      bounds=theta_bounds, 
                                      maxiter=self.n_iter, 
                                      disp=None, callback=inc_optimizer_iteration)
            self.convergence_.append(log_likelihood_convergence)
            # Select the best run
            if restart == 0 or self.log_likelihood_ < -likelihood:
                self.hyperparameters_ = theta.copy()
        # Re-init all sub models with the best hyper parameters
        self._init_sub_models(X, y, theta=self.hyperparameters_)
        return self

    def get_kernel(self):
        """Return a kernel with the estimated hyperparameters"""
        return self._kernel_factory(self._X, theta=self.hyperparameters_)

    def _models_probability_matrix(self, X):
        """Returns non-normalized probabilities of X per model

        Returns
        -------
        C : array-like, shape = (n_samples, n_classes)
        """
        n_samples, n_features = X.shape
        prob = np.zeros((n_samples, len(self.classes_)))
        for idx, cla in enumerate(self.classes_):
            prob[:, idx] = self.models_[cla].predict_proba(X)[:, 1].ravel()
        return prob

    def predict(self, X):
        """Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
        Returns
        -------
        C : array, shape = (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """
        prob = self.predict_proba(X)
        classes = list(self.classes_)
        return [classes[idx] for idx in np.argmax(prob, axis=1)]

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
        Returns
        -------
        C : array-like, shape = (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        probs = self._models_probability_matrix(X)
        # Normalize per column
        for i in range(probs.shape[0]):
            probs[i, :] = probs[i, :] / np.sum(probs[i, :])
        return probs

    def score_covar_ntop(self, X, y):
        """
        """
        return self._affinity_ntop(affinity_matrix=self.get_kernel()(X), labels=y)

    def _affinity_ntop(self, affinity_matrix, labels):
        sorted_K = np.argsort(-affinity_matrix, axis=0)
        labeld_K = labels[sorted_K]
        m = 5
        ntop = [0]*(2*m-1)
        for n in range(1, m+1):
            soft_ntop = 0
            hard_ntop = 0
            for i in range(len(labels)):
                matches = labeld_K[1:n+1, i] == labeld_K[0, i]
                if np.sum(matches) > 0:
                    soft_ntop += 1/len(labels)
                if np.all(matches):
                    hard_ntop += 1/len(labels)
            ntop[m+n-2] = soft_ntop
            ntop[m-n] = hard_ntop
        return ntop
