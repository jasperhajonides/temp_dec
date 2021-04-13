#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:09:49 2021

@author: jasperhvdm
"""


import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import softmax
from temp_dec import decoding_functions
from scipy.special import expit
#%%
class CustomEstimator(BaseEstimator):
    """
    A wrapper for scikit-learn estimators 

    Parameters
    ----------
    est_class
        The estimator class to use
    **kwargs
        Keyword arguments to initialize the estimator
    """

    def __init__(self, n_bins = 8, est_class=LinearDiscriminantAnalysis(), **kwargs):

        self.est_class = est_class
        self.n_bins = n_bins

        # kwargs depend on the model used, so assign them whatever they are
        for key, value in kwargs.items():
            setattr(self, key, value)

        # these attributes support the logging functionality
        self._param_names = ['est_class'] + list(kwargs.keys())

    # in the transformer case, we did not implement get_params
    # nor set_params since we inherited them from BaseEstimator
    # but such implementation will not work here due to the **kwargs
    # in the constructor, so we implemented it

    def get_params(self, deep=True):
        # Note: we are ignoring the deep parameter
        # this will not work with estimators that have sub-estimators
        # see https://scikit-learn.org/stable/developers/develop.html#get-params-and-set-params
        return {param: getattr(self, param)
                for param in self._param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    # our fit method instantiates the actual model, and
    # it forwards any extra keyword arguments
    def fit(self, X, y, **kwargs):
        est_kwargs = self.get_params()
        del est_kwargs['est_class']
        # remember the trailing underscore
        self.model_ = self.est_class(**est_kwargs)
        self.model_.fit(X, y, **kwargs)
        # fit must return self
        return self

    
    def predict(self, X):
        check_is_fitted(self)

        # we use the fitted model and log if logging is enabled
        y_pred = self.model_.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        check_is_fitted(self)
        decision = self.decision_function(X)
        if self.classes_.size == 2:
            proba = expit(decision)
            return np.vstack([1-proba, proba]).T
        else:
            return softmax(decision)
        
    def predict_conv(self, X, y):
        check_is_fitted(self)
        decision = self.decision_function(X)
        if self.classes_.size == 2:
            proba = expit(decision)
            return np.vstack([1-proba, proba]).T
        else:
            matrix = softmax(decision)
            nbins = len(set(y))
            theta = np.arange(-math.pi-0.000001,math.pi - (2*math.pi/nbins),(2*math.pi/nbins))
            matrix_shift = np.zeros((len(matrix),nbins))
            for val in range(0,matrix.shape[0]):
                matrix_shift[val,:] = np.roll(matrix[val,:],math.floor(nbins/2)-int(y[val]))
            return np.sum(np.cos(theta)*matrix_shift,axis=1)
        
    def predict_conv_proba(self, X, y):
        check_is_fitted(self)
        decision = self.decision_function(X)
        if self.classes_.size == 2:
            proba = expit(decision)
            return np.vstack([1-proba, proba]).T
        else:
            matrix = softmax(decision)
            nbins = len(set(y))
            theta = np.arange(-math.pi-0.000001,math.pi - (2*math.pi/nbins),(2*math.pi/nbins))
            matrix_shift = np.zeros((len(matrix),nbins))
            for val in range(0,matrix.shape[0]):
                matrix_shift[val,:] = np.roll(matrix[val,:],math.floor(nbins/2)-int(y[val]))
            return np.cos(theta)*matrix_shift
        
        

    # requiring a score method is not documented but throws an
    # error if not implemented
    def score(self, X, y, **kwargs):

        y_class_pred = self.model_.predict_proba(X)
        ysh = decoding_functions.matrix_vector_shift(y_class_pred,y,12)
        centered_prediction = ysh.mean(0)
        return decoding_functions.convolve_matrix_with_cosine(centered_prediction[:,None])
#         return accuracy_score(y, self.model_.predict(X))
    

    # some models implement custom methods. Anything that is not implemented here
    # will be delegated to the underlying model. There is one condition we have
    # to cover: if the underlying estimator has class attributes they won't
    # be accessible until we fit the model (since we instantiate the model there)
    # to fix it, we try to look it up attributes in the instance, if there
    # is no instance, we look up the class. More info here:
    # https://scikit-learn.org/stable/developers/develop.html#estimator-types
    def __getattr__(self, key):
        if key != 'model_':
            if hasattr(self, 'model_'):
                return getattr(self.model_, key)
            else:
                return getattr(self.est_class, key)
        else:
            raise AttributeError(
                "'{}' object has no attribute 'model_'".format(type(self).__name__))
    
        
# https://ploomber.io/posts/sklearn-custom/
 
 