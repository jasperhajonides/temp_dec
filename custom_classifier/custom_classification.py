#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:09:49 2021

@author: jasperhvdm
"""



import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.utils.extmath import softmax

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

    def __init__(self, est_class=LinearDiscriminantAnalysis(), **kwargs):

        self.est_class = est_class

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
                matrix_shift[val,:] = np.roll(matrix[val,:],math.floor(nbins/2)-y[val])
            return np.sum(np.cos(theta)*matrix_shift,axis=1)
        

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
 
 #%%
 
 X,y=datasets.load_wine(return_X_y=True)
 
 X_train, X_test, y_train, y_test = train_test_split(
  	      X, y, test_size=0.2, random_state=42) #X[:,:,110:180].mean(2)
  
  
#%%

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression

pipe = Pipeline([('pca',PCA()),
                 ('scaler', StandardScaler()),
                 ('reg', CustomEstimator(est_class=LinearDiscriminantAnalysis))])

# perform hyperparameter tuning
grid = GridSearchCV(pipe, param_grid={'pca__n_components': [10,8,5]}, 
                    n_jobs=-1,scoring='accuracy') #'reg__n_clusters': [10],

best_pipe = grid.fit(X_train, y_train).best_estimator_

# make predictions using the best model
#y_predp = best_pipe.predict_proba(X_test)
y_pred = best_pipe.predict(X_test)

print(grid.best_params_)
print(f'MAE: {np.abs(y_test - y_pred).mean():.2f}')
print('accuracy score: ' + str(accuracy_score(y_test,y_pred)))
#%%




from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

#%%
from sklearn.utils.validation import check_is_fitted

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
size_window=20


acc = np.zeros(128)
cos_acc = np.zeros(128)
pca_comp = np.zeros((128,5))


[n_trials,n_features,n_time] = X_all[:,:,:].shape

y=thetas[:,0]

for tp in range(30,35):
    y_class_pred = np.zeros((n_trials,12))
    X_demeaned = np.zeros((n_trials,n_features,size_window)) * np.nan

    cc=0
    for s in range((1-size_window),1):
        X_demeaned[:,:,cc] = X_all[:,:,tp+s] - X_all[:,:,(tp-(size_window-1)):(tp+1)].mean(2)
        cc=cc+1
    # reshape into trials by features*time
    X = X_demeaned.reshape(X_demeaned.shape[0],X_demeaned.shape[1]*X_demeaned.shape[2])

    ccc=0
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        pipe = Pipeline([('pca',PCA()),
                         ('scaler', StandardScaler()),
                         ('reg', CustomEstimator(est_class=LinearDiscriminantAnalysis))])

        # perform hyperparameter tuning
        grid = GridSearchCV(pipe, param_grid={'pca__n_components': [.999,.95,.9]}, 
                            n_jobs=-1,scoring='accuracy',cv=10) #'reg__n_clusters': [10],

        best_pipe = grid.fit(X_train, y_train).best_estimator_

        y_class_pred[test_index,:] = best_pipe.predict_conv(X_test,y_test)
        pca_comp[tp,ccc] = grid.best_params_['pca__n_components']
        ccc += 1
    cos_acc[tp] = y_class_pred.mean(0)
    
    print('cos accuracy score: ' + str(cos_acc[tp]*1000) + '  pca:' +  str(pca_comp[tp,:].mean(0)))


