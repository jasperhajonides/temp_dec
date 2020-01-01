#!/usr/bin/python
# -*- coding: utf-8 -*-

# /Users/jasperhvdm/Documents/Scripts/EEG_feature_object/Functions/decoding_functions.py
# usage:
"""
import sys
sys.path.append('/Users/jasperhvdm/Documents/Scripts/EEG_feature_object/Functions')
from decoding_functions import *
"""
# author: JE Hajonides Oct 2019


import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import progressbar
from sklearn.decomposition import PCA


def matrix_vector_shift(matrix,vector,n_bins):

	""" Shift rows of a matrix by the amount of columns specified 
		in the corresponding cell of the vector.
	
	e.g. M =0  1  0     V = 0 0 1 2     M_final =   0 1 0 
			0  1  0									0 1 0
			1  0  0									0 1 0
			0  0  1									0 1 0 """
	r,c = matrix.shape
	matrix_shift = np.zeros((r,c))
	for row in range(0,r):
		matrix_shift[row,:] = np.roll(matrix[row,:],round(n_bins/2)-vector[row])
	return matrix_shift
    
    
    
    
def temporal_decoding(X_all,y,time,n_bins=12,size_window=5,n_folds=5,classifier='LDA',use_pca=False,pca_components=.95,temporal_dymanics=True):
    """
    This function takes a time course and uses a sliding window approach to 
    decode the feature of interest. We reshape the window from trials x channels x 
    time to trials x channels*time and demean every channel 

    Parameters
    ----------
    X_all : ndarray
            trials by channels by time 
    y     : ndarray
            vector of labels of each trial
    time  : ndarray
            vector with time labels locked to the cue
    bins  : integer
            how many stimulus classes do you have in total?
    size_window :  integer
            size of the sliding window
    n_folds :  integer
            folds of cross validation
    classifier : string
            Classifier used for decoding
            options:
            - LDA: LinearDiscriminantAnalysis
            - LG: LogisticRegression
    use_pca     : {bool, integer}  
    		Apply PCA or not     
    pca_components   : integer
            reduce features to N principal components, if N < 1 it indicates the % of explained variance
    temporal_dynamics : {bool, integer}
            use sliding window (default is True), if false its just single time-point decoding

    Returns
    --------
    accuracy : ndarray
            matrix containing class predictions for each time point. 
    """
    [n_trials,n_channels,n_time] = X_all[:,:,:].shape

    prediction = np.zeros(([n_trials,n_bins,n_time])) * np.nan
    label_pred = np.zeros(([n_trials,n_time])) * np.nan
    accuracy   = np.zeros(n_time) * np.nan
    centered_prediction = np.zeros(([n_bins,n_time])) * np.nan
    X_demeaned = np.zeros((n_trials,n_channels,size_window)) * np.nan
    #progressbar 
    bar = progressbar.ProgressBar(maxval=n_time,\
         	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()


    for tp in range((size_window-1),n_time):
        bar.update(tp+1)

        #demean channels
        if temporal_dymanics == True:
            for s in range((1-size_window),1): 
                X_demeaned[:,:,s] = X_all[:,:,tp+s] - X_all[:,:,(tp-(size_window-1)):(tp+1)].mean(2)
            X = X_demeaned.reshape(X_demeaned.shape[0],X_demeaned.shape[1]*X_demeaned.shape[2])
        else:
        	X = X_all[:,:,tp]

        # reduce dimensionality
        if use_pca == True:
            pca = PCA(n_components=pca_components)
            X = pca.fit(X).transform(X)
            
        #train test set
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=1, random_state=42)
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #standardisation
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            #define classifier 
            if classifier == 'LDA':
                clf = LinearDiscriminantAnalysis()
            elif classifier == 'LR':
                clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
            # train
            clf.fit(X_train ,y_train)
            # test either binary or class probabilities
            prediction[test_index,:,tp] = clf.predict_proba(X_test)
            label_pred[test_index,tp] = clf.predict(X_test)

        out = matrix_vector_shift(prediction[:,:,tp],y,n_bins) #we want the predicted class to always be in the centre.
        centered_prediction[:,tp] = out.mean(0) # avg across trials
        accuracy[tp] = accuracy_score(y,label_pred[:,tp])
    bar.finish()

    return centered_prediction, accuracy, time, prediction


