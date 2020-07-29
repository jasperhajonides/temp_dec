#!/usr/bin/python
# -*- coding: utf-8 -*-

# /Users/jasperhvdm/Documents/Scripts/Supervised_Learning_TB/decoding_functions.py
# usage:
"""
import sys
sys.path.append('/Users/jasperhvdm/Documents/Scripts/Supervised_Learning_TB')
from decoding_functions import *
"""
# author: JE Hajonides Oct 2019


import numpy as np
import math
import scipy.io
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# import progressbar
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

def convolve_with_cosine(distances):
    #read in data
    [nbins,ntimepts] = distances.shape
    output = np.zeros(ntimepts) # define output
    #theta -pi to pi
    theta = np.arange((nbins+1))/(nbins)*(2*math.pi)-math.pi
    theta = theta[1:(nbins+1)]
    for tp in range(0,ntimepts):
        t = np.cos(theta)*distances[:,tp]
        output[tp] = t.mean(0)            
    return output
    

    
    
def temporal_decoding(X_all,y,time,n_bins=12,size_window=5,n_folds=5,classifier='LDA',use_pca=False,pca_components=.95,temporal_dymanics=True):
    """
    Apply a multi-class classifier (amount of class is equal to n_bins) to each time point.
    
    The temporal_dynamics decoding takes a time course and uses a sliding window approach to 
    decode the feature of interest. We reshape the window from trials x channels x 
    time to trials x channels*time and demean every channel 
    
    
    Temporal_dynamics:
    (rows=features; columns=time)
    (n = window_size)
    
    	t  t+1  t+2     t+n		combined t until t+n
        1 	 1 	 1 ..	 1        1
        2 	 2 	 2 ..	 2        2
        3 	 3 	 3 ..	 3        3
        4 	 4 	 4 ..	 4 ---->  4
        5 	 5 	 5 ..	 5        5
        6 	 6 	 6 ..	 6        6
        7 	 7 	 7 .. 	 7        7
    							  1
								  2
								  3
								  4
								  5
								  6
								  7
								  1
								  ..
								  7

    Parameters
    ----------
    X_all : ndarray
            trials by channels by time 
    y     : ndarray
            vector of labels of each trial
    time  : ndarray
            vector with time labels locked to the cue
    bins  : integer
            how many stimulus classes are present
    size_window :  integer
            size of the sliding window
    n_folds :  integer
            folds of cross validation
    classifier : string
            Classifier used for decoding
            options:
            - LDA: LinearDiscriminantAnalysis
            - LG: LogisticRegression
            - maha: nearest neighbours mahalanobis distance
            - GNB: Gaussian Naive Bayes 
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
    
    # Get shape 
    [n_trials,n_features,n_time] = X_all[:,:,:].shape

	#initialise variables
    prediction = np.zeros(([n_trials,n_bins,n_time])) * np.nan
    label_pred = np.zeros(([n_trials,n_time])) * np.nan
    accuracy   = np.zeros(n_time) * np.nan
    centered_prediction = np.zeros(([n_bins,n_time])) * np.nan
    X_demeaned = np.zeros((n_trials,n_features,size_window)) * np.nan

	
    for tp in range((size_window-1),n_time):

        if temporal_dymanics == True:
            #demean features within the sliding window.
            
            cc=0 
            for s in range((1-size_window),1): 
                X_demeaned[:,:,cc] = X_all[:,:,tp+s] - X_all[:,:,(tp-(size_window-1)):(tp+1)].mean(2)
                cc=cc+1
            # reshape into trials by features*time     
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
            if classifier=='LDA':
                clf = LinearDiscriminantAnalysis()
            elif classifier=='GNB':
                clf=GaussianNB()
            elif classifier == 'svm':
                clf=CalibratedClassifierCV(LinearSVC())
            elif classifier == 'LG':
                clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
            elif classifier == 'maha':
                clf = KNeighborsClassifier(n_neighbors=15,metric='mahalanobis', metric_params={'V': np.cov(X_train[:,:].T)})

	
            # train
            clf.fit(X_train ,y_train)
            
            # test either binary or class probabilities
            prediction[test_index,:,tp] = clf.predict_proba(X_test)
            label_pred[test_index,tp] = clf.predict(X_test)

        out = matrix_vector_shift(prediction[:,:,tp],y,n_bins) #we want the predicted class to always be in the centre.
        centered_prediction[:,tp] = out.mean(0) # avg across trials
        accuracy[tp] = accuracy_score(y,label_pred[:,tp])
    cos_convolved = convolve_with_cosine(centered_prediction)

    return centered_prediction, accuracy, time, prediction,cos_convolved
