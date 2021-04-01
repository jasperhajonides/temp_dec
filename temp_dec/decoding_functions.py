#!/usr/bin/python
# -*- coding: utf-8 -*-

# author: JE Hajonides Jul 2020
# email: jasperhajonides@gmail.com
import math
import sys
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
 from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def check_input_dim(x_all, y):
    """Check if the labels match first dimension
       Check if time vector matches third dimension"""
    if len(y) != x_all.shape[0]:
        raise ValueError('Number of labels (y) does not match the first dimension of the data.')


def get_classifier(classifier, x_train):
    """This function is used to initialise the classifier """
    #define classifier #get classifier function
    if classifier == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif classifier == 'GNB':
        clf = GaussianNB()
    elif classifier == 'svm':
        clf = CalibratedClassifierCV(LinearSVC())
    elif classifier == 'LG':
        clf = LogisticRegression(random_state=0, solver='lbfgs', 
                                 multi_class='multinomial')
    elif classifier == 'maha':
        clf = KNeighborsClassifier(n_neighbors=15, metric='mahalanobis',
                                   metric_params={'V': np.cov(x_train[:, :].T)})
    else:
        raise ValueError('Classifier not correctly defined.')
    return clf


def temporal_decoding(x_all, y, n_bins=12, size_window=5,
                      n_folds=5, classifier='LDA',
                      pca_components=.95, demean='window'):

    """
    Apply a multi-class classifier (amount of class is equal to n_bins)
    to each time point.

    The temporal_dynamics decoding takes a time course and uses a 
    sliding window approach to decode the feature of interest. We reshape 
    the window from trials x channels x time to trials x channels*time and 
    demean every channel


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
    x_all : ndarray
            trials by channels by time
    y     : ndarray
            vector of labels of each trial
    time  : ndarray
            vector with time labels locked to the cue
    bins  : integer
            how many stimulus classes are present
    size_window :  integer
            size of the sliding window.
            if 1, just single time-point by time-point decoding
    n_folds :  integer
            folds of cross validation
    classifier : string
            Classifier used for decoding
            options:
            - LDA: LinearDiscriminantAnalysis
            - LG: LogisticRegression
            - maha: nearest neighbours mahalanobis distance
            - GNB: Gaussian Naive Bayes
    pca_components   : integer
            reduce features to N principal components,
            if N < 1 it indicates the % of explained variance
    demean : string
            'window' option demeans each feature within each window
            any other input disables demeaning.
    Returns
    --------
    dictionary:
        accuracy : ndarray
                dimensions: time
                matrix containing class predictions for each time point.

        single_trial_evidence: ndarray
                dimensions: trials, classes, time
                evidence for each class, for each timepoint, for each trial
        

    """

    #### do dimensions of the input data match?
    check_input_dim(x_all, y)

    # Get shape
    [n_trials, n_features, n_time] = x_all[:, :, :].shape
    time = np.arange(n_time)

	#initialise variables
    # np.nan to avoid zeroes in resulting variable
    single_trial_evidence = np.zeros(([n_trials, n_bins, n_time])) * np.nan
    label_pred = np.zeros(([n_trials, n_time])) * np.nan
    accuracy = np.zeros(n_time) * np.nan
    centered_prediction = np.zeros(([n_bins, n_time])) * np.nan
    x_demeaned = np.zeros((n_trials, n_features, size_window)) * np.nan

 
    for tp in range((size_window-1), n_time):

        if size_window > 1:
            #demean features within the sliding window if demean=='window'
            for count, s in enumerate(np.arange(size_window)-(size_window-1)):
                x_demeaned[:, :, count] = (x_all[:, :, tp+s] -
                                           x_all[:, :, (tp-(size_window-1)):(tp+1)].mean(2)*
                                           ('window' in demean))
            # reshape into trials by features*time
            X = x_demeaned.reshape(x_demeaned.shape[0],
                                   x_demeaned.shape[1]*x_demeaned.shape[2])
        else:
            X = x_all[:, :, tp]

        # reduce dimensionality
        if pca_components != 1:
            pca = PCA(n_components=pca_components)
            X = pca.fit(X).transform(X)

        #train test set
        rskf = RepeatedStratifiedKFold(n_splits=n_folds,
                                       n_repeats=1, random_state=42)
        for train_index, test_index in rskf.split(X, y):
            x_train, x_test = X[train_index], X[test_index]
            y_train = y[train_index]

            #standardisation
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            #initiate classifier
            clf = get_classifier(classifier, x_train)

            # train
            clf.fit(x_train, y_train)

            # test either binary or class probabilities
            single_trial_evidence[test_index, :, tp] = clf.predict_proba(x_test)
            label_pred[test_index, tp] = clf.predict(x_test)

        #compute accuracy score 
        accuracy[tp] = accuracy_score(y, label_pred[:, tp])
    
    evidence = {"accuracy": accuracy,
        "single_trial_evidence": single_trial_evidence,
        "time": time,
        "y": y}

    return evidence


def cos_convolve(evidence):
    """Take as input the classifier evidence for single trials in dictionary format
    and return alligned evidence and evidence convolved with a cosine.
    
    Input:
    dictionary:
        accuracy : ndarray
                dimensions: time
                matrix containing class predictions for each time point.
        single_trial_evidence: ndarray
                dimensions: trials, classes, time
                evidence for each class, for each timepoint, for each trial
        y: ndarray
                dimensions: ndarray
                vector of integers indicating the class on each trial.
    returns:
    dictionary, same as input with added:

        centered_prediction: ndarray
                dimensions: classes, time
                matrix containing evidence for each class for each time point.
        cos_convolved: ndarray
                dimensions: time
                cosine convolved evidence for each timepoint.
        single_trial_cosine_fit: ndarray
                dimensions: trial by time
                cosine convolved evidence for each timepoint and each trial.
        single_trial_ev_centered: ndarray    
                dimensions: trials, classes, time
                evidence for each class, for each timepoint, for each trial
                centered around the class of interest.
                """
                
    n_trials, n_bins, n_time = evidence["single_trial_evidence"].shape

    evidence_shifted = np.zeros((n_trials, n_bins, n_time))
    cos_ev_single_trial = np.zeros((n_trials, n_time))
    centered_prediction = np.zeros((n_bins, n_time))
    for tp in range(n_time):
        #we want the predicted class to always be in the centre.
        evidence_shifted[:,:,tp] = matrix_vector_shift(evidence["single_trial_evidence"][:, :, tp], evidence["y"], n_bins)
        centered_prediction[:, tp] = evidence_shifted[:, :, tp].mean(0) # avg across trials


    for trl in range(n_trials):
        cos_ev_single_trial[trl, :] = convolve_matrix_with_cosine(evidence_shifted[trl, :, :])

    # convolve trial average tuning curves with cosine    
    cos_convolved = convolve_matrix_with_cosine(centered_prediction)
    #fit tuning curve
    out_tc = least_squares_fit_cos(centered_prediction)

    evidence["centered_prediction"] = centered_prediction
    evidence["cos_convolved"] = cos_convolved
    evidence["tuning_curve_conv"] = out_tc['amplitude']
    evidence["single_trial_ev_centered"] = evidence_shifted
    evidence["single_trial_cosine_fit"] = cos_ev_single_trial

    return evidence



