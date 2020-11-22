import numpy as np
import math
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.optimize import leastsq


import sys
sys.path.append('/Users/jasperhvdm/Documents/Scripts/temp_dec/temp_dec/')
from decoding_functions import *


def check_bin_shift(n_bins, theta, shifts):
    """ if we shift the label assignment by a fraction of a bin,
    how much data is assigned differently?

    Parameters
    ----------
    n_bins : integer
            the stimulus space will be subdivided into a set amount of stimulus
            classes.
    theta: ndarray
            all original available raw labels
    shifts: integer
            shift bin assignment by this fraction of a bins
            value of 2 means bins overlap 50%
            value of 3 means bins overlap 66.67%
            value of 4 means bins overlap 75%
            ...
            value of 100 means bins overlap 99%
    Returns
    --------
    prints the percentage of data that moves from initial bin assignment
    to the new bin assignment
            """

    bins = np.arange(math.pi/n_bins,math.pi, math.pi/n_bins)
    ask = np.zeros((n_bins,shifts))
    use_bins = np.append(bins,0)
    for s in range(0,shifts):
        for i in range(0, n_bins):
            ask[i,s] = np.sum((theta > use_bins[i]+0.001) &
                              (theta < use_bins[i]+(np.pi/n_bins/shifts)+0.001))
        use_bins = use_bins + (np.pi/n_bins/shifts)

    print('{0} of data changes bin assignemnt every 1/{1} shift ({2}/{3})'.format((np.sum(ask))/(len(theta)*shifts),
                                                                shifts,
                                                                int((np.sum(ask))),
                                                                (len(theta)*shifts)))
    if np.sum(ask == 0):
        print('Warning: {0}/{1} bin(s) stayed the same, consider decreasing shift'.format(np.sum(ask == 0),n_bins*shifts))

def configure_recursive_tuning_curves(theta, shifts = 8, n_bins = 8, pca_components = .95,
window_size = 1, n_folds = 10, classifier = 'LDA'):
    """ This function creates the config file for the decoding function.
        In addition, it calls functions to check whether the stimulus classes
        are sensibly defined.

        """

    check_bin_shift(n_bins, theta, shifts)

    config = {'n_folds':n_folds,
              'pca_components':pca_components,
              'n_bins': n_bins,
              'shifts': shifts,
              'classifier':classifier,
              'size_window': window_size}
    print('--successful configuration--')
    return config



def recursive_tuning_curves(X,y_in,config):
    """This function runs decoding as normal but runs it multiple times, every time it offsets the labels slightly

    Parameters
    ----------
    X : ndarray
            3d: trials by channels by time

    y     : ndarray
            1d: trials
            0-pi values of each presented stimulus

    config: dictionary
        shifts: integer
                how many shifts in stimulus values
        bins  : integer
                how many stimulus classes are present

    Returns
    -------


    """

    #binning values
    bins = np.arange(math.pi/config['n_bins'],math.pi, math.pi/config['n_bins'])
    all_evidence_combined = np.zeros((X.shape[0], config['n_bins'],config['shifts']))
    for q in range(0, config['shifts']):

        # we bin y differently by shifting the binning boundries by a fraction of a binsize
        offset = q/config['n_bins']/config['shifts']*np.pi
        shifted_y = np.mod(y_in+offset, np.pi)
        y = np.digitize(shifted_y, bins) #convert to categorical labels



        ###### temporal decoding
        if len(X.shape) == 2:
            [n_trials, n_features] = X.shape
            n_time = 1
        elif len(X.shape) == 3:
            [n_trials, n_features, n_time] = X[:, :, :].shape

        single_trial_evidence = np.zeros(([n_trials, config['n_bins'], n_time])) * np.nan
        X_demeaned = np.zeros((n_trials, n_features, n_time)) * np.nan
        single_trial_evidence = np.zeros(([n_trials, config['n_bins']])) * np.nan


        if n_time > 1:
            #demean features within the sliding window.
            cc=0
            for s in range(0, n_time):
                X_demeaned[:, :, cc] = X[:, :, 0+s] - X[:, :, :].mean(2)
                cc=cc+1
            # reshape into trials by features*time
            X = X_demeaned.reshape(X_demeaned.shape[0],
                                   X_demeaned.shape[1]*X_demeaned.shape[2])
        else:
            X = X #redundant but for clarity it helps

        # reduce dimensionality
        if config['pca_components'] != 1:
            pca = PCA(n_components=config['pca_components'])
            X = pca.fit(X).transform(X)

        #crossvalidation
        rskf = RepeatedStratifiedKFold(n_splits=config['n_folds'],
                                       n_repeats=1, random_state=42)
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #standardisation
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            #initiate classifier
            clf = get_classifier(config['classifier'], X_train)

            # train
            clf.fit(X_train, y_train)
            # test either binary or class probabilities
            single_trial_evidence[test_index, :] = clf.predict_proba(X_test)

        # now reallign so that the predicted bin is always in the center
        all_evidence_combined[:, :, q] = matrix_vector_shift(single_trial_evidence,
                                                             y, config['n_bins'])

    return all_evidence_combined



def cosine_least_squares_fit(tuning):
    """Fit a cosine to tuning curve data to obtain model fit Parameters

    Parameters
    ----------
    tuning : ndarray
            1d values of tuning curve sampled from [-pi-x to pi]
            or
            2d array of tuning curve bins by repeats
    t     : ndarray
            vector of tuning curve phase ranging from [-pi-x to pi]
            where x is the the stepsize
    Returns
    --------
    dictionary:
        data_fit : ndarray
                estimated cosine fit to data
        amplitude: ndarray
                highest value - mean of the cosine
        phase: ndarray
                phase of cosine. negative value means peak is shifted left,
                postitive value means peak is shifted right
        mean: ndarray
                mean value of cosine
    """
    n_steps = tuning.shape[0]
    t = np.linspace(-np.pi+(0.5*2*np.pi/n_steps),np.pi-(0.5*2*np.pi/n_steps),n_steps)
    guess_mean = 1/9
    guess_std = np.std(tuning)#3*np.std(0.0033)/(2**0.5)/(2**0.5)
    guess_phase = 0
    guess_amp = 1

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.cos(t+guess_phase) + guess_mean

    # if input is 2d array we loop over the second dimension
    if len(tuning.shape) > 1:
        repeats = tuning.shape[1]
        #initialise parameter estimates
        est_amp = np.zeros(repeats)
        est_phase = np.zeros(repeats)
        est_mean = np.zeros(repeats)
        data_fit = np.zeros((n_steps,repeats))

        for rep in range(0,tuning.shape[1]):
            optimize_func = lambda x: x[0]*np.cos(t+x[1]) + x[2] - tuning[:,rep]
            ea, ep, em = leastsq(optimize_func, [guess_amp, guess_phase, guess_mean])[0]
            # if max amplitude is negative we also multiply phase shift by -1
            est_amp[rep] = ea * np.sign(ea)
            est_phase[rep] = ep * np.sign(ea)
            est_mean[rep] = em

            # recreate the fitted curve using the optimized parameters
            data_fit[:,rep] = ea*np.cos(t+ep) + em
    else:
        optimize_func = lambda x: x[0]*np.cos(t+x[1]) + x[2] - tuning
        est_amp, est_phase, est_mean = leastsq(optimize_func, [guess_amp,
                                               guess_phase, guess_mean])[0]

        # if max amplitude should not be negative, otherwise we'll shift phase accordingly
        if (est_amp < 0) & (est_phase<0):
            est_amp = est_amp * np.sign(est_amp)
            est_phase = np.pi + est_phase
        elif (est_amp < 0) & (est_phase>0):
            est_amp = est_amp * np.sign(est_amp)
            est_phase = np.mod(est_phase + 2*np.pi,2*np.pi)-np.pi


        # recreate the fitted curve using the optimized parameters
        data_fit = est_amp * np.cos(est_freq*t+est_phase) + est_mean


    output = {
    "data_fit": data_fit,
    "amplitude": est_amp,
    "phase": -est_phase,
    "mean" : est_mean
    }

    return output

def bias_fitting_single_timepoint(tuning,distractor,split,nr_bins=64):
    """Fit a cosine tuning curve and determine the phase shift of the present information
    towards or away a second orientation

    Parameters
    ----------
    tuning     : ndarray
            2d array of tuning curve bins by repeats
    distractor : ndarray
            vector of tuning curve phase ranging from [-pi-x to pi]
            where x is the the stepsize
    splits     : ndarray
            vector of the same length as the 'distractor' variable. 
            where integers indicate different conditions that are averaged separately.
            zeros will not be included in these conditions 
    options    : dict
        bins        : resolution of the circular distractor 

    Returns
    --------
    dictionary:
        xxxx : ndarray
        xxxx : ndarray"""

    pe = np.zeros((tuning.shape[0],tuning.shape[2]))
    phase_conditions = np.zeros((int(split.max()),tuning.shape[2]))*np.nan

    # define bins
    distr_digi_bins = np.arange(math.pi/nr_bins,math.pi, math.pi/nr_bins) #define x bins
    distr_digitize =  np.digitize((distractor[~np.isnan(distractor)])/2,distr_digi_bins) #digitise into these defined bins

    # if splitting classifier output for different conditions
    if len(split) == tuning.shape[1]:
        for con in range(1,int(split.max()+1)): 
            print('%s. '%con + '%s' %len(np.unique(distr_digitize[split==con])) + '/%s' %nr_bins +  ' unique target-distractor bins')
        
    for tp in range (tuning.shape[2]):

        output_fit = cosine_least_squares_fit(tuning[:,~np.isnan(distractor) ,tp])
        phase_est = output_fit['phase']

        #get the mean for all previous orientations
        df = pd.DataFrame({'inx': distr_digitize,
                            'distractor': distr_digitize,
                            'phase_est': phase_est})

        if len(split) == tuning.shape[1]:
            for con in range(1,int(split.max()+1)):
                phase_conditions[con-1,tp] = output_fit['phase'][split==con].mean()
                
            df_avg = df.groupby('inx').mean()
            pe[df_avg['distractor'],tp] = df_avg['phase_est']

    output_fit['phase_estimate_bins'] = pe
    output_fit['phase_conditions'] = phase_conditions

    return output_fit