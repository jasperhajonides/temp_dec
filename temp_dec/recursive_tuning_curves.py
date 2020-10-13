import convolve_matrix_with_cosine
import numpy as np
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
        print(q)

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
