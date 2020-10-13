from sklearn.feature_selection import RFECV
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def select_features_univariate(X,y,method='Decision_Tree'):
    """ with high dimensional datasets it aids classifier performance to select
    features of interest
    This function rejects features below a certain (univariate) threshold.


    Parameters
    ----------
    X : ndarray
            repetitions by features
    y     : ndarray
            vector of labels of each repetition
    method : string
            function used for data reduction
            {'decision_tree','decision_tree_RFECV','mutual_information',...
            'univariate_select'}
    Returns
    --------
    dictionary:
        X_transformed : ndarray
                repetitions by features (reduced)
        weights: ndarray or Boolean
                relative importance features or binary (important or not)

        """
    # based on the method we choose the clf to fit and transform the data
    if method == 'decision_tree_RFECV':
        clf = DecisionTreeClassifier()
        trans = RFECV(clf)
        X_transformed  = trans.fit_transform(X, y)
        weights = trans.get_support()
    elif method == 'decision_tree':
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        # choose features with an importance that is more than avg.
        selected_features = np.where(clf.feature_importances_ > clf.feature_importances_.mean(0),1,0)
        X_transformed = X[:,selected_features==1]
        weights = clf.feature_importances_
    elif method == 'mutual_information':
        mutual_info = mutual_info_classif(X, y)
        # choose features above the avg mutual information threshold.
        selected_features = np.where(mutual_info > mutual_info.mean(0),1,0)
        X_transformed = X[:,selected_features==1]
        weights = mutual_info #continuous
    elif method == 'univariate_select':
        # select features with more univariate activity than avg.
        trans = GenericUnivariateSelect(score_func=lambda X, y: X.mean(axis=0), mode='percentile', param=50)
        X_transformed = trans.fit_transform(X, y)
        weights = trans.get_support() #binary

    return X_transformed, weights
