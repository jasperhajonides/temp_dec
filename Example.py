#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:52:42 2021

@author: jasperhajonides
"""


from sklearn.model_selection import train_test_split
from tools.sliding_window import sliding_window
from custom_classifier import sample_data
from custom_classifier.custom_classification import CustomEstimator

# define parameters of simulated data + window size for decoding
n_classes = 10
n_trials = 100
num_samples = 100
window_size = 10

# simulate 
X, y = sample_data.sample_data_time(n_classes = n_classes, n_trials = n_trials,
                                   num_samples = num_samples)

# Compute sliding window. Every time point includes the data from the past
# n time points, where n is the window_size. 
X_time = sliding_window(X,size_window = window_size,demean=False)

# Simple split in train and test set
X_train, X_test, y_train, y_test = train_test_split(X_time, y, test_size = 0.2, 
                                                    random_state = 42) 
 
 
#%% Run LDA

# initiate arrays for classifier output
out_cos = np.zeros((int((n_trials*n_classes)*0.2),n_classes,num_samples))
out_cos_conv = np.zeros((int((n_trials*n_classes)*0.2),num_samples))
out_predict = np.zeros((num_samples))


# r
for tp in range(window_size,num_samples):
    #
    clf = CustomEstimator(est_class=LinearDiscriminantAnalysis).fit(X_train[:,:,tp],y_train)
    # out = clf.predict_proba(X_test[:,:,10])
    out_cos[:,:,tp] = clf.predict_conv_proba(X_test[:,:,tp],y_test)
    out_cos_conv[:,tp] = clf.predict_conv(X_test[:,:,tp],y_test)
    out_predict[tp] = clf.score(X_test[:,:,tp],y_test)



fig = plt.figure(figsize=(8,8))
ax = plt.subplot(3,1,1)

# plot data
ax.plot(X[0::n_trials,0,:].T)
ax.set_title('Data')
    
# plot class predictions, centered around correct class
ax = plt.subplot(3,1,2)
ax.imshow(out_cos.mean(0),aspect='auto')
ax.set_title('Predictions')
ax.set_yticks([np.floor(n_classes/2)])
ax.set_yticklabels(['Correct class'])

# 
ax = plt.subplot(3,1,3)
ax.plot(out_cos_conv.mean(0),label='cosine convolved')
ax.plot(out_predict,label='class prediction acc (%)')
ax.set_title('Accuracy predictions')

ax.legend()

plt.tight_layout()