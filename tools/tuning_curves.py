#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:09:38 2021

@author: jasperhajonides
"""

# cos convolve

import pycircstat
from tools.constr import *

def compute_asymmetry(data, y = None, reference = None, min_deg = 30, max_deg = 60):
    """Compute distribution of x by phase-bins in the Instantaneous Frequency.

    Parameters
    ----------
    data : ndarray
        three-dimensional array of [trial repeats by classes by time points]. 
        Correct class is in position n_classes/2 
        
        *alternatively 'data', 'y', and 'reference' can be specified
         as a dictionary
    y : ndarray
        Input vector for presented stimulus in radians  with the same 
        length as trials 
    reference : ndarray
        Input vector for reference stimulus in radians  with the same 
        length as trials 
    min_deg : float
         min angular distance cut-off point for trials to include
    max_deg : float
         max angular distance cut-off point for trials to include

    Returns
    -------
    shift : ndarray
        array containing asymmetry score for every time point.
    CW : ndarray
        array containing evidence for CW angular difference trials
    CCW : ndarray
        array containing evidence for CCW angular difference trials

    """
    
    # check input
    if type(data) is dict:
        reference = data['reference']
        y = data['y']
        data = data['single_trial_ev_centered']
        
    if type(data) is np.ndarray:
        if y is None or reference is None:
            raise Exception("Specifiy both y and reference variable") # check  
            
        if y.max() > (2*np.pi) or reference.max() > (2*np.pi):
            # if variables are out of range 
            raise Exception("Define y and reference in radians") 

    # main body
        
    n_trials, n_classes, n_tps = data.shape
    
    angular_difference = pycircstat.cdiff(reference,y)
    
    # compute evidence for all classes for trials CW from current angle
    CW = data[(angular_difference <= -min_deg/90*np.pi) & 
                (angular_difference > -max_deg/90*np.pi), :, :].mean(0)
    
    # compute evidence for all classes for trials CCW from current angle
    CCW =data[(angular_difference >= min_deg/90*np.pi) &
                (angular_difference < max_deg/90*np.pi), :, :].mean(0)

    # sum 
    shift = ((CW[0:int(n_classes/2-1),:].mean(0) - CW[int(n_classes/2):(n_classes-1), :].mean(0)) - 
    (CCW[0:int(n_classes/2-1), :].mean(0) - CCW[int(n_classes/2):(n_classes-1), :].mean(0))) 

    return shift, CW, CCW



# -------------------------

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



def convolve_matrix_with_cosine(distances):
    """Fits a cosine to the class predictions. This assumes
   neighbouring classes are more similar than distant classes """
    #read in data
    [nbins, ntimepts] = distances.shape
    output = np.zeros(ntimepts) # define output
    #theta -pi to pi
    theta = np.arange((nbins+1))/(nbins)*(2*math.pi)-math.pi
    theta = theta[1:(nbins+1)]
    for tp in range(0, ntimepts):
        t = np.cos(theta)*distances[:, tp]
        output[tp] = t.mean(0)
    return output



from scipy.optimize import leastsq
import numpy as np


def least_squares_fit_cos(tuning,axis=0):
    """Fit a cosine to tuning curve data to obtain model fit Parameters

    Parameters
    ----------
    tuning : ndarray
            1d values of tuning curve sampled between [-pi to pi] in even intervals
            or
            2d array of tuning curve bins by repeats
            or
            3d array of tuning curve bins with 2 dimensions of repeats
    axis     : integer
            indicates which axis indexes the tuning curve
          
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
    
    if axis != 0: tuning = np.transpose(tuning,np.roll([0,1,2],-axis))
    if len(tuning.shape) == 1: tuning = tuning[:,None]#np.expand_dims(tuning,axis=1)
    if len(tuning.shape) == 2: tuning = tuning[:,:,None] #np.expand_dims(tuning,axis=2)

    n_steps = tuning.shape[0]
    
#   vector of tuning curve phase ranging from [-pi-x to pi]
#         where x is the the stepsize
    t = np.linspace(-np.pi+(0.5*2*np.pi/n_steps),np.pi-(0.5*2*np.pi/n_steps),n_steps)

    #initialise
    est_amp = np.zeros((tuning.shape[1],tuning.shape[2]))
    est_phase = np.zeros((tuning.shape[1],tuning.shape[2]))
    est_mean = np.zeros((tuning.shape[1],tuning.shape[2]))
    data_fit = np.zeros((tuning.shape))
    
    ea, ep, em = 0.1,0,1/n_steps
    for i in range(0,tuning.shape[1]):
        for j in range(0,tuning.shape[2]):
             
             optimize_func = lambda x: x[0]*np.cos(t-x[1]+1/n_steps*np.pi) + x[2] - tuning[:,i,j]
             ea, ep, em = leastsq(optimize_func, [ea, ep, em])[0]
#             
             # cap phase in range [-pi, pi]
             ep = (ep + np.pi)%(2*np.pi)-np.pi
             
             #amplitude of cos is always positive, adjust the phase 
             if (ea < 0) & (ep>0):
                 ea = -ea 
                 ep = -(np.pi - ep)
             elif (ea < 0) & (ep<0):
                 ea = -ea
                 ep = (np.pi + ep)
                 
             est_amp[i,j] = ea
             est_phase[i,j] = ep
             est_mean[i,j] = em
             data_fit[:,i,j] = ea*np.cos(t+ep) + em

    output = {
    "data_fit": data_fit,
    "amplitude": est_amp,
    "phase": est_phase,
    "mean" : est_mean
    }
   

    return output