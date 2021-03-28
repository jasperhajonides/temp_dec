#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:51:02 2021

@author: jasperhvdm
"""

from scipy.optimize import leastsq
import numpy as np
#%%

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

