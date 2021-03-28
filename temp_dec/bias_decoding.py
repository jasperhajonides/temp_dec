#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:11:38 2020

@author: jasperhvdm
"""

import math
import numpy as np
import pycircstat as circstat
import sys
sys.path.append('/Users/jasperhvdm/Documents/Scripts/temp_dec/temp_dec')
from decoding_functions import *
from recursive_tuning_curves import *
sys.path.append('/Users/jasperhvdm/Documents/Scripts/DPhil_toolbox/')
from pc_neuro import *
from temp_dec import decoding_functions
from cosine_least_squares_fit import *
from least_squares_fit_cos import *
    
    
def decoding_evidence_shift(evidence,y,Distr,smooth=False,nbin=100,pbin=.25,skip_fitting=False):
    """ shifts in tuning curve based on y and y2 distance dividing output into 
    nbins
    
    Parameters
    ----------
    distractor : ndarray
        vector of tuning curve phase ranging from [-pi-x to pi]
        where x is the the stepsize
    target orientation: ndarray
    distractor orientation: ndarray
    """
    
    if isinstance(evidence,dict): tuning = evidence['single_trial_ev_centered']
    else: tuning = evidence
    
    tb = np.zeros((nbin,tuning.shape[1],
                              tuning.shape[2]))

    #get the sizes
    y_diff = circstat.cdiff(Distr,y) 
    if smooth == False:
        # bins = np.arange(-math.pi,math.pi,2*math.pi/nbin)
        y, y_diff_binned = bin_array(y_diff,nbin)
        for i in range(1,nbin+1):
            tb[i-1,:,:] = evidence['single_trial_ev_centered'][y_diff_binned == i,:,:].mean(0)

    elif smooth == True:
        y_diff_binned = circ_bini(y_diff[~np.isnan(y_diff)],nbin,pbin)
        for i in range(nbin):
            tb[i,:,:] = np.nanmean(evidence['single_trial_ev_centered'][y_diff_binned[i,:] == 1])
   
        
        
    
    tuning_binned = np.zeros((2,tuning.shape[1],tuning.shape[2]))
    x_bins = np.arange(-math.pi,math.pi,2*math.pi/nbin) + 1/nbin*math.pi
    
    tuning_binned[0,:,:] = np.nanmean(tb[(x_bins >= -50/90*np.pi) & (x_bins <= -25/90*np.pi),:,:],axis=0)
    tuning_binned[1,:,:] = np.nanmean(tb[(x_bins <= 50/90*np.pi) & (x_bins >= 25/90*np.pi),:,:],axis=0)
        
    if skip_fitting == False:
        output_biasfit = least_squares_fit_cos(tuning_binned,1)  
    else:
        output_biasfit = []
        
    return output_biasfit,tuning_binned



    