#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:47:31 2021

@author: jasperhajonides
"""
import numpy as np

def sliding_window(data, size_window=20, demean=True):
    """ Reformat array so that time point t includes all information from
        features up to t-n where n is the size of the predefined window.

         Parameters
        ----------
        data : ndarray
            3-dimensional array of [trial repeats by features by time points].
            Demeans each feature within specified window.
        size_window : int
            number of time points to include in the sliding window
        demean : bool 
            subtract mean from each feature within the specified sliding window
        Returns
        --------
        output : ndarray
            reshaped array where second dimension increased by size of
            size_window

        example:
            
        100, 60, 240 = data.shape

        data_out = sliding_window(data, size_window=5)

        100, 300, 240 = data_out.shape

    """
    
    try:
        n_obs, n_features, n_time = data.shape
    except ValueError:
        raise ValueError("Data has the wrong shape")
    
    if size_window <= 1 or len(data.shape) < 3 or n_time < size_window:
        print('not suitable')
        return data

    # predefine variables
    output = np.zeros((n_obs, n_features*size_window, n_time))
    
    # loop over third dimension
    for time in range(size_window-1, n_time):
        #concatenate features within window 
        # and demean features if selected
        mean_value = data[:, :, (time-size_window+1):(time+1)].mean(2)
        x_window = data[:, :, (time-size_window+1):(time+1)].reshape(
            n_obs, n_features*size_window) - np.tile(mean_value.T, size_window).reshape(
            n_obs, n_features*size_window)*float(demean)        
        # add to array
        output[:, :, time] = x_window 
    return output

    
 