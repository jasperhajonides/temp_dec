#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:54:26 2021

@author: jasperhajonides
"""
import numpy as np

def sample_data_time(n_trials=100, n_features=30, num_samples=100, n_classes = 10):
    """ Create sample time series dataset"""
    # n_features = 30
    # n_classes = 5
    # n_trials = 100
    # num_samples = 100
    max_theta = np.pi*2
    
    thetas = np.linspace(0,max_theta,num_samples)
    x = np.zeros((n_trials*n_classes, n_features, len(thetas)))
    y = np.zeros(n_trials*n_classes)
    counter = 0
    for cl in range(n_classes):
        for trial in range(n_trials):
            for ii, feat in enumerate(np.sin(np.arange(-np.pi, np.pi, np.pi*2/30))):
    
    
                x_temp  = cl*np.sin(thetas) + feat
        
                # Add a little noise - with low frequencies removed to make this example a
                # little cleaner...
                np.random.seed(42)
                n = np.random.randn(len(x_temp),) * 5
                
                x[counter, ii, :] = x_temp + n
            y[counter] = cl
            counter += 1
    return x, y