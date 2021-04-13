#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:54:26 2021

@author: jasperhajonides
"""
import numpy as np

def sample_data_time(n_trials=100, n_features=5, num_samples=200, n_classes = 5):
    """ Create sample time series dataset"""
    # n_features = 30
    # n_classes = 5
    # n_trials = 100
    # num_samples = 100
    max_theta = np.pi*5
    
    thetas = np.linspace(0,max_theta,num_samples)
    x = np.zeros((n_trials*n_classes, n_features-1, len(thetas)))
    y = np.zeros(n_trials*n_classes)
    counter = 0
    for i, cl in enumerate(np.arange(-np.pi, np.pi, np.pi*2/n_classes)): 
        for trial in range(n_trials):
            for feat in range(1,n_features):
    
                x_temp  = n_features/feat*np.sin(thetas+cl+np.random.rand()) + feat/n_features*np.cos(thetas+cl+np.random.rand())
                # add noise 
                # np.random.seed(42)
                for tp in range(num_samples):
                    n = np.random.rand() * 50*tp/num_samples
                
                    x[counter, feat-1, tp] = x_temp[tp] + n
            y[counter] = i
            counter += 1
    return x, y