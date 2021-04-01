#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:36:48 2021

@author: jasperhajonides
"""
#
import math
import numpy as np
import pycircstat



def reject_outliers(data, m=2):
    """ Remove outliers"""
    return abs(data - np.mean(data)) < m * np.std(data)

def bin_array(y,nr_bins):
    """Takes circular array and digitises into n bins"""
    y = np.array(y)
    bins = np.arange(-math.pi-0.000001,math.pi - (2*math.pi/nr_bins),(2*math.pi/nr_bins))
    y_bin = np.digitize(pycircstat.cdiff(y,0), bins) 
    return y,y_bin


def matrix_vector_shift(matrix, vector, n_bins):

    """ Shift rows of a matrix by the amount of columns specified
		in the corresponding cell of the vector.

	e.g. M =0  1  0     V = 0 0 1 2     M_final =   0 1 0
			0  1  0									 0 1 0
			1  0  0									 0 1 0
			0  0  1								  	 0 1 0
            """
    row, col = matrix.shape
    matrix_shift = np.zeros((row, col))
    for row_id in range(0, row):
        matrix_shift[row_id, :] = np.roll(matrix[row_id, :], int(np.floor(n_bins/2)-vector[row_id]))
    return matrix_shift

#%%

