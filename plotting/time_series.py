#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:04:19 2021

@author: jasperhajonides
"""
#%%
import numpy as np

def tsplot(ax, data, time, color = 'k', linestyle = 'solid',legend='target',chance=0.0):
    x = time
    est = np.mean(data, axis=1)
    sd = np.std(data, axis=1)
    se = (sd/np.sqrt(data.shape[1]))
    cis = (est - se, est + se)
    
    ax.fill_between(x,cis[0],cis[1],alpha = 0.25, facecolor = color)
    ax.plot(x,est,color = color, linestyle = linestyle,label=legend)
    ax.margins(x=0)
    ax.hlines(chance,-.4,0.8,color='gray',linestyle='dashed')

    ax.set_xlabel('Time (s)',fontsize=14)
    ax.set_ylabel('Evidence',fontsize=14)  # Area Under the Curve
    ax.legend()
    # ax.axvline(.0, color='k', linestyle='-')