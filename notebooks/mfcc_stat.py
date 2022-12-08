# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:00:31 2021

@author: deep
"""

from __future__ import print_function
import tensorflow.compat.v1 as tf
import numpy as np
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import scipy
import math



def rmsValue(arr, n): 
    """Basic function to get rms that we will use later"""
    square = 0
    mean = 0.0
    root = 0.0
    for i in range(0,n): 
        square += (arr[i]**2)  
    mean = (square / (float)(n)) 
    root = math.sqrt(mean) 
    return root 



class mfcc_stat():
    """Class that will extract different descriptive statistics from each of the 13 components of MFCCs"""
    def __init__(self,data):
        self.data = data
    
    def mean(self):
        MEAN = []
        for mfcc in self.data:
            mean=[]
            samples = 13
            for idx in range(samples):
                temp = np.mean(mfcc[idx])
                mean.append(temp)
            MEAN.append(mean)
        return MEAN
    
    def std(self):
        STD = []
        for mfcc in self.data:
            std=[]
            samples = 13
            for idx in range(samples):
                temp = np.std(mfcc[idx])
                std.append(temp)
            STD.append(std)
        return STD
    
    def median(self):
        MEDIAN = []
        for mfcc in self.data:
            median=[]
            samples = 13
            for idx in range(samples):
                temp = np.median(mfcc[idx])
                median.append(temp)
            MEDIAN.append(median)
        return MEDIAN

    def maximum(self):
        MAX = []
        for mfcc in self.data:
            maximum=[]
            samples = 13
            for idx in range(samples):
                temp = np.max(mfcc[idx])
                maximum.append(temp)
            MAX.append(maximum)
        return MAX

    def minimum(self):
        MIN = []
        for mfcc in self.data:
            minimum=[]
            samples = 13
            for idx in range(samples):
                temp = np.min(mfcc[idx])
                minimum.append(temp)
            MIN.append(minimum)
        return MIN

    def quartile_1st(self):
        QTL = []
        for mfcc in self.data:
            qtl=[]
            samples = 13
            for idx in range(samples):
                temp = np.quantile(mfcc[idx],0.25)
                qtl.append(temp)
            QTL.append(qtl)
        return QTL
    
    def quartile_3rd(self):
        QTL = []
        for mfcc in self.data:
            qtl=[]
            samples = 13
            for idx in range(samples):
                temp = np.quantile(mfcc[idx],0.75)
                qtl.append(temp)
            QTL.append(qtl)
        return QTL

    def interquartilerange(self):
        IQR = []
        for mfcc in self.data:
            iqr=[]
            samples = 13
            for idx in range(samples):
                temp = scipy.stats.iqr(mfcc[idx])
                iqr.append(temp)
            IQR.append(iqr)
        return IQR
    
    def skewness(self):
        SKEW = []
        for mfcc in self.data:
            skew=[]
            samples = 13
            for idx in range(samples):
                temp = scipy.stats.skew(mfcc[idx])
                skew.append(temp)
            SKEW.append(skew)
        return SKEW 
    
    def kurtosis(self):
        KURTOSIS = []
        for mfcc in self.data:
            kurtosis=[]
            samples = 13
            for idx in range(samples):
                temp = scipy.stats.skew(mfcc[idx])
                kurtosis.append(temp)
            KURTOSIS.append(kurtosis)
        return KURTOSIS
    
    
    def get_rms(self):
        RMS = []
        for mfcc in self.data:
            rms=[]
            samples = 13
            n = len(mfcc[0])
            for idx in range(samples):
                temp = rmsValue(mfcc[idx],n)
                rms.append(temp)
            RMS.append(rms)
        return RMS
    
 
    
                
            