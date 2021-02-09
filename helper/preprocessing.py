# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 16:36:56 2021

@author: deep
"""

import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
from tqdm import tqdm_notebook
from scipy.io import wavfile
import json
from tqdm import tqdm
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import math



#basic function to calculate root-mean-square
def rmsValue(arr, n): 
    square = 0
    mean = 0.0
    root = 0.0
    for i in range(0,n): 
        square += (arr[i]**2)  
    mean = (square / (float)(n)) 
    root = math.sqrt(mean) 
    return root 





class Get_Data():   
    """Using librosa to get foating time series in 22.5Hz from raw audio input having the label 0 for Non-Covid and 1 for Covid """
    def __init__(self,filedir,root):
        #creating a dictionary that will hash the labels to 0 and 1
        dictionary={'healthy':0,'resp_illness_not_identified':np.nan, 'recovered_full':0,'positive_mild':1,'positive_asymp':1,
               'positive_moderate':1,'no_resp_illness_exposed':np.nan}
        metadata=[]
        self.filedir = filedir
        self.root=root
        for i in self.filedir:
            f = open(self.root+r"/"+i+r"/"+'metadata.json')
            val = json.load(f)
            metadata.append(val['covid_status']) #metadata of covid status
        self.metadata=metadata
        self.dictionary=dictionary
    def get_breathe_deep(self):
        temp1=[]
        filename=[]
        df= pd.DataFrame({'label':[0]*len(self.filedir)})
        for i in tqdm(self.filedir):
            ret,sr=librosa.load(self.root+r"/"+i+r"/"+'breathing-deep.wav') #it will downsample to 22.5Hz by default
            temp1.append(ret)
            filename.append(self.root+r"/"+i+r"/"+'breathing-deep.wav')
        df['data']=temp1
        df['label']=self.metadata
        df['label'].replace(self.dictionary,inplace=True)
        df['filename']=filename # adding filename that we will use later to extract VGGish based embeddings
        return df
    def get_metadata(self):
        return self.metadata
    def get_breathe_shallow(self):
        temp1=[]
        filename=[]
        df= pd.DataFrame({'label':[0]*len(self.filedir)})
        for i in tqdm(self.filedir):
            ret,sr=librosa.load(self.root+r"/"+i+r"/"+'breathing-shallow.wav')
            temp1.append(ret)
            filename.append(self.root+r"/"+i+r"/"+'breathing-shallow.wav')
        df['data']=temp1
        df['label']=self.metadata
        df['label'].replace(self.dictionary,inplace=True)
        df['filename']=filename
        return df
    def get_cough_heavy(self):
        temp1=[]
        filename=[]
        df= pd.DataFrame({'label':[0]*len(self.filedir)})
        for i in tqdm(self.filedir):
            ret,sr=librosa.load(self.root+r"/"+i+r"/"+'cough-heavy.wav')
            temp1.append(ret)
            filename.append(self.root+r"/"+i+r"/"+'cough-heavy.wav')
        df['data']=temp1
        df['label']=self.metadata
        df['label'].replace(self.dictionary,inplace=True)
        df['filename']=filename
        return df
    def get_cough_shallow(self):
        temp1=[]
        filename=[]
        df= pd.DataFrame({'label':[0]*len(self.filedir)})
        for i in tqdm(self.filedir):
            ret,sr=librosa.load(self.root+r"/"+i+r"/"+'cough-shallow.wav')
            temp1.append(ret)
            filename.append(self.root+r"/"+i+r"/"+'cough-shallow.wav')
        df['data']=temp1
        df['label']=self.metadata
        df['label'].replace(self.dictionary,inplace=True)
        df['filename']=filename
        return df
    def get_counting_fast(self):
        temp1=[]
        filename=[]
        df= pd.DataFrame({'label':[0]*len(self.filedir)})
        for i in tqdm(self.filedir):
            ret,sr=librosa.load(self.root+r"/"+i+r"/"+'counting-fast.wav')
            temp1.append(ret)
            filename.append(self.root+r"/"+i+r"/"+'counting-fast.wav')
        df['data']=temp1
        df['label']=self.metadata
        df['label'].replace(self.dictionary,inplace=True)
        df['filename']=filename
        return df
    def get_counting_normal(self):
        temp1=[]
        filename=[]
        df= pd.DataFrame({'label':[0]*len(self.filedir)})
        for i in tqdm(self.filedir):
            ret,sr=librosa.load(self.root+r"/"+i+r"/"+'counting-normal.wav')
            temp1.append(ret)
            filename.append(self.root+r"/"+i+r"/"+'counting-normal.wav')
        df['data']=temp1
        df['label']=self.metadata
        df['label'].replace(self.dictionary,inplace=True)
        df['filename']=filename
        return df
    def get_vowel_a(self):
        temp1=[]
        filename=[]
        df= pd.DataFrame({'label':[0]*len(self.filedir)})
        for i in tqdm(self.filedir):
            ret,sr=librosa.load(self.root+r"/"+i+r"/"+'vowel-a.wav')
            temp1.append(ret)
            filename.append(self.root+r"/"+i+r"/"+'vowel-a.wav')
        df['data']=temp1
        df['label']=self.metadata
        df['label'].replace(self.dictionary,inplace=True)
        df['filename']=filename
        return df
    def get_vowel_e(self):
        temp1=[]
        filename=[]
        df= pd.DataFrame({'label':[0]*len(self.filedir)})
        for i in tqdm(self.filedir):
            ret,sr=librosa.load(self.root+r"/"+i+r"/"+'vowel-e.wav')
            temp1.append(ret)
            filename.append(self.root+r"/"+i+r"/"+'vowel-e.wav')
        df['data']=temp1
        df['label']=self.metadata
        df['label'].replace(self.dictionary,inplace=True)
        df['filename']=filename
        return df 
    def get_vowel_o(self):
        temp1=[]
        filename=[]
        df= pd.DataFrame({'label':[0]*len(self.filedir)})
        for i in tqdm(self.filedir):
            ret,sr=librosa.load(self.root+r"/"+i+r"/"+'vowel-o.wav')
            temp1.append(ret)
            filename.append(self.root+r"/"+i+r"/"+'vowel-o.wav')
        df['data']=temp1
        df['label']=self.metadata
        df['label'].replace(self.dictionary,inplace=True)
        df['filename']=filename
        return df

   

class Extract_Data():
    """Extracting Different Spectral and Temporal Audio Features"""
    signals=[]
    def __init__(self,data):
        self.data=data['data']
    
    def get_mfcc(self):
        temp=[]
        sr=22050
        for data in self.data:
            S = librosa.feature.melspectrogram(data,n_mels=128,sr=sr) 
            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13) #we are taking the first 13 components of MFCCs
            temp.append(mfcc)
        return temp
    
    def get_deltamfcc(self):
        temp=[]
        sr=22050
        for data in self.data:
            S = librosa.feature.melspectrogram(data,n_mels=128,sr=sr)
            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
            delta_mfcc  = librosa.feature.delta(mfcc)
            temp.append(delta_mfcc)
        return temp

    def get_delta2mfcc(self):
        temp=[]
        sr=22050
        for data in self.data:
            S = librosa.feature.melspectrogram(data,n_mels=128,sr=sr)
            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
            delta2_mfcc  = librosa.feature.delta(mfcc,order=2)
            temp.append(delta2_mfcc)
        return temp
    
        
    def get_rms(self):
        temp=[]
        for data in self.data:
            S, phase = librosa.magphase(librosa.stft(data))
            rms = librosa.feature.rms(S=S)
            temp.append(rms)
        return temp
    
    def get_zero_crossing(self):
        temp=[]
        for data in self.data:
            zero = librosa.feature.zero_crossing_rate(data)
            temp.append(zero)
        return temp
    
    def get_spectral_centroid(self):
        temp=[]
        for data in self.data:
            S_, phase_ = librosa.magphase(librosa.stft(y=data))
            sp_cnt=librosa.feature.spectral_centroid(S=S_)
            temp.append(sp_cnt)
        return temp
    
    def roll_off(self):
        temp=[]
        sr=22050
        for data in self.data:
            S, phase = librosa.magphase(librosa.stft(data))
            roll = librosa.feature.spectral_rolloff(S=S, sr=sr)
            temp.append(roll)
        return temp
    
    def get_chromagram(self):
        sr=22050
        temp=[]
        for data in self.data:
            y_harmonic, y_percussive = librosa.effects.hpss(data)
            S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
            log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36)
            temp.append(chroma)
        return temp
    
    def get_tempogram(self):
        sr=22050
        temp=[]
        hop_length = 512
        for data in self.data:
            oenv = librosa.onset.onset_strength(y=data, sr=sr, hop_length=hop_length)
            tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                      hop_length=hop_length)
            temp.append(tempogram)
        return temp
    
    
    def onset(self):
        temp=[]
        sr=22050
        for data in self.data:
            o_env = librosa.onset.onset_strength(data, sr=sr)
            times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
            temp.append(len(onset_frames))
        return temp
    
    

class get_statistics():
    """Getting Different Statistical Features From Different Spectral and Temporal Audio Features"""
    
    def mean(data):
        temp=[]
        for D in data:
            mean=np.mean(D)
            temp.append(mean)
        return temp
    
    def median(data):
        temp=[]
        for D in data:
            median=np.median(D)
            temp.append(median)
        return temp
    
    def maximum(data):
        temp=[]
        for D in data:
            maximum=np.max(D)
            temp.append(maximum)
        return temp
    
    def minimum(data):
        temp=[]
        for D in data:
            minimum=np.min(D)
            temp.append(minimum)
        return temp
    
    def std(data):
        temp=[]
        for D in data:
            std=np.std(D)
            temp.append(std)
        return temp
    
    def quartile_1st(data):
        temp=[]
        for D in data:
            qtl=np.quantile(D,0.25)
            temp.append(qtl)
        return temp
    
    def quartile_3rd(data):
        temp=[]
        for D in data:
            qtl=np.quantile(D,0.75)
            temp.append(qtl)
        return temp
    
    def interquartilerange(data):
        temp=[]
        for D in data:
            iqr = scipy.stats.iqr(D)
            temp.append(iqr)
        return temp

    def kurtosis(dataset):
        temp=[]
        for data in dataset:
            for D in data:
                kurtosis = scipy.stats.kurtosis(D)
                temp.append(kurtosis)
        return temp
      
    def skewness(dataset):
        temp=[]
        for data in dataset:
            for D in data:
                skewness = scipy.stats.skew(D)
                temp.append(skewness)
        return temp
    
    def rms(data):
        RMS = []
        for rms_ in data:
            rms=[]
            samples = 1
            n = len(rms_[0])
            for idx in range(samples):
                temp = rmsValue(rms_[idx],n)
                rms.append(temp)
            RMS.append(rmsValue(rms,len(rms)))
        return RMS
    
    

            
        
def postprocess(df,labels):
    """function that will compute all the statistics automatically and return the dataframe"""
    for label in labels:
        df[label+str('_mean')]=get_statistics.mean(df[label])
        df[label+str('_std')]=get_statistics.std(df[label])
        df[label+str('_median')]=get_statistics.median(df[label])
        df[label+str('_max')]=get_statistics.maximum(df[label])
        df[label+str('_min')]=get_statistics.minimum(df[label])
        df[label+str('_quartile_1st')]=get_statistics.quartile_1st(df[label])
        df[label+str('_quartile_3rd')]=get_statistics.quartile_3rd(df[label])
        df[label+str('_interquartilerange')]=get_statistics.interquartilerange(df[label])
        df[label+str('_skewness')]=get_statistics.skewness(df[label])
        df[label+str('_kurtosis')]=get_statistics.kurtosis(df[label])
        df[label+str('_rms')]=get_statistics.rms(df[label])
        
        