#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:06:58 2018

@author: lemn
"""

import wave
import numpy as np
import math
import matplotlib.pyplot as plt
import struct
import os

def entropyVad(s,fs):
    t = 200 #20ms
    frameLength = fs*t//1000
    frames = [s[i:i+frameLength] for i in range(0,len(s),frameLength)]
    entropys = []
    dics = {}
    for fid,frame in enumerate(frames):
        dic = {}
        for d in frame:
            if d in dic.keys():
                dic[d]=dic[d]+1
            else:
                dic[d] = 1
            if d in dics.keys():
                dics[d] = dics[d]+1
            else:
                dics[d] = 1
        ns = np.array([dic[key] for key in dic.keys()])
        ps = ns/len(frame)
        logps = [math.log(p) for p in ps]
        entropy = -sum(np.array(ps)*np.array(logps))
        entropys.append(entropy)
    enthreshold = np.mean(entropys)
    tags = np.array(entropys)>enthreshold
    tags = np.repeat(tags,frameLength)
    return (tags[0:len(s)],entropys)

if __name__ == '__main__':
    wavfile = '/home/lemn/dataSet/ASVspoof2017/ASVspoof2017_dev/D_1000373.wav'
    f = wave.open(wavfile,'rb')
    params = f.getparams()
    nchannels,sampwidth,framerate,nframes = params[:4]
    str_data = f.readframes(nframes)
    wav_data = np.fromstring(str_data,dtype=np.short)
    data = wav_data
    wav_data = wav_data/max(np.abs(wav_data))
    Max = max(wav_data)
    Min = min(wav_data)
    tags,entropys = entropyVad(wav_data,framerate)
    plt.plot(range(len(entropys)),entropys)
    plt.xlabel('frameId')
    plt.ylabel('entropy')
    plt.show()
    flags = [0.5 if tags[t] else -0.5 for t in range(len(tags))]
    plt.plot(np.array(range(nframes))/framerate,flags,'r',label='VAD')
    plt.plot(np.array(range(nframes))/framerate,wav_data,'b',label='Signal')
    plt.xlabel('time/s')
    plt.ylabel('signal-data')
    f0 =wave.open('myD_1000373.wav','wb')
    plt.legend(prop={'size':10})
    f0.setparams(f.getparams())
    s = []
    for i in range(len(data)):
        if tags[i]:
            s.append(data[i])
    f0.setnframes = len(s)
    wav_s = [struct.pack('h',d) for d in s]
    wav_s = b''.join(wav_s)
    f0.writeframesraw(wav_s)
    f.close()
    f0.close()
    
