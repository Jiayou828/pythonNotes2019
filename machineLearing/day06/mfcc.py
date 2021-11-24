# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import python_speech_features as sf
import matplotlib.pyplot as mp
sample_rate, sigs = wf.read(
    '../../data/speeches/training/orange/orange01.wav')
mfcc = sf.mfcc(sigs, sample_rate)
print(mfcc.shape)
mp.matshow(mfcc.T, cmap='gist_rainbow',
           fignum='MFCC')
mp.title('MFCC', fontsize=16)
mp.xlabel('Sample', fontsize=12)
mp.ylabel('Feature', fontsize=12)
mp.tick_params(which='both', top=False,
               labeltop=False, labelbottom=True,
               labelsize=10)
mp.show()
