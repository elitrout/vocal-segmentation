"""
Extract Vocal Variance feature from audio
"""

import essentia
import essentia.standard
from essentia.standard import *
import numpy

loader = essentia.standard.MonoLoader(filename = 'sample.mp3', sampleRate=22050)

audio = loader()

w = Windowing(type = 'hann')
spectrum = Spectrum()
mfcc = MFCC()
mfccs = []

for frame in FrameGenerator(audio, frameSize=4096, hopSize=1024):
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    mfccs.append(mfcc_coeffs)

mfccs = essentia.array(mfccs)
vv = numpy.empty([len(mfccs), 5])

for i in range(len(mfccs)):
    for j in range(5):
        # variance of first five MFCCs (excluding the 0th) over 11 successive frames centered on the current frame
        vv[i][j] = numpy.var(mfccs[max(0, i-5) : min(len(mfccs), i+6), j+1])

