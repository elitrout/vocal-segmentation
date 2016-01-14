"""
Extract Fluctogram feature from audio
"""

import essentia
import essentia.standard
from essentia.standard import *
import numpy as np

FRAMESIZE = 4096
HOPSIZE = 1024
SAMPLERATE = 22050.0

loader = essentia.standard.MonoLoader(filename = 'sample.mp3', sampleRate=SAMPLERATE)

audio = loader()

w = Windowing(type = 'hann', size=FRAMESIZE)
spectrum = Spectrum()
spectrogram = []

for frame in FrameGenerator(audio, frameSize=FRAMESIZE, hopSize=HOPSIZE):
    dft = spectrum(w(frame))
    spectrogram.append(dft)

spectrogram = essentia.array(spectrogram)

# Map the spectrum into pitch scale
pitchScale = []    # Value is spectrum index

for i in range(120*6+1):    # 10 bins per semitone. 6 octave.
    freq = 164.814 * 2 ** (i / 120.0)    # Lowest note is E3 (164.814 Hz)
    idx = int(np.round(freq / SAMPLERATE * (FRAMESIZE / 2)))
    pitchScale.append(idx)

# 240 bins per band. 17 bands.
for i in range(17):
    freqRange = [pitchScale[i * 30], pitchScale[i * 30 + 240]]
    # Weight each band by triangle window
    weightedSpectrogram = spectrogram
    bandwidth = freqRange[1] - freqRange[0]
    wTri = np.empty([bandwidth, ])
    for j in range(bandwidth):
        wTri[j] = 1 - abs(2.0 / (bandwidth - 1) * ((bandwidth - 1) / 2.0 - j))
    for k in len(spectrogram):
        weightedSpectrogram[k, freqRange[0] : freqRange[1]] = wTri * weightedSpectrogram[k, freqRange[0] : freqRange[1]]
