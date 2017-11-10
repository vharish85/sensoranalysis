
# coding: utf-8

# In[1]:

#Signal Processing - Feature Extraction
#KitCat 

#References

#1. https://docs.scipy.org/doc/scipy/reference/signal.html
#2. 


# In[2]:

#Import Library

from numpy import *
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from matplotlib import pyplot as plt
import scipy.ndimage
import math
import os
import csv


# In[3]:

def readFile(inputfile):
    print("readFile")
    data= pd.read_csv(inputfile)
    return data


# In[4]:

def dist(df):
    df['Ax^2'] = df['Ax']**2
    df['Ay^2'] = df['Ay']**2
    df['Az^2'] = df['Az']**2

    df['Gx^2'] = df['Gx']**2
    df['Gy^2'] = df['Gy']**2
    df['Gz^2'] = df['Gz']**2

    df['A']=(df['Ax^2']+df['Ay^2']+df['Az^2'])
    df['sqrtA'] = np.sqrt(df['A'])
    df['G']=(df['Gx^2']+df['Gy^2']+df['Gz^2'])
    df['sqrtG'] = np.sqrt(df['G'])
    return df


# In[5]:

def filesave(data,fname):
    path= "%s%s.%s" % ("C:\\Users\\usb7kor\\Desktop\\kitcat\\Featuresssss\\",fname,"csv")
    data=str(data).replace('[','').replace(']','')
    with open(path, 'w') as output:
        output.write(str(data))   


# In[6]:

def autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result


# In[7]:

def corr(a,b):
    corr=np.corrcoef(a,b)
    return corr


# In[8]:

def powerEnergy(x,fs,time,amp,noise_power):
    
    # Estimate PSD `S_xx_welch` at discrete frequencies `f_welch`
    f_welch, S_xx_welch = scipy.signal.welch(x, fs=fs)

    # Integrate PSD over spectral bandwidth
    # to obtain signal power `P_welch`
    df_welch = f_welch[1] - f_welch[0]
    P_welch = np.sum(S_xx_welch) * df_welch

    # Compute DFT
    Xk = np.fft.fft(x)

    # Compute corresponding frequencies
    dt = time[1] - time[0]
    f_fft = np.fft.fftfreq(len(x), d=dt)

    # Estimate PSD `S_xx_fft` at discrete frequencies `f_fft`
    T = time[-1] - time[0]
    S_xx_fft = ((np.abs(Xk) * dt) ** 2) / T
    

    # Integrate PSD over spectral bandwidth to obtain signal power `P_fft`
    df_fft = f_fft[1] - f_fft[0]
    P_fft = np.sum(S_xx_fft) * df_fft

    # Power in sinusoidal signal is simply squared RMS, and
    # the RMS of a sinusoid is the amplitude divided by sqrt(2).
    # Thus, the sinusoidal contribution to expected power is
    P_exp = (amp / np.sqrt(2)) ** 2 

    # For white noise, as is considered in this example,
    # the noise is simply the noise PSD (a constant)
    # times the system bandwidth. This was already
    # computed in the problem statement and is given
    # as `noise_power`. Simply add to `P_exp` to get
    # total expected signal power.
    P_exp += noise_power

    # Energy obtained via "integrating" over time
    E = np.sum(x ** 2)

    # Energy obtained via "integrating" DFT components over frequency.
    # The fact that `E` = `E_fft` is the statement of 
    # the discrete version of Parseval's theorem.
    N = len(x)
    E_fft = np.sum(np.abs(Xk) ** 2) / N

    # Signal energy from Welch's PSD
    E_welch = (1. / dt) * (df_welch / df_fft) * np.sum(S_xx_welch)
    return E_welch,E,P_exp,S_xx_fft,f_fft,Xk,P_welch


# In[9]:

def skewness(data):
    skewness=scipy.stats.skew(data,axis=0,bias=True)
    return skewness


# In[10]:

def kurtosis(data):
    kurtosiss=scipy.stats.kurtosis(data,axis=0,bias=True)
    return kurtosiss


# In[11]:

def entropy(data):
    entropy=scipy.stats.entropy(data)
    return entropy


# In[12]:

def periodogram(data):
    sampleFreqArray,spectralDensity=scipy.signal.periodogram(data,fs=50,axis=-1)
    return sampleFreqArray,spectralDensity


# In[13]:

def csd(data1,data2):
    sampleFreqArray,csdd=scipy.signal.csd(data1,data2,fs=50,axis=-1)
    return sampleFreqArray,csdd


# In[14]:

def coherence(data1,data2):
    f,coherences=scipy.signal.coherence(data1,data2,fs=50,axis=-1)
    return f,coherences


# In[15]:

##
#MAIN function.....
##
''''
ap=argparse.ArgumentParser()
ap.add_argument("-i", required=True,  help="path to the input file")
ap.add_argument("-r", required=False, help="destination directory for saving Results")
ap.add_argument("-p", required=False, help="path")
ap.add_argument("-n", required=False, help="file name")

args = vars(ap.parse_args())
workingdir=args["p"]
inputdata=args["i"]
uniqueFilename=args["n"]
result_filepath=args["r"]
'''

#Read Input Data
#data = readFile("C:\\Users\\subba\\Desktop\\Hackathon\\Data\\w1.csv")
data = readFile("C:\\Users\\usb7kor\\Desktop\\kitcat\\sample.csv")

fs = 50 #Sampling Frequency - 50Hz
N = len(data)
time = np.arange(N) / fs
amp = 2*np.sqrt(2)
noise_power = 0.001 * fs / 2

data=dist(data)

#Define variables
bufA=data.A
bufG=data.G

autoCorrA=autocorrelation(bufA)
autoCorrG=autocorrelation(bufG)

correlationFactor=corr(bufA,bufG)

Energy_WelchA,EnergyA,RMSA,psdA,f_fftA,discreteFFTA,P_welchA=powerEnergy(bufA,fs,time,amp,noise_power)
Energy_WelchG,EnergyG,RMSG,psdG,f_fftG,discreteFFTG,P_welchG=powerEnergy(bufG,fs,time,amp,noise_power)

skewA=skewness(bufA)
skewG=skewness(bufG)

kurtA=kurtosis(bufA)
kurtG=kurtosis(bufG)

entropyA=entropy(bufA)
entropyG=entropy(bufG)

sampleFreqArrayA,spectralDensityA=periodogram(bufA)
sampleFreqArrayG,spectralDensityG=periodogram(bufG)

sampleFreq,csd1=csd(bufA,bufG)

f,cxy=coherence(bufA,bufG)


buffer=[]
#buffer.append('A')
#buffer.append('G')
buffer.append('minA')
buffer.append('maxA')
buffer.append('minG')
buffer.append('maxG')
buffer.append('A_maxAutoCorr')
buffer.append('A_minAutoCorr')
buffer.append('G_maxAutoCorr')
buffer.append('G_minAutoCorr')
buffer.append('Energy_A')
buffer.append('Energy_G')
buffer.append('Energy(welch)_A')
buffer.append('Energy(welch)_G')
buffer.append('Power_A')
buffer.append('Power_G')
buffer.append('RMS_A')
buffer.append('RMS_G')
buffer.append('minPSD_A')
buffer.append('maxPSD_A')
buffer.append('minPSD_G') 
buffer.append('maxPSD_G')  
buffer.append('Skewness_A')
buffer.append('Skewness_G')
buffer.append('Kurtosis_A')
buffer.append('Kurtosis_G')
buffer.append('Entropy_A')
buffer.append('Entropy_G')
buffer.append('Min_spectralDensity_A')
buffer.append('Max_spectralDensity_A')
buffer.append('Min_spectralDensity_G')
buffer.append('Max_spectralDensity_G')

fname1= "%s" % ("metrics")
filesave(buffer, fname1)

path= "%s%s.%s" % ("C:\\Users\\usb7kor\\Desktop\\kitcat\\Featuresssss\\",fname1,"csv")
buffer1=[]

#buffer1.append(A)
#buffer1.append(G)
buffer1.append(min(bufA))
buffer1.append(max(bufA))
buffer1.append(min(bufG))
buffer1.append(max(bufG))
buffer1.append(max(autoCorrA))
buffer1.append(min(autoCorrA))
buffer1.append(max(autoCorrG))
buffer1.append(min(autoCorrG))
buffer1.append(EnergyA)
buffer1.append(EnergyG)
buffer1.append(Energy_WelchA)
buffer1.append(Energy_WelchG)
buffer1.append(P_welchA)
buffer1.append(P_welchG)
buffer1.append(RMSA)
buffer1.append(RMSG)
buffer1.append(min(psdA))
buffer1.append(max(psdA))
buffer1.append(min(psdG)) 
buffer1.append(max(psdG))  
buffer1.append(skewA)
buffer1.append(skewG)
buffer1.append(kurtA)
buffer1.append(kurtG)
buffer1.append(entropyA)
buffer1.append(entropyG)
buffer1.append(min(spectralDensityA))
buffer1.append(max(spectralDensityA))
buffer1.append(min(spectralDensityG))
buffer1.append(max(spectralDensityG))

with open(path,'a',newline='') as f:
    writer=csv.writer(f)
    writer.writerow([])
    writer.writerow(buffer1)


# In[16]:

'''
f1  = pd.read_csv("C:\\Users\\subba\\Desktop\\Hackathon\\w1.csv")
x=f1.Ax
fs = 10e3
N = 1e5
time = np.arange(N) / fs
Energy_Welch,Energy,Power=powerEnergy(x,fs,time,N)
print("Energy(welch) of signals:",Energy_Welch)
print("Energy of signals:",Energy)
print("Power of signals:",Power)
'''


# In[17]:

'''
#Spectral Analysis - Welch’s Method
#Noise Immunity

fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 1270.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

f, Pwelch_spec = signal.welch(x, fs, scaling='spectrum')

plt.semilogy(f, Pwelch_spec)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.grid()
plt.show()
'''


# In[18]:

'''
def correlation():
    sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
    sig_noise = sig + np.random.randn(len(sig))
    corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128
    print(corr)
    return sig,sig_noise,corr
sig,sig_noise,corr=correlation()

clock = np.arange(64, len(sig), 128)
fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(sig)
ax_orig.plot(clock, sig[clock], 'ro')
ax_orig.set_title('Original signal')
ax_noise.plot(sig_noise)
ax_noise.set_title('Signal with noise')
ax_corr.plot(corr)
ax_corr.plot(clock, corr[clock], 'ro')
ax_corr.axhline(0.5, ls=':')
ax_corr.set_title('Cross-correlated with rectangular pulse')
ax_orig.margins(0, 0.1)
fig.tight_layout()
fig.show()

'''

