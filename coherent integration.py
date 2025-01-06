import scipy.io as spio
from scipy import signal
import os
import matplotlib.pyplot as plt
import numpy as np
#importing the signal
DataPath='C:/Users/Karam/Desktop/masters/summer semester/telecommunications project/files/'
Files=os.listdir(DataPath)
currentfile=str(DataPath)+str(Files[0])
mat = spio.loadmat(currentfile, squeeze_me=True)
datenums=mat['datenums']
ranges=mat['ranges']
data=mat['data']
time=(datenums-np.floor(np.min(datenums)))*24#hours
timesample=(time[1]-time[0])*60*60   # delta-t : sampling distance in time [s]
noRG=np.size(data,0)
noDP=np.size(data,1)
noRx=np.size(data,2)
t=time*60*60#convert from hours to seconds
def make_ci(t, y, ci):
    nptsn=int(np.floor(len(y)/ci))
    yn=np.empty(nptsn)+1j*np.empty(nptsn)
    tn=np.empty(nptsn)
    for i in range(0,nptsn):
        yn[i]=np.mean(y[i*ci:i*ci+ci-1])
        tn[i]=np.mean(t[i*ci:(i+1)*ci])
    return tn,yn
ci=7
noDPn=len(data[1,:,1])/ci
noDPn=int(noDPn)
datan=np.zeros([noRG,noDPn,noRx])+1j*np.zeros([noRG,noDPn,noRx])
for rx in range(noRx):
    for rg in range(noRG):
        tn,datan[rg,:,rx]=make_ci(t,data[rg,:,rx],ci)
def make_fft(t,y):
    dt = t[1]-t[0] # dt -> temporal resolution ~ sample rate
    f = np.fft.fftfreq(t.size, dt) # frequency axis
    Y = np.fft.fft(y)   # FFT
    f=np.fft.fftshift(f)
    Y= np.fft.fftshift(Y)/(len(y))
    return f,Y
Spectr=np.zeros([noRG,noDP,noRx])+1j*np.zeros([noRG,noDP,noRx])
for rg in range(noRG):
    for rx in range(noRx):
        f,Spectr[rg,:,rx]=make_fft(t,data[rg,:,rx])

# Spectra for averaged for all ranges and the first reciever integrated time series
Spectrn=np.zeros([noRG,noDPn,noRx])+1j*np.zeros([noRG,noDPn,noRx])
for rg in range(noRG):
    for rx in range(noRx):
        fn,Spectrn[rg,:,rx]=make_fft(tn,datan[rg,:,rx])

i=0#range
plt.figure(figsize=(13,10 ))
plt.suptitle("Coherent integration ",fontsize=20)
plt.subplot(2,2,1)
plt.plot(t,data[i,:,0],'g')
plt.grid()
plt.xlabel('Time/sec')
plt.ylabel('Amplitude')
plt.title('First range signal for the first sensor before using ci')
plt.subplot(2,2,2)
plt.plot(f,abs(Spectr[i,:,0]),'g')
plt.grid()
plt.xlabel('Frequency/HZ')
plt.ylabel('Amplitude')
plt.title('First range Spectra for the first sensor before using ci')
plt.subplot(2,2,3)
plt.plot(tn,datan[i,:,0],'r')
plt.xlabel('Time/sec')
plt.ylabel('Amplitude')
plt.title('First range signal for the first sensor after applying ci')
plt.grid()
plt.subplot(2,2,4)
plt.plot(fn,abs(Spectrn[i,:,0]),'r')
plt.xlabel('Frequency/HZ')
plt.ylabel('Amplitude')
plt.title('First range Spectra for the first sensor after applying ci')
plt.grid()