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

def smooth(y,box_pts):
    box=np.ones(box_pts)/box_pts
    y_smooth=np.convolve(y,box,mode="same")
    return y_smooth
x1=data[:,:,0]

acor=np.zeros([noRG,2*noDP-1])
acorp=np.zeros([noRG,2*noDP-1])
for i in range(noRG):
    acor[i,:]=signal.correlate(x1[i,:],x1[i,:],method='direct',mode='full')#acor for all ranges
    acorp[i,:]=20*np.log10(abs(acor[i,:]))#power of the acor/db
onesn=np.zeros([noRG])
datas=abs(np.array(data[:,:,0]))
for i in range(noRG):
    z=0
    c=max(acorp[i,:])/2
    hp=int(np.round((2*noDP-1)/2))
    z=np.where(acorp[i,hp::-1]<c)
    z=np.array(z)
    if len(z[0,:])==0:
        datas[i,:]=smooth(datas[i,:], noDP)
        onesn[i]=noDP
    if len(z[0,:])>0:
        z=z[0,0]
        datas[i,:]=smooth(datas[i,:], z)
        onesn[i]=z
        
        
Spectrs=np.zeros([noRG,noDP])+1j*np.zeros([noRG,noDP])
for rg in range(noRG):
    fs,Spectrs[rg,:]=make_fft(t,datas[rg,:])


datas=10*np.log10(datas)
Spectrs=10*np.log10(abs(Spectrs))
#pcolor
plt.figure(figsize=(10,10 ))
plt.subplot(2,2,1)
plt.suptitle("First reciever Amplitude and Spectra ",fontsize=20)
plt.title('Raw amp-before smoothing')
plt.pcolor(t,ranges,10*np.log10(abs(data[:,:,0])),cmap='jet')
plt.xlabel('Time')
plt.ylabel('Ranges')
plt.colorbar()
plt.subplot(2,2,2)
plt.pcolor(t,ranges,10*np.log10(abs(Spectr[:,:,0])),cmap='jet')
plt.title('Raw Spectr-before smoothing')
plt.xlabel('F/hz')
plt.colorbar()
plt.subplot(2,2,3)
plt.title('Raw amp -after smoothing')
plt.pcolor(t,ranges,datas,cmap='jet')
plt.xlabel('Time')
plt.ylabel('Ranges')
plt.colorbar()
plt.subplot(2,2,4)
plt.pcolor(fs,ranges,Spectrs,cmap='jet')
plt.title('Raw Spectr-after smoothing')
plt.xlabel('F/hz')
plt.colorbar()

#line plot for the resulting correlation length for each altitude used to smooth the time series
plt.figure()
plt.plot(ranges,onesn)
plt.grid()
plt.xlabel('ranges')
plt.ylabel('smoothed data')