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

mean=np.zeros([noRG,2])
for rx in range(2):
    mean[:,rx]=10*np.log10(np.mean(np.abs(data[:,:,rx]),1))
plt.figure()
plt.title('Mean values')
plt.plot(mean[:,0],ranges,mean[:,1],ranges)
plt.legend(['RX1','RX2'])
plt.xlabel('Power/DB')
plt.ylabel('ranges /km')
plt.grid()
# for 1-2 recievers
median=np.zeros([noRG,2])
for rx in range(2):
    median[:,rx]=10*np.log10(np.median(np.abs(data[:,:,rx]),1))
plt.figure()
plt.title('Median values')
plt.plot(median[:,0],ranges,median[:,1],ranges)
plt.legend(['RX1','RX2'])
plt.xlabel('Power/DB')
plt.ylabel('ranges /km')
plt.grid()
#STD for 1-2 recievers
std=np.zeros([noRG,2])
for rx in range(2):
    std[:,rx]=10*np.log10(np.std(np.abs(data[:,:,rx]),1))
plt.figure()
plt.title('STD values')
plt.plot(std[:,0],ranges,std[:,1],ranges)
plt.legend(['RX1','RX2'])
plt.xlabel('stand. dev values for all ranges')
plt.ylabel('ranges /km')
plt.grid()