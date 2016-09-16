# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:25:27 2016

@author: martin

#attempt at fast smoothing in time as a preprocessing/denoising step
#registration still not good enough
"""
import numpy as np
import imp
from matplotlib.pyplot import imshow
import csv
import time
import cPickle as pickle
import sklearn.preprocessing
import scipy.signal
import copy
from tifffile2 import imsave
import pdb
TiffFile=imp.load_source('TiffFile','/Users/martin/Documents/quantifly3d-martin/tifffile2.py')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg.tif')

tiffarray=atiff.asarray(memmap=True)
meta = atiff.series[0]
order = meta.axes
shape=meta.shape
print shape
print meta
[tpts,zs,cs,ys,xs]=shape

def get_tiff_slice(tiffarray,tpt=[0],zslice=[0],x=[0],y=[0],c=[0]):
    tiff=(tiffarray[:,zslice,c,y,x])
    #tiff.SetSpacing((1,1,3))
    return tiff
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float,axis=0)
    ret[n:,:,:,:,:] = ret[n:,:,:,:,:] - ret[:-n,:,:,:,:]
    return ret[:,:,:,:,:] / n
image= scipy.ndimage.filters.convolve1d(tiffarray.astype('float32'), [1,1,1], axis=0, mode='reflect')
image=image/image.max()*255
#image[0,:,:,:,:]=(3/2)*image[0,:,:,:,:]
#image[-1,:,:,:,:]=(3/2)*image[-1,:,:,:,:]
'''
image = np.zeros([tpts,zs,cs,ys,xs], 'float32')
for z in range(zs):
    for x in range(xs):
        for y in range(ys):
            for c in range(cs):
                image[:,z,cs,ys,xs]=scipy.signal.convolve(get_tiff_slice(tiffarray,tpt=range(tpts),zslice=z,x=x,y=y,c=c),np.ones(4),mode='same')'''
     
imsave('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_smooth.tif',image.astype('uint8'), imagej=True)

print 'Prediction written to disk'
