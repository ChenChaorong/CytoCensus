# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:07:48 2016

@author: martin

#Create graphs of dividing cells
    Includes some rudimentary tracking
    Currently very  much a work-in-progress
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
import pdb
TiffFile=imp.load_source('TiffFile','/Users/martin/Documents/quantifly3d-martin/tifffile2.py')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_DivNBCC1Mask.tif')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_NBmodelCC1Mask.tif')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_NBmodelCC1_Prediction.tif')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_NBmodelCC1_Hess.tif')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_DivNBCC1_Prediction.tif')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_DblTrnNB_Prediction.tif')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_DivNBCC1_Prediction.tif')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_divNBModeldouble_Prediction.tif')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_dblTrnDivNB2_Prediction.tif')


atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_combDivNBmodel_Prediction.tif')
atiff=TiffFile.TiffFile('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_dblCombDivNB_Prediction.tif')

tiffarray=atiff.asarray(memmap=True)
meta = atiff.series[0]
order = meta.axes
shape=meta.shape
print shape
print meta
[tpts,zs,xs,ys]=shape
def get_tiff_slice(tiffarray,tpt=[0],zslice=[0],x=[0],y=[0],c=[0]):
    tiff=np.squeeze((tiffarray[tpt,:,:,:][:,zslice,:,:][:,:,y,:][:,:,:,x]))
    #tiff.SetSpacing((1,1,3))
    return tiff
    
get_tiff_slice(tiffarray,tpt=[0],zslice=[0],x=[0],y=[0],c=[0])

npts=[]
for num in range(tpts):
    npts.append([])
with open('/Users/martin/Pictures/20150602/20150602_JupGFP_HisRFP_60x_L3_resized_reg_DblTrnNB_outputPoints.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    firstline=0

    for row in spamreader:
        a=row[5].split(',')
        if firstline==0:
            firstline=1
        else:
            '''for num in range(tpts):
                if int(a[2])==num+1:
                    npts[num].append((int(a[3]),int(a[4]),int(a[5])))'''
            for num in range(tpts):
                if int(a[2])==num+1:
                    npts[num].append((int(a[3]),int(a[4]),int(a[5])))



pts=copy.deepcopy(npts[50])
out=[]
out_track=[]
dist=[16,16,8]
for num,tpt in enumerate(range(tpts)):
    out.append([])
    out_track.append([])
    print tpt
    ts=get_tiff_slice(tiffarray,[tpt],range(zs),range(xs),range(ys),c=[0]).copy()
    for numll,obj in enumerate(pts):
        #make spheroid representing objects
        [a,b,c]=dist
        [al,bl,cl]=[2*x+1 for x in dist]
        [asm,bsm,csm]=np.mgrid[0:al,0:bl,0:cl]   
        selem=(np.square(asm-a)/(a*a)+np.square(bsm-b)/(b*b)+np.square(csm-c)/(c*c))<=1
        '''
        selem=skimage.morphology.ball(np.round(dist[0],0)).astype('bool')
        if dist[2] is not 0:
            drange=range(selem.shape[0]/2,selem.shape[0],np.round(dist[0]/dist[2],0).astype('uint8'))
            lrange=range(0,selem.shape[0]/2,np.round(dist[0]/dist[2],0).astype('uint8'))
            selem2=selem[np.newaxis,lrange+drange,np.newaxis,:,:]'''
        #for positions within spheroid sum values at object
        [ix,iy,iz]=np.meshgrid(range(obj[0]-a,obj[0]+a+1),range(obj[1]-b,obj[1]+b+1),range(obj[2]-c,obj[2]+c+1))
        [x,y,z]=np.where(selem[asm,bsm,csm])
        '''
        x=[i-a+obj[0] for i in x]
        y=[i-b+obj[1] for i in y]
        z=[i-c+obj[2] for i in z]
        '''
        x=list(x)
        y=list(y)
        z=list(z)
        remove=[]
        for c in range(len(x)):
            ix=x[c]
            iy=y[c]
            iz=z[c]
            newx=ix-a+obj[0]
            newy=iy-b+obj[1]
            newz=iz-a+obj[2]
            if newx<0 or newx>xs or newy<0 or newy>ys or newz<0 or newz>zs:#boundary check
                remove.append(c)
            else:
                x[c]=int(newx)
                y[c]=int(newy)
                z[c]=int(newz)
        remove.reverse()
        for c in remove:
            del x[c],y[c],z[c]
        im=ts[z,x,y]
        '''sg=np.mgrid[(obj[2]-1):(obj[2]+2),obj[0]-2:obj[0]+3,obj[1]-2:obj[1]+3]
        opt=ts[sg]'''
        #nm=im.argmax()
        gh=[]
        for i in range(len(x)):
            for pt in npts[tpt]:
                if pt == (x[i],y[i],z[i]) and pt not in npts[tpt]:
                    gh.append(pt)

        if len(gh)>0:
            newpt=gh[int(len(gh)/2)]
            #pts[numll]=copy.deepcopy(gh[int(len(gh)/2)])
            #pts[numll]=tuple([int((pts[numll][i]+newpt[i])/2) for i in range(3)])
            print 'update'
        #pts[numll]=(int((x[nm]+obj[0])/2),int((y[nm]+obj[1])/2),int((z[nm]+obj[2])/2))
        #st=[x[nm],y[nm],z[nm]]
        #ex=[]
        #for i in range(len(st)):
        #    ex.append(int((st[i]+obj[i])/2))
        #pts[numll]=st
        #pts[numll]=(sg[nm])
        out_track[num].append(pts[numll][2])
        out[num].append(sum(im))
out_array=np.asarray(copy.deepcopy(out))
import matplotlib.pyplot as plt
for i in range(out_array.shape[1]):
        
        #x=scipy.signal.convolve(np.array(x), np.ones(5),mode='same')
        out_array[:,i]=sklearn.preprocessing.scale(out_array[:,i]+np.roll(out_array[:,i],1),copy=True)
        plt.plot(out_array[:,i])
        plt.axis([None,None,-2.5,2.5])
        plt.savefig('NB_'+str(i)+'_'+str(dist)+'.png', bbox_inches='tight')
        plt.cla()