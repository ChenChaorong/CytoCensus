# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:25:35 2016

@author: martin

Feature calculation methods (moved from v2_functions for clarity)
 makes adding new features more straightforward
 might be nice to add subclasses to this
"""
import vigra
import scipy
import PIL.Image
from skimage.filters.rank import entropy
from skimage import feature as skfeat

from skimage import exposure
from skimage import morphology
from sklearn import linear_model
from sklearn import preprocessing
import scipy.ndimage as ndimage
import scipy.signal as signal
from scipy import special
import skimage
import time
import numpy as np
import threading
#from v2_functions import get_tiff_slice
from scipy.ndimage.interpolation import shift
from skimage.feature import daisy
from sklearn.decomposition import PCA,FastICA
from sklearn import ensemble
from sklearn import tree
from sklearn.pipeline import Pipeline
from scipy.ndimage.morphology import distance_transform_edt

def RF(par_obj, RF_type='ETR'):

    if RF_type== 'ETR':
        method = ensemble.ExtraTreesRegressor(par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1)
    elif RF_type== 'GBR':
        method = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.01, n_estimators=par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features)  
    elif RF_type== 'GBR2':
        method = ensemble.GradientBoostingRegressor(loss='lad', learning_rate=0.1, n_estimators=par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features)  
    elif RF_type== 'ABR':
        method = ensemble.AdaBoostRegressor(base_estimator=ensemble.ExtraTreesRegressor(10, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1), n_estimators=3, learning_rate=1.0, loss= 'square')
  
    return method

def get_feature_lengths(feature_type):
    #dictionary of feature sets
    #can add arbitrary feature sets by defining a name, length, and function that accepts two arguments
    feature_dict={'basic': [13,local_shape_features_basic],
                  'fine': [21,local_shape_features_fine],
                  'fine3': [26,local_shape_features_fine3],
                  'imhist': [26,local_shape_features_fine_imhist],
                  'dual': [36, local_shape_features_dual]}

    feat_length=None
    if feature_dict.has_key(feature_type):
        feat_length=feature_dict[feature_type][0]
        feat_func=feature_dict[feature_type][1]
    else:
        raise Exception('Feature set not found')
        
    return feat_length,feat_func

def feature_create_threadable(par_obj,imRGB):
    
    time1 = time.time()
    [feat_length,feat_func]=get_feature_lengths(par_obj.feature_type)
    feat = np.zeros(((int(par_obj.crop_y2)-int(par_obj.crop_y1)),(int(par_obj.crop_x2)-int(par_obj.crop_x1)),feat_length*(par_obj.ch_active.__len__())))
    if par_obj.numCH==0:
        imG = imRGB[:,:].astype(np.float32)

        feat = feat_func(imG,par_obj.feature_scale)  
    else:
        for b in range(0,par_obj.ch_active.__len__()):
            imG = imRGB[:,:,b].astype(np.float32)
            feat[:,:,(b*feat_length):((b+1)*feat_length)] = feat_func(imG,par_obj.feature_scale)
    '''        if b==1:#dirty hack to test
                imG = imRGB[:,:,b].astype(np.float32)*imRGB[:,:,b].astype(np.float32)
                feat[:,:,(2*feat_length):((2+1)*feat_length)]=feat_func(imG,par_obj.feature_scale)'''

    return feat
    
def feature_create_z(par_obj,):
    # calculates simple Z based features
    # somewhat buggy
    # little benefit from these features
    import v2
    #checks if features already in array
    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:
        imRGB = v2.get_tiff_slice(par_obj,[tpt],zslice,range(0,par_obj.ori_width,int(par_obj.resize_factor)),range(0,par_obj.ori_height,int(par_obj.resize_factor)),par_obj.ch_active,imno)

    #imRGBlist.append(imRGB)
    imRGB=imRGB.astype('float32')/ par_obj.tiffarraymax
    
    #If you want to ignore previous features which have been saved.
    int_obj.report_progress('Calculating Features for Z:' +str(zslice+1) +' Timepoint: '+str(tpt+1)+' File: '+str(imno+1))
    feat =feature_create_threadable(par_obj,imRGB)
    
    par_obj.num_of_feat = feat.shape[2]
    par_obj.data_store['feat_arr'][imno][tpt][zslice] = feat

    #not currently intended to be threaded
    time1 = time.time()
    [feat_length,feat_func]=get_feature_lengths(par_obj.feature_type)
    feat = np.zeros(((int(par_obj.crop_y2)-int(par_obj.crop_y1)),(int(par_obj.crop_x2)-int(par_obj.crop_x1)),feat_length*par_obj.ch_active.__len__()))
    if par_obj.numCH==0:
        imG = imRGB[:,:].astype(np.float32)

        feat = feat_func(imG,par_obj.feature_scale)  
    else:
        for b in range(0,par_obj.ch_active.__len__()):
            imG = imRGB[:,:,b].astype(np.float32)
            feat[:,:,(b*feat_length):((b+1)*feat_length)] = feat_func(imG,par_obj.feature_scale)  

    return feat


def feature_create_threadable_auto(par_obj,imno,tpt,zslice):
    #allows auto-context based features to be included
    #currently slow, could optimise to calculate more efficiently
    [feat_length,feat_func]=get_feature_lengths(par_obj.feature_type)
    feat=feat_func(par_obj.data_store['pred_arr'][imno][tpt][zslice],par_obj.feature_scale)
    return feat
def auto_context_features(feat_array):
    #pattern=[1,2,3,5,7,10,12,15,20,25,30,35,40,45,50,60,70,80,90,100]
    pattern=[1,2,4,8,16,32,64,128]
    feat_list=[]
    for xshift in pattern:
        for yshift in pattern:
            a=shift(feat_array, (xshift,0),order=0, cval=0)
            b=shift(feat_array, (-xshift,0),order=0, cval=0)
            c=shift(feat_array, (yshift,0),order=0, cval=0)
            d=shift(feat_array, (-yshift,0),order=0, cval=0)
            feat_list.append(a)
            feat_list.append(b)
            feat_list.append(c)
            feat_list.append(d)
    return feat_list

    
def local_shape_features_fine(im,scaleStart):
    #Exactly as in the Luca Fiaschi paper.
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,21))
    
    st08 = vigra.filters.structureTensorEigenvalues(im,s*1,s*2)
    st16 = vigra.filters.structureTensorEigenvalues(im,s*2,s*4)
    st32 = vigra.filters.structureTensorEigenvalues(im,s*4,s*8)
    st64 = vigra.filters.structureTensorEigenvalues(im,s*8,s*16)
    st128 = vigra.filters.structureTensorEigenvalues(im,s*16,s*32)
    
    f[:,:, 0]  = im
    f[:,:, 1]  = vigra.filters.gaussianGradientMagnitude(im, s,window_size=2.5)
    f[:,:, 2]  = st08[:,:,0]
    f[:,:, 3]  = st08[:,:,1]
    f[:,:, 4]  = vigra.filters.laplacianOfGaussian(im, s ,window_size=2.5)
    f[:,:, 5]  = vigra.filters.gaussianGradientMagnitude(im, s*2,window_size=2.5) 
    f[:,:, 6]  =  st16[:,:,0]
    f[:,:, 7]  = st16[:,:,1]
    f[:,:, 8]  = vigra.filters.laplacianOfGaussian(im, s*2 ,window_size=2.5)
    f[:,:, 9]  = vigra.filters.gaussianGradientMagnitude(im, s*4,window_size=2.5) 
    f[:,:, 10] =  st32[:,:,0]
    f[:,:, 11] =  st32[:,:,1]
    f[:,:, 12] = vigra.filters.laplacianOfGaussian(im, s*4 ,window_size=2.5)
    f[:,:, 13] = vigra.filters.gaussianGradientMagnitude(im, s*8,window_size=2.5) 
    f[:,:, 14] =  st64[:,:,0]
    f[:,:, 15] =  st64[:,:,1]
    f[:,:, 16] = vigra.filters.laplacianOfGaussian(im, s*8 ,window_size=2.5)
    f[:,:, 17] = vigra.filters.gaussianGradientMagnitude(im, s*16,window_size=2.5) 
    f[:,:, 18] =  st128[:,:,0]
    f[:,:, 19] =  st128[:,:,1]
    f[:,:, 20] = vigra.filters.laplacianOfGaussian(im, s*16 ,window_size=2.5)
    return f
    
def local_shape_features_fine3_1(im,scaleStart):
    # Uses gaussian pyramid to calculate at multiple scales
    # Smoothing and scale parameters chosen to approximate Luca Fiashi paper
    #FIXME normalisation. Comparing image broken?? due to scaling, implement based on max datatype stored instead
    pyr_levels=5
    
    scale_mode='nearest'
    s = scaleStart
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,1+pyr_levels*4))
    f[:,:, 0]  = im
    
    pyr=skimage.transform.pyramid_gaussian(im,sigma=1.5, max_layer=pyr_levels, downscale=1.5)
    for layer in range(0,pyr_levels):
        a=pyr.next()
        scale=[float(im.shape[0])/float(a.shape[0]),float(im.shape[1])/float(a.shape[1])]
        lap=scipy.ndimage.filters.laplace(a)
        lap=scipy.ndimage.interpolation.zoom(lap, scale,order=1,mode=scale_mode)
        [m,n]=np.gradient(a)
        ggm=np.hypot(m,n)
        ggm=scipy.ndimage.interpolation.zoom(ggm, scale,order=1,mode=scale_mode)
    
        x,y,z=skfeat.structure_tensor(a,1)
        st =skfeat.structure_tensor_eigvals(x,y,z)
        st0=scipy.ndimage.interpolation.zoom(st[0], scale,order=1,mode=scale_mode)
        st1=scipy.ndimage.interpolation.zoom(st[1], scale,order=1,mode=scale_mode)


        f[:,:, layer*4+1]  = lap
        f[:,:, layer*4+2]  = ggm
        f[:,:, layer*4+3]  = st0
        f[:,:, layer*4+4]  = st1

    return f

def local_shape_features_fine3(im,scaleStart):
    
    # Uses gaussian pyramid to calculate at multiple scales
    # Smoothing and scale parameters chosen to approximate Luca Fiashi paper
    #FIXME normalisation. Broke comparing image, implement based on max datatype stored instead
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,26))
    f[:,:, 0]  = im
    #im=exposure.equalize_adapthist(im, kernel_size=5)
    pyr=skimage.transform.pyramid_gaussian(im,sigma=1.5, max_layer=5, downscale=2)
    a=im

    for layer in range(0,5):
        scale=[float(im.shape[0])/float(a.shape[0]),float(im.shape[1])/float(a.shape[1])]
        lap=scipy.ndimage.filters.laplace(a)
        lap=scipy.ndimage.interpolation.zoom(lap, scale,order=1)
        
        [m,n]=np.gradient(a)
        ggm=np.hypot(m,n)
        ggm=scipy.ndimage.interpolation.zoom(ggm, scale,order=1)

        x,y,z=skfeat.structure_tensor(a,1)
        st =skfeat.structure_tensor_eigvals(x,y,z)
        st0=scipy.ndimage.interpolation.zoom(st[0], scale,order=1)
        st1=scipy.ndimage.interpolation.zoom(st[1], scale,order=1)

        #ent=entropy(a,skimage.morphology.disk(3))
        ent=scipy.ndimage.interpolation.zoom(a, scale,order=1)
        
        #hess=vigra.filters.hessianOfGaussianEigenvalues(a.astype('float32'),2)
        #hess0=scipy.ndimage.interpolation.zoom(hess[:,:,0], scale,order=1,mode='nearest')
        #hess1=scipy.ndimage.interpolation.zoom(hess[:,:,1], scale,order=1,mode='nearest')
        f[:,:, layer*5+1]  = lap
        f[:,:, layer*5+2]  = ggm
        f[:,:, layer*5+3]  = st0
        f[:,:, layer*5+4]  = st1
        f[:,:, layer*5+5]  = ent
        a=pyr.next()
    return f
    
def local_shape_features_fine_imhist(im,scaleStart):
    
    # Uses gaussian pyramid to calculate at multiple scales
    # Smoothing and scale parameters chosen to approximate Luca Fiashi paper
    #FIXME normalisation. Broke comparing image, implement based on max datatype stored instead
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,26))
    im=exposure.equalize_hist(im)
    f[:,:, 0]  = im
    #im=exposure.equalize_adapthist(im, kernel_size=5)
    pyr=skimage.transform.pyramid_gaussian(im,sigma=1.5, max_layer=5, downscale=2)
    a=im

    for layer in range(0,5):
        scale=[float(im.shape[0])/float(a.shape[0]),float(im.shape[1])/float(a.shape[1])]
        lap=scipy.ndimage.filters.laplace(a)
        lap=scipy.ndimage.interpolation.zoom(lap, scale,order=1)
        
        [m,n]=np.gradient(a)
        ggm=np.hypot(m,n)
        ggm=scipy.ndimage.interpolation.zoom(ggm, scale,order=1)

        x,y,z=skfeat.structure_tensor(a,1)
        st =skfeat.structure_tensor_eigvals(x,y,z)
        st0=scipy.ndimage.interpolation.zoom(st[0], scale,order=1)
        st1=scipy.ndimage.interpolation.zoom(st[1], scale,order=1)

        #ent=entropy(a,skimage.morphology.disk(3))
        ent=scipy.ndimage.interpolation.zoom(a, scale,order=1)
        
        #hess=vigra.filters.hessianOfGaussianEigenvalues(a.astype('float32'),2)
        #hess0=scipy.ndimage.interpolation.zoom(hess[:,:,0], scale,order=1,mode='nearest')
        #hess1=scipy.ndimage.interpolation.zoom(hess[:,:,1], scale,order=1,mode='nearest')
        f[:,:, layer*5+1]  = lap
        f[:,:, layer*5+2]  = ggm
        f[:,:, layer*5+3]  = st0
        f[:,:, layer*5+4]  = st1
        f[:,:, layer*5+5]  = ent
        a=pyr.next()
    return f
 

def local_shape_features_dual(im,scaleStart):
    # include bar and edge detector filters for texture
    # based on MR8 filterbank
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,26+2*5))
    f[:,:, 0]  = im
    #im=exposure.equalize_adapthist(im, kernel_size=5)
    pyr=skimage.transform.pyramid_gaussian(im,sigma=1.5, max_layer=5, downscale=2)
    n_orientations=6
    edge, bar, rot=makeRFSfilters(radius=10, sigmas=[1], n_orientations=n_orientations)
    a=im
    for layer in range(0,5):
        scale=[float(im.shape[0])/float(a.shape[0]),float(im.shape[1])/float(a.shape[1])]
        lap=scipy.ndimage.filters.laplace(a)
        lap=scipy.ndimage.interpolation.zoom(lap, scale,order=1)
        
        [m,n]=np.gradient(a)
        ggm=np.hypot(m,n)
        ggm=scipy.ndimage.interpolation.zoom(ggm, scale,order=1)

        x,y,z=skfeat.structure_tensor(a,1)
        st =skfeat.structure_tensor_eigvals(x,y,z)
        st0=scipy.ndimage.interpolation.zoom(st[0], scale,order=1)
        st1=scipy.ndimage.interpolation.zoom(st[1], scale,order=1)

        #ent=entropy(a,skimage.morphology.disk(3))
        ent=scipy.ndimage.interpolation.zoom(a, scale,order=1)
        #hess=vigra.filters.hessianOfGaussianEigenvalues(a.astype('float32'),2)
        #hess0=scipy.ndimage.interpolation.zoom(hess[:,:,0], scale,order=1,mode='nearest')
        #hess1=scipy.ndimage.interpolation.zoom(hess[:,:,1], scale,order=1,mode='nearest')
        barmax=edgemax=np.zeros(a.shape)
        for orient in range(n_orientations):
            edgemax = np.maximum(edgemax,scipy.ndimage.convolve(a,edge[0,orient,:,:]))
            barmax = np.maximum(edgemax,scipy.ndimage.convolve(a,bar[0,orient,:,:]))
        edgemax=scipy.ndimage.interpolation.zoom(edgemax, scale,order=1)     
        barmax=scipy.ndimage.interpolation.zoom(barmax, scale,order=1)
        
        f[:,:, layer*5+1]  = lap
        f[:,:, layer*5+2]  = ggm
        f[:,:, layer*5+3]  = st0
        f[:,:, layer*5+4]  = st1
        f[:,:, layer*5+5]  = ent
        f[:,:, layer*5+6]  = edgemax
        f[:,:, layer*5+7]  = barmax
        a=pyr.next()
    return f

def local_shape_features_basic(im,scaleStart):
    #Exactly as in the Luca Fiaschi paper.
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,13))

    st08 = vigra.filters.structureTensorEigenvalues(im,s*1,s*2)
    st16 = vigra.filters.structureTensorEigenvalues(im,s*2,s*4)
    st32 = vigra.filters.structureTensorEigenvalues(im,s*4,s*8)

    f[:,:, 0]  = im
    f[:,:, 1]  = vigra.filters.gaussianGradientMagnitude(im, s,window_size=2.5)

    f[:,:, (2,3)]  = st08
    f[:,:, 4]  = vigra.filters.laplacianOfGaussian(im, s ,window_size=2.5)

    f[:,:, 5]  = vigra.filters.gaussianGradientMagnitude(im, s*2,window_size=2.5) 

    f[:,:, (6,7)]  =  st16

    f[:,:, 8]  = vigra.filters.laplacianOfGaussian(im, s*2 ,window_size=2.5)

    f[:,:, 9]  = vigra.filters.gaussianGradientMagnitude(im, s*4,window_size=2.5) 

    f[:,:, (10,11)] =  st32

    f[:,:, 12] = vigra.filters.laplacianOfGaussian(im, s*4 ,window_size=2.5)
    return f


def local_shape_features_basic3(im,scaleStart):
    #Exactly as in the Luca Fiaschi paper.
    #clunky multiscale version
    intp='bilinear' 
    s = scaleStart
    pyr=skimage.transform.pyramid_gaussian(im,sigma=1.5, max_layer=4, downscale=2)
    a=pyr.next()
    a1=scipy.ndimage.filters.laplace(a)
    a1=scipy.misc.imresize(a1,im.shape,interp=intp)
    [m,n]=np.gradient(a)
    a2=np.hypot(m,n)
    a2=scipy.misc.imresize(a2,im.shape,interp=intp)
    
    x,y,z=skfeat.structure_tensor(a,1)
    st =skfeat.structure_tensor_eigvals(x,y,z)
    a3=scipy.misc.imresize(st[0],im.shape,interp=intp)
    a4=scipy.misc.imresize(st[1],im.shape,interp=intp)
    
    b=pyr.next()
    b1=scipy.ndimage.filters.laplace(b)
    b1=scipy.misc.imresize(b1,im.shape,interp=intp)
    [m,n]=np.gradient(b)
    b2=np.hypot(m,n)
    b2=scipy.misc.imresize(b2,im.shape,interp=intp)
    
    x,y,z=skfeat.structure_tensor(b,1)
    st =skfeat.structure_tensor_eigvals(x,y,z)
    b3=scipy.misc.imresize(st[0],im.shape,interp=intp)
    b4=scipy.misc.imresize(st[1],im.shape,interp=intp)
    c=pyr.next()
    
    c1=scipy.ndimage.filters.laplace(c)
    c1=scipy.misc.imresize(c1,im.shape,interp=intp)
    [m,n]=np.gradient(c)
    c2=np.hypot(m,n)
    c2=scipy.misc.imresize(c2,im.shape,interp=intp)

    x,y,z=skfeat.structure_tensor(c,1)
    st =skfeat.structure_tensor_eigvals(x,y,z)
    c3=scipy.misc.imresize(st[0],im.shape,interp=intp)
    c4=scipy.misc.imresize(st[1],im.shape,interp=intp)
    '''
    d=pyr.next()
    
    d1=scipy.ndimage.filters.laplace(d)
    d1=scipy.misc.imresize(d1,im.shape)
    [m,n]=np.gradient(d)
    d2=np.hypot(m,n)
    d2=scipy.misc.imresize(d2,im.shape)
    
    st=vigra.filters.structureTensorEigenvalues(d.astype('float32'),1,2)
    d3=scipy.misc.imresize(st[:,:,0],im.shape)
    d4=scipy.misc.imresize(st[:,:,1],im.shape)
    '''
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,13))
    f[:,:, 0]  = im
    f[:,:, 1]  = a1
    f[:,:, 2]  = a2
    f[:,:, 3]  = a3
    f[:,:, 4]  = a4
    f[:,:, 5]  = b1
    f[:,:, 6]  = b2
    f[:,:, 7]  = b3
    f[:,:, 8]  = b4
    f[:,:, 9]  = c1
    f[:,:, 10]  = c2
    f[:,:, 11]  = c3
    f[:,:, 12] = c4
    return f
    
    