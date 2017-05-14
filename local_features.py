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

    elif RF_type== 'BR':
        method =linear_model.BayesianRidge(par_obj.n_iter, par_obj.tol, par_obj.alpha_1, par_obj.alpha_2, par_obj.lambda_1, par_obj.lambda_2)    
    elif RF_type== 'ABR':
        method = ensemble.AdaBoostRegressor(base_estimator=ensemble.ExtraTreesRegressor(10, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1), n_estimators=3, learning_rate=1.0, loss= 'square')
  
    return method
'''
        elif RF_type== 'ETC':
            method = ensemble.ExtraTreesClassifier(par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1,class_weight='balanced')
            self.fit = self.method.fit
            self.predict = self.method.predict_proba

        elif RF_type== 'DRETR':
            #self.method = ensemble.ExtraTreesRegressor(par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1)
            #self.method2 = PCA(n_components=0.9)
            
            self.method = Pipeline([
                ('reduce_dim', FastICA(n_components=70)),
                ('regress', ensemble.ExtraTreesRegressor(par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, bootstrap=True, n_jobs=-1))])
            self.fit = self.method.fit
            self.predict = self.method.predict
            #self.fit = self.__DRETRfit__
            #self.predict = self.__DRETRpredict__
'''
def get_feature_lengths(feature_type):
    #dictionary of feature sets
    #can add arbitrary feature sets by defining a name, length, and function that accepts two arguments
    feature_dict={'basic': [13,local_shape_features_basic],
                  'fine': [21,local_shape_features_fine],
                  'daisy': [101,local_shape_features_daisy],
                  'fine3': [26,local_shape_features_fine3],
                  'imhist': [26,local_shape_features_fine_imhist],
                  'texton': [21,local_shape_features_texton],
                  'comb': [51, local_shape_features_fine_comb],
                  'dual': [36, local_shape_features_dual],
                  'normal': [26, local_shape_features_fine3_normal],
                  'median': [26, local_shape_features_fine3_median],
                  'patch': [45,local_shape_features_patch],
                  'key': [14,local_shape_features_key],
                  'sharp': [8,local_shape_features_sharp],
                  'canny': [7,local_shape_features_canny]}
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
def local_shape_features_fine3_median(im,scaleStart):
    # Uses gaussian pyramid to calculate at multiple scales
    # Smoothing and scale parameters chosen to approximate Luca Fiashi paper
    #FIXME normalisation. Broke comparing image, implement based on max datatype stored instead
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,26))
    a=im
    #median based scaling (inefficient to do this to every image)
    #TODO subsample whole file, then normalise
    scaled=preprocessing.robust_scale(a,with_scaling=False)
    mscaled=scaled.max()
    scaled/=mscaled
    mscaled=mscaled*100
    a=scaled.reshape(a.shape)
    pyr=skimage.transform.pyramid_gaussian(a,sigma=1.5, max_layer=5, downscale=2,mode='constant',cval=0)
    f[:,:, 0]  = a*mscaled
    for layer in range(0,5):
        scale=[float(im.shape[0])/float(a.shape[0]),float(im.shape[1])/float(a.shape[1])]
        lap=scipy.ndimage.filters.laplace(a*mscaled,mode='nearest')
        lap=scipy.ndimage.interpolation.zoom(lap, scale,order=1)
        
        [m,n]=np.gradient(a*mscaled)
        ggm=np.hypot(m,n)
        ggm=scipy.ndimage.interpolation.zoom(ggm, scale,order=1)

        k=skfeat.structure_tensor(a,1,mode='nearest')
        st =skfeat.structure_tensor_eigvals(k[0],k[1],k[2])*mscaled
        st0=scipy.ndimage.interpolation.zoom(st[0], scale,order=1)
        st1=scipy.ndimage.interpolation.zoom(st[1], scale,order=1)

        #ent=entropy(a,skimage.morphology.disk(3))
        ent=scipy.ndimage.interpolation.zoom(a, scale,order=1)*mscaled
        
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
    
def local_shape_features_fine3_normal(im,scaleStart):
    # Uses gaussian pyramid to calculate at multiple scales
    # Smoothing and scale parameters chosen to approximate Luca Fiashi paper
    # FIXME normalisation. Broke comparing image, implement based on max datatype stored instead. DONE?
    # Vector based normalisation for each feature.
    # Worse than without normalisation
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,26))
    f[:,:, 0]  = im
    
    pyr=skimage.transform.pyramid_gaussian(im,sigma=1.5, max_layer=5, downscale=2)
    a=im
    for layer in range(0,5):
        scale=[float(im.shape[0])/float(a.shape[0]),float(im.shape[1])/float(a.shape[1])]

        lap=scipy.ndimage.filters.laplace(a).flat
        lap=preprocessing.scale(lap)
        lap=lap.reshape(a.shape)
        lap=scipy.ndimage.interpolation.zoom(lap, scale,order=1)
        
        [m,n]=np.gradient(a)
        ggm=np.hypot(m,n).flat
        ggm=preprocessing.scale(ggm)
        ggm=ggm.reshape(a.shape)
        ggm=scipy.ndimage.interpolation.zoom(ggm, scale,order=1)

        x,y,z=skfeat.structure_tensor(a,1)
        st =skfeat.structure_tensor_eigvals(x,y,z)
        st0=preprocessing.scale(st[0].flat)
        st0=st0.reshape(a.shape)
        st1=preprocessing.scale(st[1].flat)
        st1=st1.reshape(a.shape)
        
        st0=scipy.ndimage.interpolation.zoom(st0, scale,order=1)
        st1=scipy.ndimage.interpolation.zoom(st1, scale,order=1)

        #ent=entropy(a,skimage.morphology.disk(3))
        ent=a.flat
        ent=preprocessing.scale(ent)
        ent=ent.reshape(a.shape)
        ent=scipy.ndimage.interpolation.zoom(ent, scale,order=1)
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
    
def local_shape_features_daisy(im,scaleStart):
    # skiimage daisy
    # fast SIFT-like features
    # still slow

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,105))
    #im=exposure.equalize_adapthist(im, kernel_size=5)
    a=im
    daisyfeat= daisy(a,radius=scaleStart*3 ,histograms=8,step=1, orientations=4,normalization='daisy')
    scale=[float(im.shape[0])/float(daisyfeat.shape[0]),float(im.shape[1])/float(daisyfeat.shape[1])]  
    for i in range(daisyfeat.shape[2]):
        f[:,:, i+1]=scipy.ndimage.interpolation.zoom(daisyfeat[:,:,i],scale,order=0)
    return f
def local_shape_features_canny(im,scaleStart):
    # skiimage daisy
    # fast SIFT-like features
    # still slow
    

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,7))
    #im=exposure.equalize_adapthist(im, kernel_size=5)
    f[:,:, 1] = distance_transform_edt(1-skfeat.canny(im, sigma=1))
    f[:,:, 1]=f[:,:, 1]/(f[:,:, 1].max()+1)
    f[:,:, 2] = distance_transform_edt(1-skfeat.canny(im, sigma=3))
    f[:,:, 2]=f[:,:, 2]/(f[:,:, 2].max()+1)
    f[:,:, 3] = distance_transform_edt(1-skfeat.canny(im, sigma=5))
    f[:,:, 3]=f[:,:, 3]/(f[:,:, 3].max()+1)
    f[:,:, 4] = distance_transform_edt(1-skfeat.canny(im, sigma=9))
    f[:,:, 4]=f[:,:, 4]/(f[:,:, 4].max()+1)
    f[:,:, 5] = distance_transform_edt(1-skfeat.canny(im, sigma=17))
    f[:,:, 5]=f[:,:, 5]/(f[:,:, 5].max()+1)
    f[:,:, 6] = distance_transform_edt(1-skfeat.canny(im, sigma=33))
    f[:,:, 6]=f[:,:, 6]/(f[:,:, 6].max()+1)
    #f[:,:, 7] = distance_transform_edt(filters.canny(im, sigma=65))


    return f
def local_shape_features_key(im,scaleStart):
    # skiimage daisy
    # fast SIFT-like features
    # still slow
    

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,14))
    #im=exposure.equalize_adapthist(im, kernel_size=5)
    
    f[:,:, 1] = distance_transform_edt(1-skfeat.peak_local_max(skfeat.corner_harris(im,sigma=1), indices=False))
    #f[:,:, 1]=f[:,:, 1]/(f[:,:, 1].max()+1)
    f[:,:, 2] = distance_transform_edt(1-skfeat.peak_local_max(skfeat.corner_harris(im,sigma=3), indices=False))
    #f[:,:, 2]=f[:,:, 2]/(f[:,:, 2].max()+1)
    f[:,:, 3] = distance_transform_edt(1-skfeat.peak_local_max(skfeat.corner_harris(im,sigma=5), indices=False))
    #f[:,:, 3]=f[:,:, 3]/(f[:,:, 3].max()+1)
    f[:,:, 4] = distance_transform_edt(1-skfeat.peak_local_max(skfeat.corner_harris(im,sigma=9), indices=False))
    #f[:,:, 4]=f[:,:, 4]/(f[:,:, 4].max()+1)
    f[:,:, 5] = distance_transform_edt(1-skfeat.peak_local_max(skfeat.corner_harris(im,sigma=17), indices=False))
    #f[:,:, 5]=f[:,:, 5]/(f[:,:, 5].max()+1)
    f[:,:, 6] = distance_transform_edt(1-skfeat.peak_local_max(skfeat.corner_harris(im,sigma=33), indices=False))
    #f[:,:, 6]=f[:,:, 6]/(f[:,:, 6].max()+1)
    f[:,:,7::]=distance_transform_edt(1-skfeat.peak_local_max(skfeat.corner_harris(im,sigma=65), indices=False))
    #f[:,:, 7] = distance_transform_edt(filters.canny(im, sigma=65))
    return f
    
def local_shape_features_sharp(im,scaleStart):
    # skiimage daisy
    # fast SIFT-like features
    # still slow
    

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,8))
    #im=exposure.equalize_adapthist(im, kernel_size=5)
    
    f[:,:, 1] = im-ndimage.gaussian_filter(im,1)
    #f[:,:, 1]=f[:,:, 1]/(f[:,:, 1].max()+1)
    f[:,:, 2] = im-ndimage.gaussian_filter(im,3)
    #f[:,:, 2]=f[:,:, 2]/(f[:,:, 2].max()+1)
    f[:,:, 3] = im-ndimage.gaussian_filter(im,5)
    #f[:,:, 3]=f[:,:, 3]/(f[:,:, 3].max()+1)
    f[:,:, 4] = im-ndimage.gaussian_filter(im,9)
    #f[:,:, 4]=f[:,:, 4]/(f[:,:, 4].max()+1)
    f[:,:, 5] = im-ndimage.gaussian_filter(im,17)
    #f[:,:, 5]=f[:,:, 5]/(f[:,:, 5].max()+1)
    f[:,:, 6] = im-ndimage.gaussian_filter(im,33)
    #f[:,:, 6]=f[:,:, 6]/(f[:,:, 6].max()+1)
    f[:,:,7]= im-ndimage.gaussian_filter(im,65)
    #f[:,:, 7] = distance_transform_edt(filters.canny(im, sigma=65))
    return f
    
def local_shape_features_fine3_cust(im,scaleStart):
    # Added custom feature to test for improvement in NB detection
    # results minimal
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,26))
    f[:,:, 0]  = im
    #im=exposure.equalize_adapthist(im, kernel_size=5)
    pyr=skimage.transform.pyramid_gaussian(im,sigma=1.5, max_layer=5, downscale=2)
    a=im
    cust_feat=np.asarray([[77,78,78,74,76,81,82,82,81,84,79,81,85,83,76,76,86,88,86,86,93,95,91,88,80,77],
    [80,81,83,82,81,79,77,79,81,82,74,74,86,87,80,85,91,90,91,92,96,96,94,88,81,78],
    [88,88,90,87,83,79,79,82,86,85,79,78,86,90,87,93,95,95,93,94,94,93,95,90,83,78],
    [91,94,95,91,86,85,86,87,90,87,80,81,89,95,90,90,97,100,93,89,89,92,91,87,82,74],
    [95,96,93,92,88,89,90,89,88,80,75,73,83,90,83,79,87,94,91,89,88,88,86,83,81,76],
    [93,96,94,89,88,92,92,85,80,70,64,61,64,71,66,63,75,89,88,89,92,89,85,79,79,81],
    [86,92,94,89,89,89,86,77,67,58,56,53,50,48,48,51,61,77,82,86,92,90,86,79,77,82],
    [86,91,93,93,91,88,82,70,57,49,47,45,42,39,39,41,47,59,70,77,85,89,87,82,79,82],
    [88,90,95,100,94,89,77,59,48,41,39,38,38,35,35,35,40,48,57,67,80,88,90,86,81,81],
    [86,91,95,97,87,75,61,47,40,36,37,38,39,38,37,34,36,41,49,58,70,80,87,85,82,84],
    [76,85,93,90,79,63,51,41,35,34,41,44,42,40,42,41,37,39,47,56,64,75,80,79,74,79],
    [76,80,87,90,83,66,48,39,35,37,42,45,45,43,45,44,38,38,45,53,61,73,81,78,74,81],
    [83,87,90,95,90,71,48,39,35,38,40,43,47,47,45,42,39,38,42,50,64,83,89,86,86,85],
    [85,86,86,89,83,64,50,42,38,39,42,45,47,47,43,40,38,35,39,48,71,90,95,90,87,83],
    [81,74,78,81,73,61,53,45,38,38,44,45,43,45,45,42,37,35,39,48,66,83,90,87,80,76],
    [79,74,79,80,75,64,56,47,39,37,41,42,40,42,44,41,34,35,41,51,63,79,90,93,85,76],
    [84,82,85,87,80,70,58,49,41,36,34,37,38,39,38,37,36,40,47,61,75,87,97,95,91,86],
    [81,81,86,90,88,80,67,57,48,40,35,35,35,38,38,39,41,48,59,77,89,94,100,95,90,88],
    [82,79,82,87,89,85,77,70,59,47,41,39,39,42,45,47,49,57,70,82,88,91,93,93,91,86],
    [82,77,79,86,90,92,86,82,77,61,51,48,48,50,53,56,58,67,77,86,89,89,89,94,92,86],
    [81,79,79,85,89,92,89,88,89,75,63,66,71,64,61,64,70,80,85,92,92,88,89,94,96,93],
    [76,81,83,86,88,88,89,91,94,87,79,83,90,83,73,75,80,88,89,90,89,88,92,93,96,95],
    [74,82,87,91,92,89,89,93,100,97,90,90,95,89,81,80,87,90,87,86,85,86,91,95,94,91],
    [78,83,90,95,93,94,94,93,95,95,93,87,90,86,78,79,85,86,82,79,79,83,87,90,88,88],
    [78,81,88,94,96,96,92,91,90,91,85,80,87,86,74,74,82,81,79,77,79,81,82,83,81,80],
    [77,80,88,91,95,93,86,86,88,86,76,76,83,85,81,79,84,81,82,82,81,76,74,78,78,77]])
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
        ent=signal.convolve2d(a,cust_feat,mode='same')
        ent=scipy.ndimage.interpolation.zoom(ent, scale,order=1)
        
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
    
def local_shape_features_fine_comb(im,scaleStart):
    # Combinatorial features
    # based on idea that RF struggles with complex decision boundaries
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,26+5*5))
    f[:,:, 0]  = im
    
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
        #ent=scipy.ndimage.filters.percentile_filter(a, 50, size=2)
        #ent=scipy.ndimage.interpolation.zoom(ent, scale,order=1)
        poly=preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        ent=poly.fit_transform(np.column_stack((lap.flat,ggm.flat,st0.flat,st1.flat)))
        ent=ent.reshape((imSizeC,imSizeR,10))
        #hess=vigra.filters.hessianOfGaussianEigenvalues(a.astype('float32'),2)
        #hess0=scipy.ndimage.interpolation.zoom(hess[:,:,0], scale,order=1,mode='nearest')
        #hess1=scipy.ndimage.interpolation.zoom(hess[:,:,1], scale,order=1,mode='nearest')
        f[:,:, layer*5+1:layer*5+11]=ent
        '''f[:,:, layer*5+1]  = lap
        f[:,:, layer*5+2]  = ggm
        f[:,:, layer*5+3]  = st0
        f[:,:, layer*5+4]  = st1
        f[:,:, layer*5+5]  = ent'''

        
        a=pyr.next()
    return f
def local_shape_features_finez(im,scaleStart):
    # Allows passing of a volume
    # Little improvement, for obvious reasons wrt lack of 3D information
    #FIXME normalisation. Broke comparing image, implement based on max datatype stored instead
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,26))
    f[:,:, 0]  = im
    
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
        #ent=scipy.ndimage.interpolation.zoom(ent, scale,order=1)
        ent=scipy.ndimage.interpolation.zoom(a, scale,order=1,mode='nearest')
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
def local_shape_features_texton(im,scaleStart):
    #FIXME normalisation. Broke comparing image, implement based on max datatype stored instead
    # based on MR8 filterbank
    pyr_levels=5
    
    s = scaleStart
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,1+pyr_levels*4))
    f[:,:, 0]  = im
    
    pyr=skimage.transform.pyramid_gaussian(im,sigma=1.5, max_layer=5, downscale=2)
    #generate filters
    #could precalculate for speed
    n_orientations=6
    edge, bar, rot=makeRFSfilters(radius=10, sigmas=[1], n_orientations=n_orientations)
    #we will use only one scale, and then apply repeatedly to different parts of the pyramid
    #at each level of the pyramid we end up with two filter responses, plus a lap and a gauss
    #this differs from the original paper, but seems a reasonable way to test if it works
    for layer in range(0,pyr_levels):
        
        a=pyr.next()
        scale=[float(im.shape[0])/float(a.shape[0]),float(im.shape[1])/float(a.shape[1])]
        
        gauss=scipy.ndimage.interpolation.zoom(a, scale,order=1)        
        
        lap=scipy.ndimage.filters.laplace(a)
        lap=scipy.ndimage.interpolation.zoom(lap, scale,order=1)
        
        barmax=edgemax=np.zeros(a.shape)
        for orient in range(n_orientations):
            edgemax = np.maximum(edgemax,scipy.ndimage.convolve(a,edge[0,orient,:,:]))
            barmax = np.maximum(edgemax,scipy.ndimage.convolve(a,bar[0,orient,:,:]))
        edgemax=scipy.ndimage.interpolation.zoom(edgemax, scale,order=1)     
        barmax=scipy.ndimage.interpolation.zoom(barmax, scale,order=1)
        
        f[:,:, layer*4+1]  = gauss
        f[:,:, layer*4+2]  = lap
        f[:,:, layer*4+3]  = edgemax
        f[:,:, layer*4+4]  = barmax

    return f
    
def local_shape_features_patch(im,scaleStart):
    # RAW multiscale patches
    # Learns slowly and reaches limit quickly
    pyr_levels=5
    #check pyramid for starting point and if loops match up correctly
    s = scaleStart
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,pyr_levels*9))
    #f[:,:, 0]  = im
    pyr=skimage.transform.pyramid_gaussian(im,sigma=1.5, max_layer=pyr_levels, downscale=2)
    a=im
    for layer in range(0,pyr_levels):

        scale=[float(im.shape[0])/float(a.shape[0]),float(im.shape[1])/float(a.shape[1])]
        
        gauss=scipy.ndimage.interpolation.zoom(a, scale,order=1)        
        
        lap=scipy.ndimage.filters.laplace(a)
        lap=scipy.ndimage.interpolation.zoom(lap, scale,order=1)
        [cx,cy]=np.where(np.ones((3,3)))
        for it in range(len(cx)):
            x=cx[it]
            y=cy[it]
            shifted=scipy.ndimage.interpolation.shift(a,(x-1,y-1),order=0,mode='nearest') 
            shifted=scipy.ndimage.interpolation.zoom(shifted, scale,order=1)
            f[:,:, layer*9+it] = shifted
           
        #f[:,:, layer*4+1]  = gauss
        #f[:,:, layer*4+2]  = lap
        #f[:,:, layer*4+3]  = 
        #f[:,:, layer*4+4]  = 
        a=pyr.next()
    return f
    
def makeRFSfilters(radius=24, sigmas=[1, 2, 4], n_orientations=6):
    """ Generates filters for RFS filterbank.
    Parameters
    ----------
    radius : int, default 28
        radius of all filters. Size will be 2 * radius + 1
    sigmas : list of floats, default [1, 2, 4] as in paper
        define scales on which the filters will be computed
    n_orientations : int
        number of fractions the half-angle will be divided in
    Returns
    -------
    edge : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
        Contains edge filters on different scales and orientations
    bar : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
        Contains bar filters on different scales and orientations
    rot : ndarray (2, 2*radius+1, 2*radius+1)
        contains two rotation invariant filters, Gaussian and Laplacian of
        Gaussian
    """
    def make_gaussian_filter(x, sigma, order=0):
        if order > 2:
            raise ValueError("Only orders up to 2 are supported")
        # compute unnormalized Gaussian response
        response = np.exp(-x ** 2 / (2. * sigma ** 2))
        if order == 1:
            response = -response * x
        elif order == 2:
            response = response * (x ** 2 - sigma ** 2)
        # normalize
        response /= np.abs(response).sum()
        return response

    def makefilter(scale, phasey, pts, sup):
        gx = make_gaussian_filter(pts[0, :], sigma=3 * scale)
        gy = make_gaussian_filter(pts[1, :], sigma=scale, order=phasey)
        f = (gx * gy).reshape(sup, sup)
        # normalize
        f /= np.abs(f).sum()
        return f

    support = 2 * radius + 1
    x, y = np.mgrid[-radius:radius + 1, radius:-radius - 1:-1]
    orgpts = np.vstack([x.ravel(), y.ravel()])

    rot, edge, bar = [], [], []
    for sigma in sigmas:
        for orient in xrange(n_orientations):
            # Not 2pi as filters have symmetry
            angle = np.pi * orient / n_orientations
            c, s = np.cos(angle), np.sin(angle)
            rotpts = np.dot(np.array([[c, -s], [s, c]]), orgpts)
            edge.append(makefilter(sigma, 1, rotpts, support))
            bar.append(makefilter(sigma, 2, rotpts, support))
    length = np.sqrt(x ** 2 + y ** 2)
    rot.append(make_gaussian_filter(length, sigma=10))
    rot.append(make_gaussian_filter(length, sigma=10, order=2))

    # reshape rot and edge
    edge = np.asarray(edge)
    edge = edge.reshape(len(sigmas), n_orientations, support, support)
    bar = np.asarray(bar).reshape(edge.shape)
    rot = np.asarray(rot)[:, np.newaxis, :, :]
    return edge, bar, rot
    
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
    
    