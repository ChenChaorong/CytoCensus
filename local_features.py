# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:25:35 2016

@author: martin
Feature calculation methods (moved from v2_functions for clarity)
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
import skimage
import time
import numpy as np
import threading
#from v2_functions import get_tiff_slice
from scipy.ndimage.interpolation import shift


def feature_create_threadable_auto(par_obj,imno,tpt,zslice):
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

    
def get_feature_lengths(feature_type):
    #dictionary of feature sets
    #can add arbitrary feature sets by defining a name, length, and function that accepts two arguments
    feature_dict={'basic': [13,local_shape_features_basic2],
                  'fine': [21,local_shape_features_fine],
                  'fine3': [26,local_shape_features_fine3],
                  'texton': [21,local_shape_features_texton],
                  'patch': [45,local_shape_features_patch] }
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
    feat = np.zeros(((int(par_obj.crop_y2)-int(par_obj.crop_y1)),(int(par_obj.crop_x2)-int(par_obj.crop_x1)),feat_length*par_obj.ch_active.__len__()))

    for b in range(0,par_obj.ch_active.__len__()):
            imG = imRGB[:,:,b].astype(np.float32)

            feat[:,:,(b*feat_length):((b+1)*feat_length)] = feat_func(imG,par_obj.feature_scale)  
    #TODO?? Investigate if this is really necessary
    if par_obj.numCH==0:
        imG = imRGB[:,:,0].astype(np.float32)

        feat = feat_func(imG,par_obj.feature_scale)  

    return feat
    
def feature_create_z(par_obj,imRGB):
    import v2
    #not currently intended to be threaded
    time1 = time.time()
    feat_length
    feat = np.zeros(((int(par_obj.crop_y2)-int(par_obj.crop_y1)),(int(par_obj.crop_x2)-int(par_obj.crop_x1)),feat_length*par_obj.ch_active.__len__()))

    for b in range(0,par_obj.ch_active.__len__()):
            imG = imRGB[:,:,b].astype(np.float32)

            feat[:,:,(b*feat_length):((b+1)*feat_length)] = local_shape_features_fine3(imG,par_obj.feature_scale)  
            
    if par_obj.numCH==0:
        imG = imRGB[:,:,0].astype(np.float32)

        feat = feat_func(imG,par_obj.feature_scale)  

    return feat

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
    #Exactly as in the Luca Fiaschi paper.
    #FIXME normalisation. Broke comparing image, implement based on max datatype stored instead
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
    #Exactly as in the Luca Fiaschi paper.
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
def local_shape_features_finez(im,scaleStart):
    #Exactly as in the Luca Fiaschi paper.
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
def local_shape_features_texton(im,scaleStart):
    #FIXME normalisation. Broke comparing image, implement based on max datatype stored instead
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
    #FIXME normalisation. Broke comparing image, implement based on max datatype stored instead
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

def local_shape_features_basic2(im,scaleStart):
    #Exactly as in the Luca Fiaschi paper.
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,13))

    #st08 = vigra.filters.structureTensorEigenvalues(im,s*1,s*2)
    x,y,z=skfeat.structure_tensor(im,s*1)
    st08 =skfeat.structure_tensor_eigvals(x,y,z)
    #st16 = vigra.filters.structureTensorEigenvalues(im,s*2,s*4)
    x,y,z=skfeat.structure_tensor(im,s*2)
    st16 =skfeat.structure_tensor_eigvals(x,y,z)
    #st32 = vigra.filters.structureTensorEigenvalues(im,s*4,s*8)
    x,y,z=skfeat.structure_tensor(im,s*4)
    st32 = skfeat.structure_tensor_eigvals(x,y,z)

    f[:,:, 0]  = im
    f[:,:, 1]  = ndimage.gaussian_gradient_magnitude(im,s,truncate=2.5)

    f[:,:, 2]  = st08[0]
    f[:,:, 3]  = st08[1]
    f[:,:, 4]  = ndimage.gaussian_laplace(im,s,truncate=2.5)

    f[:,:, 5]  =ndimage.gaussian_gradient_magnitude(im,s*2,truncate=2.5)

    f[:,:, 6]  = st08[0]
    f[:,:, 7]  = st08[1]

    f[:,:, 8]  = ndimage.gaussian_laplace(im,s*2,truncate=2.5)

    f[:,:, 9]  = ndimage.gaussian_gradient_magnitude(im,s*4,truncate=2.5)

    f[:,:, 10]  = st08[0]
    f[:,:, 11]  = st08[1]

    f[:,:, 12] = ndimage.gaussian_laplace(im,s*4,truncate=2.5)
    return f
def local_shape_features_basic3(im,scaleStart):
    #Exactly as in the Luca Fiaschi paper.
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
    
    