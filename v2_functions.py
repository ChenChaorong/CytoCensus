from PyQt4 import QtGui, QtCore, Qt
import PIL.Image
import numpy as np
import os
import vigra
import pylab
import csv
import time

from sklearn.ensemble import *
import sklearn
from scipy.ndimage import filters
from scipy.ndimage import distance_transform_edt
import cPickle as pickle

from oiffile import OifFile
from tifffile import TiffFile, imsave #Install with pip install tifffile.

import itertools as itt
import struct
import copy
import functools
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.path import Path

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
import modest_image
import pdb
import numpy as np
from local_features import *

"""QuantiFly3d Software v0.1

    Copyright (C) 2016  Dominic Waithe Martin Hailstone

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

def peak_local_max(image,min_distance=10, threshold_abs=0, threshold_rel=0.1,
                   exclude_border=False, indices=True, num_peaks=np.inf,
                   footprint=None, labels=None):
    """
    Find peaks in an image, and return them as coordinates or a boolean array.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    NOTE: If peaks are flat (i.e. multiple adjacent pixels have identical
    intensities), the coordinates of all such pixels are returned.

    Parameters
    ----------
    image : ndarray of floats
        Input image.
    min_distance : int
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`). If `exclude_border` is True, this value also excludes
        a border `min_distance` from the image boundary.
        To find the maximum number of peaks, use `min_distance=1`.
    threshold_abs : float
        Minimum intensity of peaks.
    threshold_rel : float
        Minimum intensity of peaks calculated as `max(image) * threshold_rel`.
    exclude_border : bool
        If True, `min_distance` excludes peaks from the border of the image as
        well as from each other.
    indices : bool
        If True, the output will be an array representing peak coordinates.
        If False, the output will be a boolean array shaped as `image.shape`
        with peaks present at True elements.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.  Overrides
        `min_distance`, except for border exclusion if `exclude_border=True`.
    labels : ndarray of ints, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in a image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison between
    dilated and original image, peak_local_max function returns the
    coordinates of peaks where dilated image = original.

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1.5,  0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 2],
           [3, 4]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> peak_local_max(img2, exclude_border=False)
    array([[10, 10, 10]])

    """
    out = np.zeros_like(image, dtype=np.bool)


    if np.all(image == image.flat[0]):
        if indices is True:
            return []
        else:
            return out

    image = image.copy()
    # Non maximum filter
    if footprint is not None:
        image_max = filters.maximum_filter(image, footprint=footprint,mode='constant')
    else:
        size = np.array(min_distance)*2.3548
        image_max = filters.maximum_filter(image, size=size, mode='constant')
    mask = (image == image_max)
    image *= mask

    if exclude_border:
        # zero out the image borders
        for i in range(image.ndim):
            image = image.swapaxes(0, i)
            
            min_d = np.floor(min_distance[i])
            
            image[:min_d] = 0
            image[-min_d:] = 0
            image = image.swapaxes(0, i)

    # find top peak candidates above a threshold
    peak_threshold = max(np.max(image.ravel()) * threshold_rel, threshold_abs)

    # get coordinates of peaks
    coordinates = np.argwhere(image > peak_threshold)

    if coordinates.shape[0] > num_peaks:
        intensities = image.flat[np.ravel_multi_index(coordinates.transpose(),image.shape)]
        idx_maxsort = np.argsort(intensities)[::-1]
        coordinates = coordinates[idx_maxsort][:num_peaks]

    if indices is True:
        return coordinates
    else:
        nd_indices = tuple(coordinates.T)
        out[nd_indices] = True

       
        return out
        #return out
"""rankorder.py - convert an image of any type to an image of ints whose
pixels have an identical rank order compared to the original image

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentstky
"""


def count_maxima(par_obj,time_pt,fileno):
    #count maxima won't work properly if have selected a random set of Z
    if par_obj.min_distance[2]==0 or len(par_obj.frames_2_load)==1:
        count_maxima_2d(par_obj,time_pt,fileno)
        return
    predMtx = np.zeros((par_obj.height,par_obj.width,par_obj.frames_2_load[-1]+1))
    for i in par_obj.frames_2_load:
        predMtx[:,:,i]= par_obj.data_store['pred_arr'][fileno][time_pt][i]

    gau_stk = filters.gaussian_filter(predMtx,par_obj.min_distance)
    y,x,z = np.gradient(gau_stk,1)
    xy,xx,xz = np.gradient(x)
    yy,yx,yz = np.gradient(y)
    zy,zx,zz = np.gradient(z)
    det = -1*((((yy*zz)-(yz*yz))*xx)-(((xy*zz)-(yz*xz))*xy)+(((xy*yz)-(yy*xz))*xz))
    #detl = -1*np.min(det)+det
    #det[np.where(det<0)]=0
    # if not already set, create. This is then used for the entire image and all subsequent training. A little hacky, but otherwise the normalisation screws everything up
    if not par_obj.max_det:
        par_obj.max_det=np.max(det)
        
    detn = det/par_obj.max_det
    par_obj.data_store['maxi_arr'][fileno][time_pt] = {}
    for i in par_obj.frames_2_load:
        #par_obj.data_store[time_pt]['maxi_arr'][i] = np.sqrt(detn[:,:,i]*par_obj.data_store[time_pt]['pred_arr'][i])
        par_obj.data_store['maxi_arr'][fileno][time_pt][i] = detn[:,:,i]
    

    pts = peak_local_max(detn, min_distance=par_obj.min_distance,threshold_abs=par_obj.abs_thr,threshold_rel=par_obj.rel_thr)

    #par_obj.pts = v2._prune_blobs(par_obj.pts, min_distance=[int(self.count_txt_1.text()),int(self.count_txt_2.text()),int(self.count_txt_3.text())])

    par_obj.show_pts = 1

    #Filter those which are not inside the region.
    if par_obj.data_store['roi_stkint_x'][fileno][time_pt].__len__() >0:
        pts2keep = []
        
        for i in par_obj.data_store['roi_stkint_x'][fileno][time_pt]:
             for pt2d in pts:


                if pt2d[2] == i:
                    #Find the region of interest.
                    ppt_x = par_obj.data_store['roi_stkint_x'][fileno][time_pt][i]
                    ppt_y = par_obj.data_store['roi_stkint_y'][fileno][time_pt][i]
                    #Reformat to make the path object.
                    pot = []
                    for b in range(0,ppt_x.__len__()):
                        pot.append([ppt_x[b],ppt_y[b]])
                    p = Path(pot)
                    if p.contains_point([pt2d[1],pt2d[0]]) == True:
                            pts2keep.append(pt2d)
        pts = pts2keep
    par_obj.data_store['pts'][fileno][time_pt] = pts
def count_maxima_2d(par_obj,time_pt,fileno):
    #count maxima won't work properly if have selected a random set of Z
    par_obj.min_distance[2]=0
    det=gau_stk = predMtx = np.zeros((par_obj.height,par_obj.width,par_obj.frames_2_load[-1]+1))
    for i in par_obj.frames_2_load:
        predMtx[:,:,i]= par_obj.data_store['pred_arr'][fileno][time_pt][i]
        gau_stk[:,:,i] = filters.gaussian_filter(predMtx[:,:,i],(par_obj.min_distance[0],par_obj.min_distance[1]))
        y,x = np.gradient(gau_stk[:,:,i],1)
        xy,xx = np.gradient(x)
        yy,yx = np.gradient(y)
        det[:,:,i] = xx*yy-xy*yx
    #detl = -1*np.min(det)+det
    #det[np.where(det<0)]=0
    # if not already set, create. This is then used for the entire image and all subsequent training. A little hacky, but otherwise the normalisation screws everything up
    if not par_obj.max_det:
        par_obj.max_det=np.max(det)
        
    detn = det/par_obj.max_det
    par_obj.data_store['maxi_arr'][fileno][time_pt] = {}
    for i in par_obj.frames_2_load:
        #par_obj.data_store[time_pt]['maxi_arr'][i] = np.sqrt(detn[:,:,i]*par_obj.data_store[time_pt]['pred_arr'][i])
        par_obj.data_store['maxi_arr'][fileno][time_pt][i] = detn[:,:,i]
    

    pts = peak_local_max(detn, min_distance=par_obj.min_distance,threshold_abs=par_obj.abs_thr,threshold_rel=par_obj.rel_thr)

    #par_obj.pts = v2._prune_blobs(par_obj.pts, min_distance=[int(self.count_txt_1.text()),int(self.count_txt_2.text()),int(self.count_txt_3.text())])

    par_obj.show_pts = 1

    #Filter those which are not inside the region.
    if par_obj.data_store['roi_stkint_x'][fileno][time_pt].__len__() >0:
        pts2keep = []
        
        for i in par_obj.data_store['roi_stkint_x'][fileno][time_pt]:
             for pt2d in pts:


                if pt2d[2] == i:
                    #Find the region of interest.
                    ppt_x = par_obj.data_store['roi_stkint_x'][fileno][time_pt][i]
                    ppt_y = par_obj.data_store['roi_stkint_y'][fileno][time_pt][i]
                    #Reformat to make the path object.
                    pot = []
                    for b in range(0,ppt_x.__len__()):
                        pot.append([ppt_x[b],ppt_y[b]])
                    p = Path(pot)
                    if p.contains_point([pt2d[1],pt2d[0]]) == True:
                            pts2keep.append(pt2d)
        pts = pts2keep
    par_obj.data_store['pts'][fileno][time_pt] = pts
    
def rank_order(image):
    """Return an image of the same shape where each pixel is the
    index of the pixel value in the ascending order of the unique
    values of `image`, aka the rank-order value.

    Parameters
    ----------
    image: ndarray

    Returns
    -------
    labels: ndarray of type np.uint32, of shape image.shape
        New array where each pixel has the rank-order value of the
        corresponding pixel in `image`. Pixel values are between 0 and
        n - 1, where n is the number of distinct unique values in
        `image`.

    original_values: 1-D ndarray
        Unique original values of `image`

    Examples
    --------
    >>> a = np.array([[1, 4, 5], [4, 4, 1], [5, 1, 1]])
    >>> a
    array([[1, 4, 5],
           [4, 4, 1],
           [5, 1, 1]])
    >>> rank_order(a)
    (array([[0, 1, 2],
           [1, 1, 0],
           [2, 0, 0]], dtype=uint32), array([1, 4, 5]))
    >>> b = np.array([-1., 2.5, 3.1, 2.5])
    >>> rank_order(b)
    (array([0, 1, 2, 1], dtype=uint32), array([-1. ,  2.5,  3.1]))
    """
    flat_image = image.ravel()
    sort_order = flat_image.argsort().astype(np.uint32)
    flat_image = flat_image[sort_order]
    sort_rank = np.zeros_like(sort_order)
    is_different = flat_image[:-1] != flat_image[1:]
    np.cumsum(is_different, out=sort_rank[1:])
    original_values = np.zeros((sort_rank[-1] + 1,), image.dtype)
    original_values[0] = flat_image[0]
    original_values[1:] = flat_image[1:][is_different]
    int_image = np.zeros_like(sort_order)
    int_image[sort_order] = sort_rank
    return (int_image.reshape(image.shape), original_values)
    
def _blob_overlap(blob1, blob2,min_distance):
    """Finds the overlapping area fraction between two blobs.
    Returns a float representing fraction of overlapped area.
    """
    
    d1 = abs(blob1[0] - blob2[0]) > min_distance[0]
    d2 = abs(blob1[1] - blob2[1]) > min_distance[1]
    d3 = abs(blob1[2] - blob2[2]) > min_distance[2]

    if d1 == False or d2 == False or d3 == False:
        #overlap detected
        
        return True
    
    return False

def _prune_blobs(blobs_array, min_distance):
    """Eliminated blobs with area overlap.

    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.

    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """

    # iterating again might eliminate more blobs, but one iteration suffices
    # for most cases
    for blob1, blob2 in itt.combinations(blobs_array, 2):

        if _blob_overlap(blob1, blob2,min_distance) == True:
            blob2[2] = -1

            #if blob1[2] > blob2[2]:
            #    blob2[2] = -1
            #else:
            #    blob1[2] = -1

    # return blobs_array[blobs_array[:, 2] > 0]
    return np.array([b for b in blobs_array if b[2] > 0])

    
    


def save_roi_fn(par_obj):
    """Saves ROI"""

    #If there is no width or height either no roi is selected or it is too thin.
    if par_obj.rect_w != 0 and par_obj.rect_h != 0:
        #If the ROI was in the negative direction.
        if par_obj.rect_w < 0:
            s_ori_x = par_obj.ori_x_2
        else:
            s_ori_x = par_obj.ori_x
        if par_obj.rect_h < 0:
            s_ori_y = par_obj.ori_y_2
        else:
            s_ori_y = par_obj.ori_y

        #Finds the current frame and file.
        par_obj.rects = (par_obj.curr_z, int(s_ori_x), int(s_ori_y), int(abs(par_obj.rect_w)), int(abs(par_obj.rect_h)),par_obj.time_pt,par_obj.curr_file)
        return True
    
    return False
    
def stratified_sample(par_obj,binlength,samples_indices,imhist,samples_at_tiers,mImRegion,denseRegion):
    indices=np.zeros(samples_indices[-1],'uint32') #preallocate array rather than extend, size corrects for rounding errors
    #Randomly sample from input ROI or im a certain number (par_obj.limit_size) patches. With replacement.
    for it in range(binlength):
        if samples_at_tiers[it]>0:
            bin1=imhist[1][it]
            bin2=imhist[1][it+1]

            
            stratified_indices=np.nonzero(((denseRegion>=bin1) & (denseRegion<bin2)).flat)
            if stratified_indices[0].__len__() != 0:
                stratified_sampled_indices =  np.random.choice(stratified_indices[0], size= samples_at_tiers[it], replace=True, p=None)
            else:
                stratified_sampled_indices=[]
            indices[range(samples_indices[it],samples_indices[it+1])] = stratified_sampled_indices
    return indices

def update_training_samples_fn_new_only(par_obj,int_obj,rects,arr='feat_arr'):
    """Collects the pixels or patches which will be used for training and 
    trains the forest."""
    #Makes sure everything is refreshed for the training, encase any regions
    #were changed. May have to be rethinked for speed later on.
    region_size = 0
    '''
    for b in range(0,par_obj.saved_ROI.__len__()):
        rects = par_obj.saved_ROI[b]
        region_size += rects[4]*rects[3]       
    '''
    calc_ratio = par_obj.limit_ratio_size
    #print 'calcratio',calc_ratio
    STRATIFY=False
    dot_im=np.pad(np.ones((1,1)),(int(par_obj.sigma_data)*6,int(par_obj.sigma_data)*4),mode='constant')
    dot_im=filters.gaussian_filter(dot_im,float(par_obj.sigma_data),mode='constant',cval=0)
    dot_im/=dot_im.max()
    binlength=10
    imhist=np.histogram(dot_im,bins=binlength,range=(0,1),density=True)
    imhist[1][binlength]=5 # adjust top bin to make sure we include everthing if we have overlapping gaussians-try to avoid though-if very common will distort bins
    samples_at_tiers=(imhist[0]/binlength*par_obj.limit_size).astype('int')
    #print samples_at_tiers
    samples_indices = [0]+(np.cumsum(samples_at_tiers)).tolist() #to allow preallocation of array
    
    
    #for b in range(0,par_obj.saved_ROI.__len__()):
        #TODO check this works for edge cases- with very sparse sampling, and with v small bin sizes
        #Iterates through saved ROI.
        #rects = par_obj.saved_ROI[b]
    
    zslice = rects[0]
    tpt =rects[5]
    imno =rects[6]
    if(par_obj.p_size == 1):
        #if rects[5] == tpt and rects[0] == zslice and rects[6] == imno:
            print rects
            #Finds and extracts the features and output density for the specific regions.
            mImRegion = par_obj.data_store[arr][imno][tpt][zslice][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3],:]
            denseRegion = par_obj.data_store['dense_arr'][imno][tpt][zslice][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3]]
            #Find the linear form of the selected feature representation
            mimg_lin = np.reshape(mImRegion, (mImRegion.shape[0]*mImRegion.shape[1],mImRegion.shape[2]))
            #Find the linear form of the complementatory output region.
            dense_lin = np.reshape(denseRegion, (denseRegion.shape[0]*denseRegion.shape[1]))
            #Sample the input pixels sparsely or densely.
            if(par_obj.limit_sample == True):
                if(par_obj.limit_ratio == True):
                    par_obj.limit_size = round(mImRegion.shape[0]*mImRegion.shape[1]/calc_ratio,0)
                    print mImRegion.shape[0]*mImRegion.shape[1]
                    if STRATIFY==True:
                        indices =stratified_sample(par_obj,binlength,samples_indices,imhist,samples_at_tiers,mImRegion,denseRegion)
                    else:
                        indices =  np.random.choice(int(mImRegion.shape[0]*mImRegion.shape[1]), size=int(par_obj.limit_size), replace=True, p=None)
                else:
                    #this works because the first n indices refer to the full first two dimensions, and np indexing takes slices
                    indices =  np.random.choice(int(mImRegion.shape[0]*mImRegion.shape[1]), size=int(par_obj.limit_size), replace=True, p=None)
                #Add to feature vector and output vector.
                par_obj.f_matrix.extend(mimg_lin[indices])
                par_obj.o_patches.extend(dense_lin[indices])
            else:
                #Add these to the end of the feature Matrix, input patches
                par_obj.f_matrix.extend(mimg_lin)
                #And the the output matrix, output patches
                par_obj.o_patches.extend(dense_lin)
def update_training_samples_fn_auto(par_obj,int_obj,rects):
    """Collects the pixels or patches which will be used for training and 
    trains the forest."""
    #Makes sure everything is refreshed for the training, encase any regions
    #were changed. May have to be rethinked for speed later on.
    region_size = 0
    '''
    for b in range(0,par_obj.saved_ROI.__len__()):
        rects = par_obj.saved_ROI[b]
        region_size += rects[4]*rects[3]       
    '''
    calc_ratio = par_obj.limit_ratio_size
    #print 'calcratio',calc_ratio
    STRATIFY=False
    dot_im=np.pad(np.ones((1,1)),(int(par_obj.sigma_data)*6,int(par_obj.sigma_data)*4),mode='constant')
    dot_im=filters.gaussian_filter(dot_im,float(par_obj.sigma_data),mode='constant',cval=0)
    dot_im/=dot_im.max()
    binlength=10
    imhist=np.histogram(dot_im,bins=binlength,range=(0,1),density=True)
    imhist[1][binlength]=5 # adjust top bin to make sure we include everthing if we have overlapping gaussians-try to avoid though-if very common will distort bins
    samples_at_tiers=(imhist[0]/binlength*par_obj.limit_size).astype('int')
    #print samples_at_tiers
    samples_indices = [0]+(np.cumsum(samples_at_tiers)).tolist() #to allow preallocation of array
    
    
    #for b in range(0,par_obj.saved_ROI.__len__()):
        #TODO check this works for edge cases- with very sparse sampling, and with v small bin sizes
        #Iterates through saved ROI.
        #rects = par_obj.saved_ROI[b]
    
    zslice = rects[0]
    tpt =rects[5]
    imno =rects[6]
    if(par_obj.p_size == 1):
        #if rects[5] == tpt and rects[0] == zslice and rects[6] == imno:
            print rects
            #Finds and extracts the features and output density for the specific regions.
            mImRegion = par_obj.data_store['double_feat_arr'][imno][tpt][zslice][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3],:]
            denseRegion = par_obj.data_store['dense_arr'][imno][tpt][zslice][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3]]
            #Find the linear form of the selected feature representation
            mimg_lin = np.reshape(mImRegion, (mImRegion.shape[0]*mImRegion.shape[1],mImRegion.shape[2]))
            #Find the linear form of the complementatory output region.
            dense_lin = np.reshape(denseRegion, (denseRegion.shape[0]*denseRegion.shape[1]))
            #Sample the input pixels sparsely or densely.
            if(par_obj.limit_sample == True):
                if(par_obj.limit_ratio == True):
                    par_obj.limit_size = round(mImRegion.shape[0]*mImRegion.shape[1]/calc_ratio,0)
                    print mImRegion.shape[0]*mImRegion.shape[1]
                    if STRATIFY==True:
                        indices =stratified_sample(par_obj,binlength,samples_indices,imhist,samples_at_tiers,mImRegion,denseRegion)
                    else:
                        indices =  np.random.choice(int(mImRegion.shape[0]*mImRegion.shape[1]), size=int(par_obj.limit_size), replace=True, p=None)
                else:
                    #this works because the first n indices refer to the full first two dimensions, and np indexing takes slices
                    indices =  np.random.choice(int(mImRegion.shape[0]*mImRegion.shape[1]), size=int(par_obj.limit_size), replace=True, p=None)
                #Add to feature vector and output vector.
                par_obj.f_matrix.extend(mimg_lin[indices])
                par_obj.o_patches.extend(dense_lin[indices])
            else:
                #Add these to the end of the feature Matrix, input patches
                par_obj.f_matrix.extend(mimg_lin)
                #And the the output matrix, output patches
                par_obj.o_patches.extend(dense_lin)
                
def train_forest(par_obj,int_obj,model_num):
            
    
    
    if par_obj.max_features>par_obj.num_of_feat[model_num]:
        par_obj.max_features=par_obj.num_of_feat[model_num]
    #par_obj.RF[model_num]=sklearn.ensemble.AdaBoostRegressor(base_estimator=sklearn.ensemble.ExtraTreesRegressor(20, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1), n_estimators=3, learning_rate=1.0, loss='linear')
    #par_obj.RF[model_num] = sklearn.ensemble.ExtraTreesClassifier(par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1,class_weight='balanced')
    par_obj.RF[model_num] = sklearn.ensemble.ExtraTreesRegressor(par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1)
    #par_obj.RF[model_num] = sklearn.ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features)    
    #par_obj.RF[model_num] = sklearn.linear_model.BayesianRidge(par_obj.n_iter, par_obj.tol, par_obj.alpha_1, par_obj.alpha_2, par_obj.lambda_1, par_obj.lambda_2)    
    #Fits the data.
    t3 = time.time()
    X=np.asfortranarray(par_obj.f_matrix)
    Y=np.asfortranarray(par_obj.o_patches)
    print 'fmatrix',X.shape
    print 'o_patches',Y.shape
    
    par_obj.RF[model_num].fit(X, Y)

    importances = par_obj.RF[model_num].feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    
    
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    '''    if f>par_obj.max_features: #retrain model with fewer features (for faster evaluation)
            X[:,f]=0
    par_obj.RF[model_num] = ExtraTreesRegressor(par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features= None, bootstrap=True, n_jobs=-1)
    par_obj.RF[model_num].fit(X, Y)'''
    t4 = time.time()
    print 'actual training',t4-t3 
    
    par_obj.o_patches=[]
    par_obj.f_matrix=[]    

def refresh_all_density(par_obj):
    number_of_saved_roi=range(0,len(par_obj.saved_ROI))

    for it in number_of_saved_roi:
        tpt=int(par_obj.saved_ROI[it][5])
        zslice=int(par_obj.saved_ROI[it][0])
        fileno=int(par_obj.saved_ROI[it][6])
        update_com_fn(par_obj,tpt,zslice,fileno)
def update_com_fn(par_obj,tpt,zslice,fileno):
    
    #Construct empty array for current image.
    dots_im = np.zeros((par_obj.height,par_obj.width))
    #In array of all saved dots.
    for i in range(0,par_obj.saved_dots.__len__()):
        #Any ROI in the present image.
        #print 'iiiii',win.saved_dots.__len__()
        if(par_obj.saved_ROI[i][0] == zslice and par_obj.saved_ROI[i][5] == tpt and par_obj.saved_ROI[i][6] == fileno):
            #Save the corresponding dots.
            dots = par_obj.saved_dots[i]
            #Scan through the dots
            for b in range(0,dots.__len__()):
               
                #save the column and row 
                c_dot = dots[b][2]
                r_dot = dots[b][1]
                #Set it to register as dot.
                dots_im[c_dot, r_dot] = 1 #change from 255
    #Convolve the dots to represent density estimation.
    print 'Using template matching to generate C-O-M representation'
    dense_im = np.zeros(dots_im.shape).astype(np.float64)
    
    size_of_kernel = np.ceil(par_obj.sigma_data * 6) #At least the 3-sigma rule.
    if size_of_kernel % 2 ==0:
        size_of_kernel = int(size_of_kernel + 1)
    
    patch = np.zeros((size_of_kernel,size_of_kernel))
    m_p = int((size_of_kernel-1)/2)
    patch[m_p,m_p] = 1
    
    kernel = filters.gaussian_filter(patch.astype(np.float32),   float(par_obj.sigma_data), order=0, output=None, mode='reflect', cval=0.0)
    #kernel = distance_transform_edt(patch==0).astype(np.float32)
    #kernel=kernel/np.max(kernel)    
    #kernel=(kernel>0.5).astype('float32')
    #kernel=-kernel+np.max(kernel)
    #kernel=kernel/np.max(kernel)
    #kernel=np.log(kernel*1000+1)
    #Replace member of dense_array with new image.
    r_arr, c_arr = np.where(dots_im > 0)
    for r, c in zip(r_arr,c_arr):
        p1 = 0
        p2 = patch.shape[0]
        p3 = 0
        p4 = patch.shape[1]
        
        r1 = int(r-m_p);
        r2 = int(r+m_p+1);
        c1 = int(c-m_p);
        c2 = int(c+m_p+1);
        
        if r1 <0:
            p1 = abs(r1-0)
            r1 =0
        if r2 > dots_im.shape[0]:
            p2 = (patch.shape[0]-1)- (abs(r2-dots_im.shape[0]))+1
            r2 = dots_im.shape[0]       
        if c1 <0:
            p3 = abs(c1-0)
            c1 =0
        if c2 > dots_im.shape[1]:
            p4 = (patch.shape[1]-1)- (abs(c2-dots_im.shape[1]))+1
            c2 = dots_im.shape[1]
        
        
        
        dense_im[r1:r2,c1:c2] =  np.max([kernel[p1:p2,p3:p4],dense_im[r1:r2,c1:c2]],0)
    
    par_obj.data_store['dense_arr'][fileno][tpt][zslice] = dense_im
    '''NORMALISE GAUSSIANS. THIS MAKES IT USELESS FOR DOING 2D DENSITY ESTIMATION,
 but super useful if you want a probability output between 0 and 1 at the end of the day
for thresholding and the like'''
    dense_im=dense_im/kernel.max()*1000
    par_obj.gaussian_im_max=kernel.max()
    #TODO? Possibly we could make the density counting assumption in 3D and use it for counting, but at the end of the day, I think we really want to know where they are
    #TODO Now that I'm normalising at this step, probably should check everything else to make sure that I only normalise here -suspect that there is a normalisation step somewhere in the processing of the probability output
    #Replace member of dense_array with new image.
    par_obj.data_store['dense_arr'][fileno][tpt][zslice] = dense_im
    
def update_density_fn_new(par_obj,tpt,zslice,imno):
    #Construct empty array for current image.
    dots_im = np.zeros((par_obj.height,par_obj.width))
    #In array of all saved dots.
    if par_obj.saved_dots.__len__()>0:
        for i in range(0,par_obj.saved_dots.__len__()):
            #Any ROI in the present image.
            #print 'iiiii',win.saved_dots.__len__()
            if(par_obj.saved_ROI[i][0] == zslice and par_obj.saved_ROI[i][5] == tpt and par_obj.saved_ROI[i][6] == imno):
                #Save the corresponding dots.
                dots = par_obj.saved_dots[i]
                #Scan through the dots
                for b in range(0,dots.__len__()):
                   
                    #save the column and row 
                    c_dot = dots[b][2]
                    r_dot = dots[b][1]
                    #Set it to register as dot.
                    dots_im[c_dot, r_dot] = 1 #change from 255
        #Convolve the dots to represent density estimation.
        #dense_im = filters.gaussian_laplace(dots_im.astype(np.float32),float(par_obj.sigma_data), output=None, mode='reflect', cval=0.0)
        dense_im = filters.gaussian_filter(dots_im.astype(np.float32),float(par_obj.sigma_data), order=0, output=None, mode='reflect', cval=0.0)
        '''NORMALISE GAUSSIANS. THIS MAKES IT USELESS FOR DOING 2D DENSITY ESTIMATION,
 but super useful if you want a probability output between 0 and 1 at the end of the day
for thresholding and the like'''
        if not par_obj.gaussian_im_max:
            dot_im=filters.gaussian_filter(np.ones((1,1)),float(par_obj.sigma_data),mode='constant',cval=0)
            par_obj.gaussian_im_max=dot_im[0,0]

        dense_im=dense_im/par_obj.gaussian_im_max*10000
        #TODO? Possibly we could make the density counting assumption in 3D and use it for counting, but at the end of the day, I think we really want to know where they are
        #TODO Now that I'm normalising at this step, probably should check everything else to make sure that I only normalise here -suspect that there is a normalisation step somewhere in the processing of the probability output
        #Replace member of dense_array with new image.
        par_obj.data_store['dense_arr'][imno][tpt][zslice] = dense_im

def im_pred_inline_fn_new(par_obj, int_obj,zsliceList,tptList,imnoList,threaded=False):
    """Accesses TIFF file slice or opens png. Calculates features to indices present in par_obj.left_2_calc"""
    print 'version1'    
    # consider cropping
    if par_obj.to_crop == False:
        par_obj.crop_x1 = 0
        par_obj.crop_x2=par_obj.width
        par_obj.crop_y1 = 0
        par_obj.crop_y2=par_obj.height
    par_obj.height = par_obj.crop_y2-par_obj.crop_y1
    par_obj.width = par_obj.crop_x2-par_obj.crop_x1
    if par_obj.FORCE_nothreading is not False:
        threaded=par_obj.FORCE_nothreading
    if threaded=='Z':
        imRGBlist=[]
        for tpt in tptList:
            for imno in imnoList:
                for zslice in zsliceList:
                    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:
                        #imRGB=return_imRGB_slice_new(par_obj,zslice,tpt,imno)
                        imRGB = get_tiff_slice(par_obj,[tpt],[zslice],range(0,par_obj.ori_width,int(par_obj.resize_factor)),range(0,par_obj.ori_height,int(par_obj.resize_factor)),par_obj.ch_active,imno)

                        #imRGBlist.append(imRGB)
                        imRGBlist.append(imRGB.astype('float32')/ par_obj.tiffarraymax)    
                #initiate pool and start caclulating features
                int_obj.report_progress('Calculating Features for Timepoint: '+str(tpt+1) +' All  Frames'+' File: '+str(tpt+1))
                featlist=[]
                tee1=time.time()
                pool = ThreadPool(8) 
                featlist=pool.map(functools.partial(feature_create_threadable,par_obj),imRGBlist)
                pool.close() 
                pool.join() 
                tee2=time.time()
                #feat =feature_create(par_obj,imRGB,imStr,i)
                print tee2-tee1
                lcount=-1

                for zslice in zsliceList:
                    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:
                        lcount=lcount+1
                        if zslice == zsliceList[0]:
                            feat=np.concatenate((featlist[lcount],featlist[lcount],featlist[lcount],featlist[lcount+1],featlist[lcount+2]),axis=2)
                        elif zslice == zsliceList[-1]:
                            feat=np.concatenate((featlist[lcount-2],featlist[lcount-1],featlist[lcount],featlist[lcount],featlist[lcount]),axis=2)
                        elif zslice == zsliceList[1]:
                            feat=np.concatenate((featlist[lcount],featlist[lcount-1],featlist[lcount],featlist[lcount+1],featlist[lcount+2]),axis=2)
                        elif zslice == zsliceList[-2]:
                            feat=np.concatenate((featlist[lcount-2],featlist[lcount-1],featlist[lcount],featlist[lcount+1],featlist[lcount+1]),axis=2)

                        else:
                            feat=np.concatenate((featlist[lcount-2],featlist[lcount-1],featlist[lcount],featlist[lcount+1],featlist[lcount+2]),axis=2)
                        int_obj.report_progress('Calculating Features for Z: '+str(zslice+1)+' Timepoint: '+str(tpt+1)+' File: '+str(imno+1))
    
                        par_obj.num_of_feat[0] = feat.shape[2]
    
                        par_obj.data_store['feat_arr'][imno][tpt][zslice] = feat  
        imRGBlist=[]
        for tpt in tptList:
            for imno in imnoList:
                for zslice in zsliceList:
                    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:
                        #imRGB=return_imRGB_slice_new(par_obj,zslice,tpt,imno)
                        imRGB = get_tiff_slice(par_obj,[tpt],[zslice],range(0,par_obj.ori_width,int(par_obj.resize_factor)),range(0,par_obj.ori_height,int(par_obj.resize_factor)),par_obj.ch_active,imno)

                        #imRGBlist.append(imRGB)
                        imRGBlist.append(imRGB.astype('float32')/ par_obj.tiffarraymax)    
                #initiate pool and start caclulating features
                int_obj.report_progress('Calculating Features for Timepoint: '+str(tpt+1) +' All  Frames'+' File: '+str(tpt+1))
                featlist=[]
                tee1=time.time()
                pool = ThreadPool(8) 
                featlist=pool.map(functools.partial(feature_create_threadable,par_obj),imRGBlist)
                pool.close() 
                pool.join() 
                tee2=time.time()
                #feat =feature_create(par_obj,imRGB,imStr,i)
                print tee2-tee1
                lcount=-1

                for zslice in zsliceList:
                    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:
                        lcount=lcount+1
                        if zslice == zsliceList[0]:
                            feat=np.concatenate((featlist[lcount],featlist[lcount],featlist[lcount],featlist[lcount+1],featlist[lcount+2]),axis=2)
                        elif zslice == zsliceList[-1]:
                            feat=np.concatenate((featlist[lcount-2],featlist[lcount-1],featlist[lcount],featlist[lcount],featlist[lcount]),axis=2)
                        elif zslice == zsliceList[1]:
                            feat=np.concatenate((featlist[lcount],featlist[lcount-1],featlist[lcount],featlist[lcount+1],featlist[lcount+2]),axis=2)
                        elif zslice == zsliceList[-2]:
                            feat=np.concatenate((featlist[lcount-2],featlist[lcount-1],featlist[lcount],featlist[lcount+1],featlist[lcount+1]),axis=2)

                        else:
                            feat=np.concatenate((featlist[lcount-2],featlist[lcount-1],featlist[lcount],featlist[lcount+1],featlist[lcount+2]),axis=2)
                        int_obj.report_progress('Calculating Features for Z: '+str(zslice+1)+' Timepoint: '+str(tpt+1)+' File: '+str(imno+1))
    
                        par_obj.num_of_feat[0] = feat.shape[2]
    
                        par_obj.data_store['feat_arr'][imno][tpt][zslice] = feat  

    elif threaded == False:
        
        print imnoList
        print tptList
        print zsliceList
        for imno in imnoList:
            for tpt in tptList:
                for zslice in zsliceList:
                    #checks if features already in array
                    
                    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:
                        #imRGB=return_imRGB_slice_new(par_obj,zslice,tpt,imno)
                        #imRGB/=par_obj.tiffarraymax
                        

                        imRGB = get_tiff_slice(par_obj,[tpt],zslice,range(0,par_obj.ori_width,int(par_obj.resize_factor)),range(0,par_obj.ori_height,int(par_obj.resize_factor)),par_obj.ch_active,imno)

                        #imRGBlist.append(imRGB)
                        imRGB=imRGB.astype('float32')/ par_obj.tiffarraymax
                        #If you want to ignore previous features which have been saved.
                        int_obj.report_progress('Calculating Features for Z:' +str(zslice+1) +' Timepoint: '+str(tpt+1)+' File: '+str(imno+1))
                        
                        feat =feature_create_threadable(par_obj,imRGB)
                        
                        par_obj.num_of_feat[0] = feat.shape[2]
                        par_obj.data_store['feat_arr'][imno][tpt][zslice] = feat
                        
    elif threaded == 'auto':
        #threaded version
        imRGBlist=[]
        for tpt in tptList:
            for imno in imnoList:
                featlist=[]
                tee1=time.time()
                pool = ThreadPool(8) 
                featlist=pool.map(functools.partial(feature_create_threadable_auto,par_obj,imno,tpt),zsliceList)
                pool.close() 
                pool.join() 
                tee2=time.time()
                #feat =feature_create(par_obj,imRGB,imStr,i)
                print tee2-tee1
                lcount=-1
    
                for zslice in zsliceList:
                        lcount=lcount+1
                        feat=featlist[lcount]
                        int_obj.report_progress('Calculating Features for Z: '+str(zslice+1)+' Timepoint: '+str(tpt+1)+' File: '+str(imno+1))
    
                        par_obj.num_of_feat[1] = feat.shape[2]+par_obj.num_of_feat[0]
    
                        par_obj.data_store['double_feat_arr'][imno][tpt][zslice] = np.concatenate((feat,par_obj.data_store['feat_arr'][imno][tpt][zslice]),axis=2)

    else:
        #threaded version
        imRGBlist=[]
        for tpt in tptList:
            for imno in imnoList:
                for zslice in zsliceList:
                    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:
                        #imRGB=return_imRGB_slice_new(par_obj,zslice,tpt,imno)
                        imRGB = get_tiff_slice(par_obj,[tpt],[zslice],range(0,par_obj.ori_width,int(par_obj.resize_factor)),range(0,par_obj.ori_height,int(par_obj.resize_factor)),par_obj.ch_active,imno)

                        #imRGBlist.append(imRGB)
                        imRGBlist.append(imRGB.astype('float32')/ par_obj.tiffarraymax)    
                #initiate pool and start caclulating features
                int_obj.report_progress('Calculating Features for Timepoint: '+str(tpt+1) +' All  Frames'+' File: '+str(tpt+1))
                featlist=[]
                tee1=time.time()
                pool = ThreadPool(8) 
                featlist=pool.map(functools.partial(feature_create_threadable,par_obj),imRGBlist)
                pool.close() 
                pool.join() 
                tee2=time.time()
                #feat =feature_create(par_obj,imRGB,imStr,i)
                print tee2-tee1
                lcount=-1
    
                for zslice in zsliceList:
                    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:
                        lcount=lcount+1
                        feat=featlist[lcount]
                        int_obj.report_progress('Calculating Features for Z: '+str(zslice+1)+' Timepoint: '+str(tpt+1)+' File: '+str(imno+1))
    
                        par_obj.num_of_feat[0] = feat.shape[2]
    
                        par_obj.data_store['feat_arr'][imno][tpt][zslice] = feat  
        int_obj.report_progress('Features calculated')
    return

def return_imRGB_slice_new(par_obj,zslice,tpt,imno):
    '''Fetches slice zslice of timepoint tpt and puts in RGB format for display'''
    if par_obj.file_ext == 'tif' or par_obj.file_ext == 'tiff':
        
        imRGB = np.zeros((int(par_obj.height),int(par_obj.width),3),'float32')
        if par_obj.ch_display.__len__() > 1 or (par_obj.ch_display.__len__() == 1 and par_obj.numCH>1):
            #input_im = par_obj.tiffarray[tpt,zslice,::int(par_obj.resize_factor),::int(par_obj.resize_factor),:]
            #imRGB = np.zeros((int(par_obj.height),int(par_obj.width),par_obj.ch_active.__len__()))
            if par_obj.numCH>3:
                for i,ch in enumerate(par_obj.ch_display):
                    if i==3: break
                    t0=time.time()
                    input_im = get_tiff_slice(par_obj,[tpt],[zslice],range(0,par_obj.ori_width,int(par_obj.resize_factor)),range(0,par_obj.ori_height,int(par_obj.resize_factor)),[ch],imno=imno)
                    imRGB[:,:,i] = input_im
            else:
                for i,ch in enumerate(par_obj.ch_display):
                    t0=time.time()
                    input_im = get_tiff_slice(par_obj,[tpt],[zslice],range(0,par_obj.ori_width,int(par_obj.resize_factor)),range(0,par_obj.ori_height,int(par_obj.resize_factor)),[ch],imno=imno)
                    imRGB[:,:,ch] = input_im

        else:
            #input_im = par_obj.tiffarray[tpt,zslice,::int(par_obj.resize_factor),::int(par_obj.resize_factor)]
            input_im = get_tiff_slice(par_obj,[tpt],zslice,range(0,par_obj.ori_width,par_obj.resize_factor),range(0,par_obj.ori_height,par_obj.resize_factor),imno=imno)

            imRGB[:,:,0] = input_im
            imRGB[:,:,1] = input_im
            imRGB[:,:,2] = input_im

    elif par_obj.file_ext == 'png':
        imRGB = pylab.imread(str(imStr))*255
    elif par_obj.file_ext == 'oib'or  par_obj.file_ext == 'oif':
        
        imRGB = np.zeros((int(par_obj.height),int(par_obj.width),3))

        print 'par_obj.ch_active',par_obj.ch_active
        for c in range(0,par_obj.ch_active.__len__()):
            
            f  = par_obj.oib_prefix+'/s_C'+str(par_obj.ch_active[c]+1).zfill(3)+'Z'+str(zslice+1).zfill(3)
            if par_obj.total_time_pt>0:
                f = f+'T'+str(tpt+1).zfill(3)
            f = f+'.tif'
            
            imRGB[:,:,par_obj.ch_active[c]] = (par_obj.oib_file.asarray(f)[::int(par_obj.resize_factor),::int(par_obj.resize_factor)])/16
    return imRGB


def evaluate_forest_new(par_obj,int_obj,withGT,model_num,zsliceList,tptList,curr_file,threaded=False,b=0,arr='feat_arr'):

    #Finds the current frame and file.
    par_obj.maxPred=0 #resets scaling for display between models
    par_obj.minPred=100
    for imno in curr_file:
        for tpt in tptList:
            for zslice in zsliceList:
                if(par_obj.p_size >1):
                    
                    mimg_lin,dense_linPatch, pos = extractPatch(par_obj.p_size, par_obj.feat_arr[zslice], None, 'dense')
                    tree_pred = par_obj.RF[model_num].predict(mimg_lin)
                    
                    linPred = v2.regenerateImg(par_obj.p_size, tree_pred, pos)
                        
                else:
                    
                    mimg_lin = np.reshape(par_obj.data_store[arr][imno][tpt][zslice], (par_obj.height * par_obj.width, par_obj.data_store[arr][imno][tpt][zslice].shape[2]))
                    t2 = time.time()
                    linPred = par_obj.RF[model_num].predict(mimg_lin)
                    #linPred=linPred[:,1]-linPred[:,0]
                    t1 = time.time()
                    
    
    
                par_obj.data_store['pred_arr'][imno][tpt][zslice] = linPred.reshape(par_obj.height, par_obj.width)
    
                maxPred = np.max(linPred)
                minPred = np.min(linPred)
                par_obj.maxPred=max([par_obj.maxPred,maxPred])
                par_obj.minPred=min([par_obj.minPred,minPred])
                sum_pred =np.sum(linPred/255)
                par_obj.data_store['sum_pred'][imno][tpt][zslice] = sum_pred
                
                print 'prediction time taken',t1 - t2
                print 'Predicted i:',par_obj.data_store['sum_pred'][imno][tpt][zslice]
                int_obj.report_progress('Making Prediction for Image: '+str(b+1)+' Frame: ' +str(zslice+1)+' Timepoint: '+str(tpt+1))
                        
    
                if withGT == True:
                    try:
                        #If it has already been opened.
                        a = par_obj.data_store['gt_sum'][imno][tpt][zslice]
                    except:
                        #Else find the file.
                        gt_im =  pylab.imread(par_obj.data_store['gt_array'][imno][tpt][zslice])[:,:,0]
                        par_obj.data_store['gt_sum'][imno][tpt][zslice] = np.sum(gt_im)

def evaluate_forest_auto(par_obj,int_obj,withGT,model_num,zsliceList,tptList,curr_file,threaded=False,b=0):

    #Finds the current frame and file.
    par_obj.maxPred=0 #resets scaling for display between models
    par_obj.minPred=100
    for imno in curr_file:
        for tpt in tptList:
            for zslice in zsliceList:
                if(par_obj.p_size >1):
                    
                    mimg_lin,dense_linPatch, pos = extractPatch(par_obj.p_size, par_obj.feat_arr[zslice], None, 'dense')
                    tree_pred = par_obj.RF[model_num].predict(mimg_lin)
                    
                    linPred = v2.regenerateImg(par_obj.p_size, tree_pred, pos)
                        
                else:
                    
                    mimg_lin = np.reshape(par_obj.data_store['double_feat_arr'][imno][tpt][zslice], (par_obj.height * par_obj.width, par_obj.data_store['double_feat_arr'][imno][tpt][zslice].shape[2]))
                    t2 = time.time()
                    linPred = par_obj.RF[model_num].predict(mimg_lin)


                    t1 = time.time()
                    
    
    
                par_obj.data_store['pred_arr'][imno][tpt][zslice] = linPred.reshape(par_obj.height, par_obj.width)
    
                maxPred = np.max(linPred)
                minPred = np.min(linPred)
                par_obj.maxPred=max([par_obj.maxPred,maxPred])
                par_obj.minPred=min([par_obj.minPred,minPred])
                sum_pred =np.sum(linPred/255)
                par_obj.data_store['sum_pred'][imno][tpt][zslice] = sum_pred
                
                print 'prediction time taken',t1 - t2
                print 'Predicted i:',par_obj.data_store['sum_pred'][imno][tpt][zslice]
                int_obj.report_progress('Making Prediction for Image: '+str(b+1)+' Frame: ' +str(zslice+1)+' Timepoint: '+str(tpt+1))
                        
    
                if withGT == True:
                    try:
                        #If it has already been opened.
                        a = par_obj.data_store['gt_sum'][imno][tpt][zslice]
                    except:
                        #Else find the file.
                        gt_im =  pylab.imread(par_obj.data_store['gt_array'][imno][tpt][zslice])[:,:,0]
                        par_obj.data_store['gt_sum'][imno][tpt][zslice] = np.sum(gt_im)

def channels_for_display2(par_obj, int_obj,imRGB):
    '''deals with displaying different channels'''
    count = 0
    '''
    CH = [0]*5 #up to 5 channels-get checkbox values
    for c in range(0,par_obj.numCH):
        if int_obj.CH_cbx[c].isChecked():
            count = count + 1
            CH[c] = 1
    '''
    count=par_obj.numCH
    newImg =np.zeros((par_obj.height,par_obj.width,3),'uint8')
    newImg=create_channels_image(newImg,par_obj.numCH,par_obj.ch_display,count,(255*imRGB).astype('uint8'))
    return newImg

def create_channels_image(newImg,numCH,CH,count,imRGB):
    '''
    if count == 0 and numCH==0: #deals with the none-ticked case
        newImg = imRGB
        elif count == 1: #only one channel selected-display in grayscale # this may be broken or unecessary
            newImg[:,:,0] = imRGB
            newImg[:,:,1] = imRGB
            newImg[:,:,2] = imRGB
    elif count ==3:
        newImg = imRGB
    else:
        if CH[0] == 1:
            newImg[:,:,0] = imRGB[:,:,0]
        if CH[1] == 1:
            newImg[:,:,1] = imRGB[:,:,1]
        if CH[2] == 1:
            newImg[:,:,2] = imRGB[:,:,2]
    '''
    if count==0:
        newImg = imRGB
    elif count==3 and len(CH)==3:
        newImg = imRGB
    elif count<=3:
        for ch in enumerate(CH):
            newImg[:,:,ch] = imRGB[:,:,ch]
    else:
        for di, ch in enumerate(CH):
            if di==3:break
            newImg[:,:,di] = imRGB[:,:,ch]
    return newImg

def goto_img_fn_new(par_obj, int_obj):
    """Loads up requested image and displays"""
    b=0
    tpt=par_obj.time_pt    
    zslice=par_obj.curr_z
    imno=par_obj.curr_file
    t0=time.time()
    #Finds the current frame and file.
    newImg=return_imRGB_slice_new(par_obj,zslice,tpt,imno)
    par_obj.save_im = newImg
    #deals with displaying different channels
    newImg/=par_obj.tiffarraymax
    #newImg=channels_for_display2(par_obj, int_obj,newImg)
    
    if par_obj.overlay and zslice in par_obj.data_store['pred_arr'][imno][tpt]:
        newImg[:,:,2]= (par_obj.data_store['pred_arr'][imno][tpt][zslice])/par_obj.maxPred

    for i in range(0,int_obj.plt1.lines.__len__()):
        int_obj.plt1.lines.pop(0)

    
    int_obj.plt1.images[0].set_data(newImg)
    #int_obj.plt1.set_ylim([0,newImg.shape[0]])
    #int_obj.plt1.set_xlim([newImg.shape[1],0])
    #int_obj.plt1.axis("off")
    #int_obj.plt1.set_xticklabels([])
    #int_obj.plt1.set_yticklabels([])
    
    int_obj.image_num_txt.setText('The Current image is No. ' + str(par_obj.curr_z+1)+' and the time point is: '+str(par_obj.time_pt+1)) # filename: ' +str(evalLoadImWin.file_array[im_num]))

    """Deals with displaying Kernel/Prediction/Counts""" 
#    int_obj.image_num_txt.setText('The Current Image is No. ' + str(zslice+1)+' and the time point is: '+str(tpt+1))
    im2draw=None
    if par_obj.show_pts == 0:
        if zslice in par_obj.data_store['dense_arr'][imno][tpt]:
            im2draw = par_obj.data_store['dense_arr'][imno][tpt][zslice]
            int_obj.plt2.images[0].set_data(im2draw)
            int_obj.plt2.images[0].autoscale()
            int_obj.canvas2.draw()

        elif np.sum(int_obj.plt2.images[0].get_array()) != 0: #don't update if just zeros 
            im2draw = np.zeros((par_obj.height,par_obj.width))
            int_obj.plt2.images[0].set_data(im2draw)
            int_obj.canvas2.draw()
            print 'test2'

    elif par_obj.show_pts == 1:
        if zslice in par_obj.data_store['pred_arr'][imno][tpt]:
            im2draw = par_obj.data_store['pred_arr'][imno][tpt][zslice].astype(np.float32)
        else:
            im2draw = np.zeros((par_obj.height,par_obj.width))
        int_obj.plt2.images[0].set_data(im2draw)
        int_obj.plt2.images[0].set_clim(None,par_obj.maxPred)
        int_obj.canvas2.draw()
        
    elif par_obj.show_pts == 2:
        #show det(hessian) array, and (I assume) the green circles?
        pt_x = []
        pt_y = []
        pts = par_obj.data_store['pts'][imno][tpt]
        
        ind = np.where(np.array(par_obj.frames_2_load) == zslice)
        #print 'ind',ind
        for pt2d in pts:
            if pt2d[2] == ind:
                    pt_x.append(pt2d[1])
                    pt_y.append(pt2d[0])
        #int_obj.plt2.clear()
        int_obj.plt2.lines = []

        int_obj.plt2.axes.plot(pt_x,pt_y, 'wo')
        int_obj.plt2.autoscale_view(tight=True)
        
        int_obj.plt1.axes.plot(pt_x,pt_y, 'wo')
        int_obj.plt1.autoscale_view(tight=True)
        
        string_2_show = 'The Predicted Count: ' + str(pts.__len__())
        if zslice in par_obj.data_store['maxi_arr'][imno][tpt]:
            im2draw = par_obj.data_store['maxi_arr'][imno][tpt][zslice].astype(np.float32)
            int_obj.plt2.images[0].set_data(im2draw)
            int_obj.plt2.images[0].set_clim(0,1)
            int_obj.canvas2.draw()



    #int_obj.plt2.set_ylim([0,newImg.shape[0]])
    #int_obj.plt2.set_xlim([newImg.shape[1],0])
    int_obj.draw_saved_dots_and_roi()
    int_obj.cursor.draw_ROI()
    
    #ax=int_obj.plt2.axes.images[0]
    #int_obj.plt2.axes.draw_artist(ax)
    print time.time() -t0
    
    #pdb.set_trace()
    #int_obj.canvas1.draw()
    #int_obj.canvas2.draw()


def load_and_initiate_plots(par_obj, int_obj):
    """prepare plots and data for display"""
    #b=0    #Finds the current frame and file.

    if ( par_obj.file_ext == 'png'):
        imStr = str(par_obj.file_array[b])
        imRGB = pylab.imread(imStr)*255

    if ( par_obj.file_ext == 'tif' or  par_obj.file_ext == 'tiff'):
        print
        #FIXIT We do this at load data, is it really necessary here?
        #par_obj.tiffarray[imno]=par_obj.tiff_file.asarray(memmap=True)
        #par_obj.tiffarraymax = par_obj.tiffarray[imno].max()
            
    if ( par_obj.file_ext == 'oib' or par_obj.file_ext == 'oif'):
        imRGB = np.zeros((int(par_obj.height),int(par_obj.width),3))
        for c in range(0,par_obj.numCH):
            f  = par_obj.oib_prefix+'/s_C'+str(c+1).zfill(3)+'Z'+str(zslice+1).zfill(3)
            if par_obj.total_time_pt>0:
                f = f+'T'+str(tpt+1).zfill(3)
            f = f+'.tif'
            imRGB[:,:,c] = (par_obj.oib_file.asarray(f)[::int(par_obj.resize_factor),::int(par_obj.resize_factor)])/16
    
    #newImg=np.zeros((int(par_obj.height),int(par_obj.width),4),'uint8')
    newImg=np.zeros((int(par_obj.height),int(par_obj.width),3),'uint8')
    #newImg[:,:,3]=1
    int_obj.plt1.cla()
    modest_image.imshow(int_obj.plt1.axes,newImg,interpolation='nearest', vmin=0, vmax=255)
    #int_obj.plt1.imshow(newImg,interpolation='nearest')
    int_obj.plt1.axis("off")
    newImg=np.zeros((int(par_obj.height),int(par_obj.width),3))
    int_obj.plt2.cla()
    int_obj.plt2.imshow(newImg,interpolation='none')
    #modest_image.imshow(int_obj.plt2.axes,newImg,interpolation='none')
    int_obj.plt2.axis("off")
    int_obj.plt2.autoscale()
    int_obj.cursor.draw_ROI()
    int_obj.image_num_txt.setText('The Current image is No. ' + str(par_obj.curr_z+1)+' and the time point is: '+str(par_obj.time_pt+1)) # filename: ' +str(evalLoadImWin.file_array[im_num]))
    
    goto_img_fn_new(par_obj, int_obj)
    
def detHess(predMtx,min_distance):
    gau_stk = filters.gaussian_filter(predMtx,min_distance)
    gradlist = np.gradient(gau_stk,1)

    pool = ThreadPool(3) 
    grad2list=pool.map(np.gradient,gradlist)
    pool.close()
    pool.join() 
    xy,xx,xz = grad2list[0]
    yy,yx,yz = grad2list[1]
    zy,zx,zz = grad2list[2]
    det = -1*((((yy*zz)-(yz*yz))*xx)-(((xy*zz)-(yz*xz))*xy)+(((xy*yz)-(yy*xz))*xz))
    detl = -1*np.min(det)+det
    return detl
def eval_pred_show_fn(par_obj,int_obj,zslice,tpt):
    """Shows Prediction Image when forest is loaded"""
    if par_obj.eval_load_im_win_eval == True:
        
        int_obj.image_num_txt.setText('The Current Image is No. ' + str(zslice+1)+' and the time point is: '+str(tpt+1))

        #if int_obj.count_maxima_plot_on.isChecked() == True:
        #    par_obj.show_pts = True
        #else:
    #        par_obj.show_pts = False

        int_obj.plt2.cla()
        if par_obj.show_pts == 0:
            im2draw = np.zeros((par_obj.height,par_obj.width))
            for ind in par_obj.data_store['dense_arr'][imno][tpt]:
                if ind == zslice:
                    im2draw = par_obj.data_store[tpt]['dense_arr'][zslice].astype(np.float32)
            int_obj.plt2.imshow(im2draw)
        if par_obj.show_pts == 1:
            im2draw = np.zeros((par_obj.height,par_obj.width))
            for ind in par_obj.data_store['pred_arr'][imno][tpt]:
                if ind == zslice:
                    im2draw = par_obj.data_store['pred_arr'][imno][tpt][zslice].astype(np.float32)
            int_obj.plt2.imshow(im2draw,vmax=par_obj.maxPred)
        if par_obj.show_pts == 2:
            
            pt_x = []
            pt_y = []
            pts = par_obj.data_store['pts'][imno][tpt]
            
            ind = np.where(np.array(par_obj.frames_2_load[0]) == zslice)
            print 'ind',ind
            for pt2d in pts:
                if pt2d[2] == ind:
                        pt_x.append(pt2d[1])
                        pt_y.append(pt2d[0])


            int_obj.plt1.plot(pt_x,pt_y, 'wo')
            int_obj.plt2.plot(pt_x,pt_y, 'wo')
            string_2_show = 'The Predicted Count: ' + str(pts.__len__())
            int_obj.output_count_txt.setText(string_2_show)
            im2draw = np.zeros((par_obj.height,par_obj.width))
            for ind in par_obj.data_store['maxi_arr'][imno][tpt]:
                if ind == zslice:
                    im2draw = par_obj.data_store['maxi_arr'][imno][tpt][zslice].astype(np.float32)
            d = int_obj.plt2.imshow(im2draw)
            d.set_clim(0,255)

        int_obj.plt2.set_xticklabels([])
        int_obj.plt2.set_yticklabels([])
        int_obj.canvas1.draw()
        int_obj.canvas2.draw()
def get_tiff_slice(par_obj,tpt=[0],zslice=[0],x=[0],y=[0],c=[0],imno=0):
    #deal with different TXYZC orderings. Always return TZYXC
    if type(zslice) is not list: zslice = [zslice]
    if type(zslice[0]) is list: zslice = zslice[0]
    t0=time.time()
    alist=[]
    blist=[]
    for n,b in enumerate(par_obj.order):
        if b=='T':
            alist.append(tpt)
            blist.append(n)
    for n,b in enumerate(par_obj.order):
        if b=='Z':
            alist.append(zslice)
            blist.append(n)
    for n,b in enumerate(par_obj.order):
        if b=='Y':
            alist.append(y)
            blist.append(n)
    for n,b in enumerate(par_obj.order):
        if b=='X':
            alist.append(x)
            blist.append(n)

    for n,b in enumerate(par_obj.order):
        if b=='C' or b=='S':
            alist.append(c)
            blist.append(n)
    
    tiff2=par_obj.tiffarray[imno].transpose(blist)
    print 'tiff',tiff2.shape
    if par_obj.order.__len__()==5:
        #tiff=np.squeeze(tiff2[alist[0],:,:,:,:][:,alist[1],:,:,:][:,:,alist[2],:,:][:,:,:,alist[3],:][:,:,:,:,alist[4]])
        tiff=np.squeeze(tiff2[np.ix_(alist[0],alist[1],alist[2],alist[3],alist[4])])
    elif par_obj.order.__len__()==4:
        #tiff=np.squeeze(tiff2[alist[0],:,:,:][:,alist[1],:,:][:,:,alist[2],:][:,:,:,alist[3]])
        tiff=np.squeeze(tiff2[np.ix_(alist[0],alist[1],alist[2],alist[3])])

    elif par_obj.order.__len__()==3:
        #tiff=np.squeeze(tiff2[alist[0],:,:][:,alist[1],:][:,:,alist[2]])
        tiff=np.squeeze(tiff2[np.ix_(alist[0],alist[1],alist[2])])
    elif par_obj.order.__len__()==2:
        #tiff=np.squeeze(tiff2[alist[0],:][:,alist[1]])
        tiff=np.squeeze(tiff2[np.ix_(alist[0],alist[1])])
    print 'tiff2',tiff.shape
    return tiff
    
def import_data_fn(par_obj,file_array):
    """Function which loads in Tiff stack or single png file to assess type."""
    prevExt = [] 
    prevBitDepth=[] 
    prevNumCH =[]
    par_obj.numCH = 0
    #par_obj.total_time_pt = 0
    
    par_obj.max_file= file_array.__len__()    
        
    
    for imno in range(0,par_obj.max_file):
            n = str(imno)
            print imno
            imStr = str(file_array[imno])
            par_obj.file_ext = imStr.split(".")[-1]
            par_obj.file_name[imno] = imStr.split(".")[0].split("/")[-1]
            if prevExt != [] and prevExt !=par_obj.file_ext:
                statusText = 'More than one file format present. Different number of image channels in the selected images'
                return False, statusText


            if par_obj.file_ext == 'tif' or par_obj.file_ext == 'tiff':
                par_obj.tiff_file = TiffFile(imStr)
                par_obj.tiffarraymax=0
                meta = par_obj.tiff_file.series[0]
                try: #if an imagej file, we know where, and can extract the x,y,z
                    x = par_obj.tiff_file.pages[0].tags.x_resolution.value
                    y = par_obj.tiff_file.pages[0].tags.x_resolution.value
                    if x!=y: raise Exception('x resolution different to y resolution')# if this isn't true then something is wrong
                    x_res=float(x[1])/float(x[0])
                    
                    z=par_obj.tiff_file.pages[0].imagej_tags['spacing']
                    par_obj.z_calibration[imno] = z/x_res
                    #TODO report Z-scale factor in an (editable) field
                    print('z_scale_factor', par_obj.z_calibration)
                except:
                    #might need to modify this to work with OME-TIFFs
                    print 'tiff resolution not recognised'
                    
                    
                order = meta.axes
                par_obj.order =order
                for n,b in enumerate(order):
                    if imno>0:
                        #TODO current asserts images are the same and takes smaller of T and Z, may want to support differing file sizes better
                        if b == 'T':
                            par_obj.total_time_pt = min(meta.shape[n],par_obj.total_time_pt)
                        if b == 'Z':
                            par_obj.max_zslices = min(meta.shape[n],par_obj.max_zslices )
                        if b == 'Y':
                            assert(par_obj.ori_height == meta.shape[n])
                        if b == 'X':
                            assert(par_obj.ori_width == meta.shape[n])
                        if b == 'S':
                            assert(par_obj.numCH == meta.shape[n])
                            #par_obj.tiff_reorder=False
                        if b == 'C':
                            assert(par_obj.numCH == meta.shape[n])
                    else:
                        if b == 'T':
                            par_obj.total_time_pt = meta.shape[n]
                        if b == 'Z':
                            par_obj.max_zslices = meta.shape[n]
                        if b == 'Y':
                            par_obj.ori_height = meta.shape[n]
                        if b == 'X':
                            par_obj.ori_width = meta.shape[n]
                        if b == 'S':
                            par_obj.numCH = meta.shape[n]
                            #par_obj.tiff_reorder=False
                        if b == 'C':
                            par_obj.numCH = meta.shape[n]
                            #par_obj.tiff_reorder=True
                if imno>0:
                    assert(par_obj.bitDepth == meta.dtype)
                par_obj.bitDepth = meta.dtype

                if par_obj.bitDepth in ['uint8','uint16','uint32']:
                    par_obj.tiffarray_typemax=np.iinfo(par_obj.bitDepth).max
                else:
                    par_obj.tiffarray_typemax=np.finfo(par_obj.bitDepth).max

                par_obj.tiffarray.append(par_obj.tiff_file.asarray(memmap=True))
                par_obj.tiffarraymax = max(par_obj.tiffarray[imno].max(),par_obj.tiffarraymax)

                imRGB = get_tiff_slice(par_obj,[0],[0],range(0,par_obj.ori_width,par_obj.resize_factor),range(0,par_obj.ori_height,par_obj.resize_factor),range(par_obj.numCH))
                #imRGB = par_obj.tiffarray[0,0,::par_obj.resize_factor,::par_obj.resize_factor]
#                if par_obj.tiff_reorder:
#                    imRGB =np.swapaxes(np.swapaxes(imRGB,0,1),1,2)
            elif par_obj.file_ext == 'oib'or  par_obj.file_ext == 'oif':
                par_obj.oib_file = OifFile(imStr)

                order = par_obj.oib_file.tiffs.axes
                for n,b in enumerate(order):
                    if b == 'T':
                        par_obj.total_time_pt = par_obj.oib_file.tiffs.shape[n]
                    if b == 'Z':
                        par_obj.max_zslices = par_obj.oib_file.tiffs.shape[n]
                    if b == 'C':
                        par_obj.numCH = par_obj.oib_file.tiffs.shape[n]
                #par_obj.oib_prefix = meta[2][0].split('/')[0]
                par_obj.oib_prefix = par_obj.oib_file.mainfile.name.split('.')[0]
                order  = meta[3]['axes']
                for n,b in enumerate(order):
                    if b == 'Y':
                        par_obj.ori_height = meta[3]['shape'][n]
                    if b == 'X':
                        par_obj.ori_width = meta[3]['shape'][n]
                
                par_obj.bitDepth = meta[3]['dtype']
                imRGB = np.zeros((par_obj.ori_height/par_obj.resize_factor,par_obj.ori_width/par_obj.resize_factor,3))
                for c in range(0,par_obj.numCH):
                    f  = par_obj.oib_prefix+'/s_C'+str(c+1).zfill(3)+'Z'+str(1).zfill(3)
                    if par_obj.total_time_pt>0:
                        f = f+'T'+str(par_obj.time_pt+1).zfill(3)
                    f = f+'.tif'
                    imRGB[:,:,c] = (par_obj.oib_file.asarray(f)[::int(par_obj.resize_factor),::int(par_obj.resize_factor)])/16
                
                
            elif par_obj.file_ext =='png':
                 
                 imRGB = pylab.imread(imStr)*255
                 par_obj.numCH =imRGB.shape.__len__()
                 par_obj.bitDepth = 8

                 if imRGB.shape[0] > par_obj.y_limit or imRGB.shape[1] > par_obj.x_limit:
                    statusText = 'Status: Your images are too large. Please reduce to less than 756x756.'
                    return False, statusText
                
            else:
                 statusText = 'Status: Image format not-recognised. Please choose either png or TIFF files.'
                 return False, statusText
            #pdb.set_trace()
            #Error Checking File Extension
            par_obj.prevExt = par_obj.file_ext
            #Error Checking number of cahnnels.
            if prevNumCH != [] and prevNumCH !=par_obj.numCH:
                statusText = 'More than one file format present. Different number of image channels in the selected images'
                return False, statusText
            prevNumCH  = par_obj.numCH
            #Error Checking Bit Depth.
            if prevBitDepth != [] and prevBitDepth != par_obj.bitDepth:
                statusText = 'More than one file format present. Different bit-depth in these different images'
                return False, statusText
            prevBitDepth = par_obj.bitDepth
            
    #Creates empty array to record density estimation.
    
    #par_obj.height = imRGB.shape[0]
    #par_obj.width = imRGB.shape[1]
    
    #par_obj.im_num_range = range(par_obj.test_im_start, par_obj.test_im_end)
    #par_obj.num_of_train_im = par_obj.test_im_end - par_obj.test_im_start
    
    if imRGB.shape.__len__() > 2:
    #If images have more than three channels. 
        if imRGB.shape[2]>2:
            #three channels.
            par_obj.ex_img = imRGB
        elif imRGB.shape[2]==2:
            #2 channels.
            par_obj.ex_img=np.zeros((par_obj.ori_height,par_obj.ori_width,3))
            par_obj.ex_img[:,:,0:2] = imRGB
        else:
            #If the size of the third dimenion is just 1, this is invalid for imshow show we have to adapt.
            par_obj.ex_img = imRGB[:,:,0]
    elif imRGB.shape.__len__() ==2:
        par_obj.ex_img = imRGB.astype('float32')

    statusText= str(file_array.__len__())+' Files Loaded.'
    return True, statusText
def save_output_prediction_fn(par_obj,int_obj):
    #funky ordering TZCYX

    for fileno in range(par_obj.max_file):
        image = np.zeros([par_obj.total_time_pt,par_obj.max_zslices,1,par_obj.height,par_obj.width], 'float32')
        for tpt in par_obj.time_pt_list:
            for zslice in range(0,par_obj.max_zslices):
                    image[tpt,zslice,0,:,:]=par_obj.data_store['pred_arr'][fileno][tpt][zslice].astype(np.float32)
        imsave(par_obj.csvPath+par_obj.file_name[fileno]+'_'+par_obj.modelName+'_Prediction.tif',image, imagej=True)
         
        print 'Prediction written to disk'
        int_obj.report_progress('Prediction written to disk '+ par_obj.csvPath)
def save_output_hess_fn(par_obj,int_obj):
    #funky ordering TZCYX

    for fileno in range(par_obj.max_file):
        image = np.zeros([par_obj.total_time_pt,par_obj.max_zslices,1,par_obj.height,par_obj.width], 'float32')
        for tpt in par_obj.time_pt_list:
            for zslice in range(0,par_obj.max_zslices):
                    image[tpt,zslice,0,:,:]=par_obj.data_store['maxi_arr'][fileno][tpt][zslice].astype(np.float32)
        print 'Prediction written to disk'
        imsave(par_obj.csvPath+par_obj.file_name[fileno]+'_'+par_obj.modelName+'_Hess.tif',image, imagej=True)
        int_obj.report_progress('Prediction written to disk '+ par_obj.csvPath)

def save_output_mask_fn(par_obj,int_obj):
    #funky ordering TZCYX
    for fileno in range(par_obj.max_file):
        image = np.zeros([par_obj.total_time_pt,par_obj.max_zslices,1,par_obj.height,par_obj.width], 'bool')
        for tpt in par_obj.time_pt_list:
            
            for i in range(0,par_obj.data_store['pts'][fileno][tpt].__len__()):
                [x,y,z]=par_obj.data_store['pts'][fileno][tpt][i]
                image[tpt,z,0,x,y]=255
        dist=list(par_obj.min_distance)
        selem=skimage.morphology.ball(np.round(dist[0],0)).astype('bool')
        if dist[2] is not 0:
            drange=range(selem.shape[0]/2,selem.shape[0],np.round(dist[0]/dist[2],0).astype('uint8'))
            lrange=range(0,selem.shape[0]/2,np.round(dist[0]/dist[2],0).astype('uint8'))
            selem2=selem[np.newaxis,lrange+drange,np.newaxis,:,:]
        image=scipy.ndimage.binary_dilation(image,selem2).astype('uint8')
        imsave(par_obj.csvPath+par_obj.file_name[fileno]+'_'+par_obj.modelName+'_Hess.tif',image, imagej=True)
        int_obj.report_progress('Prediction written to disk '+ par_obj.csvPath)
        
def save_output_data_fn(par_obj,int_obj):
    local_time = time.asctime( time.localtime(time.time()) )
    
    with open(par_obj.csvPath+str(par_obj.file_name[0])+'_'+par_obj.modelName+'outputData.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile,  dialect='excel')
        spamwriter.writerow([str(par_obj.selectedModel)]+[str('Filename: ')]+[str('Time point: ')]+[str('Predicted count: ')])

        for fileno in range(par_obj.max_file):
                
            for tpt in par_obj.time_pt_list:
                imStr = str(par_obj.file_array[fileno])
                spamwriter.writerow([local_time]+[str(imStr)]+[tpt+1]+[par_obj.data_store['pts'][fileno][tpt].__len__()])
                

    for fileno in range(par_obj.max_file):
        with open(par_obj.csvPath+str(par_obj.file_name[fileno])+'_'+par_obj.modelName+'_outputPoints.csv', 'a') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow([str(par_obj.selectedModel)]+[str('Filename: ')]+[str('Time point: ')]+[str('X: ')]+[str('Y: ')]+[str('Z: ')])

            for tpt in par_obj.time_pt_list:
                pts = par_obj.data_store['pts'][fileno][tpt]
                for i in range(0,par_obj.data_store['pts'][fileno][tpt].__len__()):
                    spamwriter.writerow([local_time]+[str(imStr)]+[tpt+1]+[pts[i][0]]+[pts[i][1]]+[pts[i][2]])
    int_obj.report_progress('Data exported to '+ par_obj.csvPath)

class ROI:
    
    def __init__(self, int_obj,par_obj):
        self.ppt_x = []
        self.ppt_y = []
        self.line = [None]
        self.int_obj = int_obj
        self.complete = False
        self.flag = False
        self.par_obj = par_obj
        
        
    def motion_notify_callback(self, event):
        #Mouse moving.
        if event.inaxes: 
            self.int_obj.plt1 = event.inaxes
            x, y = event.xdata, event.ydata
            if self.flag == True and event.button == 1: 
                i = self.flag_idx
                self.ppt_x[i] = x
                self.ppt_y[i] = y
                
                if i  == self.ppt_x.__len__()-1:
                    
                    self.line[0].set_data([self.ppt_x[i], self.ppt_x[0]],[self.ppt_y[i], self.ppt_y[0]])
                    self.line[i].set_data([self.ppt_x[i], self.ppt_x[i-1]],[self.ppt_y[i], self.ppt_y[i-1]])   
                else:
                    self.line[i+1].set_data([self.ppt_x[i], self.ppt_x[i+1]],[self.ppt_y[i], self.ppt_y[i+1]])
                    self.line[i].set_data([self.ppt_x[i], self.ppt_x[i-1]],[self.ppt_y[i], self.ppt_y[i-1]])   
                self.int_obj.canvas1.draw()
                self.par_obj.data_store['roi_stk_x'][imno][self.par_obj.time_pt][self.par_obj.curr_z] = copy.deepcopy(self.ppt_x)
                self.par_obj.data_store['roi_stk_y'][imno][self.par_obj.time_pt][self.par_obj.curr_z] = copy.deepcopy(self.ppt_y)
                #self.flag = False

        
    def button_press_callback(self, event):
        #Mouse clicking
        if event.inaxes: 
            x, y = event.xdata, event.ydata
            self.int_obj.plt1 = event.inaxes
            if event.button == 1:  # If you press the left button
                #Scan all to check if 
                if self.flag == False:
                    for i in range(0,self.ppt_x.__len__()):
                        if abs(x - self.ppt_x[i])<10 and abs(y - self.ppt_y[i])<10:
                            self.flag = True
                            self.flag_idx = i
                            break;
                            
                
                    
                    if self.flag == False and self.complete == False:
                        
                        if self.line[-1] == None: # if there is no line, create a line
                            self.line[0] = Line2D([x,  x],[y, y], marker = 'o')
                            self.ppt_x.append(x)
                            self.ppt_y.append(y) 
                            self.int_obj.plt1.add_line(self.line[0])
                            self.int_obj.canvas1.draw()
                        # add a segment
                        else: # if there is a line, create a segment
                            
                            self.line.append(Line2D([self.ppt_x[-1], x], [self.ppt_y[-1], y],marker = 'o'))
                            self.ppt_x.append(x)
                            self.ppt_y.append(y)
                            self.int_obj.plt1.add_line(self.line[-1])
                            self.int_obj.canvas1.draw()
                        self.par_obj.data_store['roi_stk_x'][self.par_obj.curr_file][self.par_obj.time_pt][self.par_obj.curr_z] = copy.deepcopy(self.ppt_x)
                        self.par_obj.data_store['roi_stk_y'][self.par_obj.curr_file][self.par_obj.time_pt][self.par_obj.curr_z] = copy.deepcopy(self.ppt_y)
                        
    def button_release_callback(self, event):
        self.flag = False
    def complete_roi(self):
        print 'ROI completed.'
        self.complete = True
        self.draw_ROI()
        self.reparse_ROI(self.par_obj.curr_z)
        #self.find_the_inside()
    def draw_ROI(self):
        #redraws the regions in the current slice.
        drawn = False
        imno=self.par_obj.curr_file
        tpt=self.par_obj.time_pt
        zslice=self.par_obj.curr_z
        for bt in self.par_obj.data_store['roi_stk_x'][imno][tpt]:
            if bt == zslice:
                cppt_x = self.par_obj.data_store['roi_stk_x'][imno][tpt][zslice]  
                cppt_y = self.par_obj.data_store['roi_stk_y'][imno][tpt][zslice]
                self.line = [None]
                for i in range(0,cppt_x.__len__()):
                    if i ==0:
                        self.line[0] = Line2D([cppt_x[0], cppt_x[0]], [cppt_y[0], cppt_y[0]], marker = 'o',color='red')
                        self.int_obj.plt1.add_line(self.line[-1])

                    elif i  < cppt_x.__len__():
                        self.line.append(Line2D([cppt_x[i-1], cppt_x[i]], [cppt_y[i-1], cppt_y[i]], marker = 'o',color='red'))
                        self.int_obj.plt1.add_line(self.line[-1])
                drawn = True
        if drawn == False:
                           
            for bt in self.par_obj.data_store['roi_stkint_x'][imno][tpt]:
                if bt == zslice:
                    cppt_x = self.par_obj.data_store['roi_stkint_x'][imno][tpt][zslice]
                    cppt_y = self.par_obj.data_store['roi_stkint_y'][imno][tpt][zslice]
                    
                    self.line = [None]
                    for i in range(0,cppt_x.__len__()):
                        if i == 0:
                            self.line[0] = Line2D([cppt_x[0], cppt_x[0]], [cppt_y[0], cppt_y[0]], color='red')
                            self.int_obj.plt1.add_line(self.line[-1])

                        elif i  < cppt_x.__len__():
                            self.line.append(Line2D([cppt_x[i-1], cppt_x[i]], [cppt_y[i-1], cppt_y[i]], color='red'))
                            self.int_obj.plt1.add_line(self.line[-1])
        



        self.int_obj.canvas1.draw()
    def reparse_ROI(self,im_num):
        #So that we can compare the ROI we resample them to have many more points.
        #This sounds straightforward but first we have to measure the distance between existing points.
        #And then we resample at specific intervals across the whole outline
        to_approve = []
        #Iterate the list of all the interpolations
        for bt in self.par_obj.data_store['roi_stkint_x'][self.par_obj.curr_file][self.par_obj.time_pt]:
            #If there is a matching hand-drawn one we 
            to_approve.append(bt)
                
        
        for cv in to_approve:
            del self.par_obj.data_store['roi_stkint_x'][self.par_obj.curr_file][self.par_obj.time_pt][cv]
            del self.par_obj.data_store['roi_stkint_y'][self.par_obj.curr_file][self.par_obj.time_pt][cv]

        for cd in self.par_obj.data_store['roi_stk_x'][self.par_obj.curr_file][self.par_obj.time_pt]:
                
            #First we make a local copy. We make a deep copy because python uses pointers and we don't want to change the original.
            cppt_x = copy.deepcopy(self.par_obj.data_store['roi_stk_x'][self.par_obj.curr_file][self.par_obj.time_pt][cd])
            cppt_y = copy.deepcopy(self.par_obj.data_store['roi_stk_y'][self.par_obj.curr_file][self.par_obj.time_pt][cd])

            dist = []
            #This is where we measure the distance between each of the defined points. 
            for i in range(0,cppt_y.__len__()-1):
                dist.append(np.sqrt((cppt_x[i]-cppt_x[i+1])**2+(cppt_y[i]-cppt_y[i+1])**2))
            dist.append(np.sqrt((cppt_x[i+1]-cppt_x[0])**2+(cppt_y[i+1]-cppt_y[0])**2))
            cmdist = [0]
            for i in range(0,dist.__len__()):
                cmdist.append(cmdist[-1]+dist[i])

            #We normalise the total distance to one in each case.
            cmdist = np.array(cmdist)
            cmdist = cmdist/cmdist[-1]

            #now we set the number of points. Should be a high number.
            npts = self.par_obj.npts
            pos = np.linspace(0,1,npts)
            
            cppt_x.append(cppt_x[0])
            cppt_y.append(cppt_y[0])
           
            nppt_x = []#[0]*npts
            nppt_y = []#[0]*npts

            #Now we interpolate between the points so that the structure is equally distributed with points.
            for i in range(0,npts):
                for b in range(0,cmdist.shape[0]-1):
                    ind0 = 0
                    ind1 = 1
                    if pos[i]>=cmdist[b] and pos[i]<=cmdist[b+1]:
                        ind0 = b
                        ind1 = b+1
                        break;
                
                pt0x = cppt_x[ind0]
                pt0y = cppt_y[ind0]
                pt1x = cppt_x[ind1]
                pt1y = cppt_y[ind1]


                l0 = pos[i] - cmdist[ind0]
                l1 = cmdist[ind1]-pos[i]
                
                #if(l0+l1) > 0.001:
                nppt_x.append(((pt0x*l1) + (pt1x*l0))/(l0+l1))
                nppt_y.append(((pt0y*l1) + (pt1y*l0))/(l0+l1))
            self.par_obj.data_store['roi_stkint_x'][self.par_obj.curr_file][self.par_obj.time_pt][cd] = nppt_x
            self.par_obj.data_store['roi_stkint_y'][self.par_obj.curr_file][self.par_obj.time_pt][cd] = nppt_y
            

        
        #self.int_obj.plt1.plot(nppt_x,nppt_y,'-')
        #self.int_obj.canvas1.draw()

    def interpolate_ROI(self):
        imno=self.par_obj.curr_file
        #We want to interpolate between frames. So we make sure the frames are in order.
        tosort = []
        for bt in self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.time_pt]:
            tosort.append(bt)
        sortd = np.sort(np.array(tosort))

        #For each of the slices which have been drawn in
        print 'sortd',sortd
        for b in range(0,sortd.shape[0]-1):
            ab = sortd[b]
            ac = sortd[b+1]
            #We assess if there are any slices which are empty within the range.
            if (ac-ab) > 1:
                #We then copy the points. (we use deepcopy because python uses pointers by defualt and we don't want to edit the original)
                lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.time_pt][ab])
                lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.time_pt][ab])
                upx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.time_pt][ac])
                upy = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.time_pt][ac])
                
                #Now we try and line up points which are closest.
                minfn1 = []
                for bx in range(0,lrx.__len__()):
                    lrx.append(lrx[0])
                    lry.append(lry[0])
                    lrx = lrx[1:]
                    lry = lry[1:]

                    ai = np.array(lrx)-np.array(upx)
                    bi = np.array(lry)-np.array(upy)
                    minfn1.append(np.sum(((ai**2)+(bi**2))**0.5))


                #The list can be reversed depending on howe the ROI was drawn (clockwise or anti-clockwise)
                lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.time_pt][ab])[::-1]
                lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.time_pt][ab])[::-1]

                minfn2 = []
                for bx in range(0,lrx.__len__()):
                    lrx.append(lrx[0])
                    lry.append(lry[0])
                    lrx = lrx[1:]
                    lry = lry[1:]

                    ai = np.array(lrx)-np.array(upx)
                    bi = np.array(lry)-np.array(upy)
                    minfn2.append(np.sum(((ai**2)+(bi**2))**0.5))
                

                #Now we take the minimum of the two.
                opt = np.argmin([np.min(np.array(minfn1)),np.min(np.array(minfn2))])
                

                #From this we find the global minima and set the lists accordingly
                if opt == 0:
                    min_dis = np.argmin(np.array(minfn1))
                    lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.time_pt][ab])
                    lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.time_pt][ab])
                else:  
                    min_dis = np.argmin(np.array(minfn2))
                    lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.time_pt][ab])[::-1]
                    lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.time_pt][ab])[::-1]

                lrx.extend(lrx[0:min_dis])
                lrx = lrx[min_dis:]
                lry.extend(lry[0:min_dis])
                lry = lry[min_dis:]

                #Finally we interpolate the ROI to provide a smooth transformation between the different frames.
                for b in range(ab+1, ac):
                    nppt_x = []
                    nppt_y = []
                    for i in range(0,lrx.__len__()):
                        
                        pt0x = lrx[i]
                        pt0y = lry[i]
                        pt1x = upx[i]
                        pt1y = upy[i]


                        l0 = ab - b
                        l1 = (b+1)-ac
                        
                        #if(l0+l1) > 0.001:
                        nppt_x.append(((pt0x*l1) + (pt1x*l0))/(l0+l1))
                        nppt_y.append(((pt0y*l1) + (pt1y*l0))/(l0+l1))  

                    #Then we save the results
                    self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.time_pt][b] = nppt_x
                    self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.time_pt][b] = nppt_y
                    
            self.int_obj.canvas1.draw()
    def interpolate_ROI_in_time(self):
        imno=self.par_obj.curr_file

        time_pts_with_roi = []
        for it_time_pt in self.par_obj.time_pt_list:
            if self.par_obj.data_store['roi_stkint_x'][imno][it_time_pt].__len__() > 0:
                time_pts_with_roi.append(it_time_pt)

        print 'time_pts_with_roi',time_pts_with_roi
        for it_pt in range(0,time_pts_with_roi.__len__()-1):
            tab = time_pts_with_roi[it_pt]
            tac = time_pts_with_roi[it_pt+1]
            first_pt_list = []
            secnd_pt_list = []
            #Check we need to interpolate. i.e. consecutive frames are more than one frame apart.
            if (tac-tab) > 1:
                #Find those frames with ROI
                for slc in self.par_obj.data_store['roi_stkint_x'][imno][tab]:
                    first_pt_list.append(slc)
                for slc in self.par_obj.data_store['roi_stkint_x'][imno][tac]:
                    secnd_pt_list.append(slc)
                mtc_list = list(set(first_pt_list) & set(secnd_pt_list))

                if mtc_list.__len__() > 0:
                    for it_zslice in mtc_list:
                        lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][tab][it_zslice])
                        lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][tab][it_zslice])
                        upx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][tac][it_zslice])
                        upy = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][tac][it_zslice])

                        #Now we try and line up points which are closest.
                        minfn1 = []
                        for bx in range(0,lrx.__len__()):
                            lrx.append(lrx[0])
                            lry.append(lry[0])
                            lrx = lrx[1:]
                            lry = lry[1:]

                            ai = np.array(lrx)-np.array(upx)
                            bi = np.array(lry)-np.array(upy)
                            minfn1.append(np.sum(((ai**2)+(bi**2))**0.5))


                        #The list can be reversed depending on howe the ROI was drawn (clockwise or anti-clockwise)
                        lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][tab][it_zslice])[::-1]
                        lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][tab][it_zslice])[::-1]

                        minfn2 = []
                        for bx in range(0,lrx.__len__()):
                            lrx.append(lrx[0])
                            lry.append(lry[0])
                            lrx = lrx[1:]
                            lry = lry[1:]

                            ai = np.array(lrx)-np.array(upx)
                            bi = np.array(lry)-np.array(upy)
                            minfn2.append(np.sum(((ai**2)+(bi**2))**0.5))
                        

                        #Now we take the minimum of the two.
                        opt = np.argmin([np.min(np.array(minfn1)),np.min(np.array(minfn2))])
                        

                        #From this we find the global minima and set the lists accordingly
                        if opt == 0:
                            min_dis = np.argmin(np.array(minfn1))
                            lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][tab][it_zslice])
                            lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][tab][it_zslice])
                        else:  
                            min_dis = np.argmin(np.array(minfn2))
                            lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][tab][it_zslice])[::-1]
                            lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][tab][it_zslice])[::-1]

                        lrx.extend(lrx[0:min_dis])
                        lrx = lrx[min_dis:]
                        lry.extend(lry[0:min_dis])
                        lry = lry[min_dis:]

                        for b in range(tab+1, tac): #timepoints
                                nppt_x = []
                                nppt_y = []
                                
                                for i in range(0,lrx.__len__()):
                                    
                                    pt0x = lrx[i]
                                    pt0y = lry[i]
                                    pt1x = upx[i]
                                    pt1y = upy[i]


                                    l0 = b-tab
                                    l1 = tac-b
                                    
                                    #if(l0+l1) > 0.001:
                                    nppt_x.append(((pt0x*l1) + (pt1x*l0))/(l0+l1))
                                    nppt_y.append(((pt0y*l1) + (pt1y*l0))/(l0+l1))  

                                #Then we save the results
                                self.par_obj.data_store['roi_stkint_x'][imno][b][it_zslice] =nppt_x
                                self.par_obj.data_store['roi_stkint_y'][imno][b][it_zslice] =nppt_y






    def find_the_inside(self):
            ppt_x = self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.time_pt][self.par_obj.curr_z]
            ppt_y = self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.time_pt][self.par_obj.curr_z]
            imRGB = np.array(self.par_obj.dv_file.get_frame(par_obj.curr_z))
            
            pot = [] 
            for i in range(0,ppt_x.__len__()):
                pot.append([ppt_x[i],ppt_y[i]])

            p = Path(pot)
            for y in range(0,imRGB.shape[0]):
                for x in range(0,imRGB.shape[1]):
                    if p.contains_point([x,y]) == True:
                        imRGB[y,x] = 255
                        
                    
            self.int_obj.plt1.imshow(newImg/par_obj.tiffarraymax)
            self.int_obj.canvas1.draw()

#from __future__ import division # Sets division to be float division
class DV_Controller:
    def __init__(self,im_str):
        f = open(im_str)
        self.dvdata = f.read()
        f.close()
         
        dvExtendedHeaderSize = struct.unpack_from("<I", self.dvdata, 92)[0]
         
        # endian-ness test
        if not struct.unpack_from("<H", self.dvdata, 96)[0] == 0xc0a0:
            print "unsupported endian-ness"
            return
         
        dvImageWidth=struct.unpack_from("<I", self.dvdata, 0)[0]
        dvImageHeight=struct.unpack_from("<I", self.dvdata, 4)[0]
        dvNumOfImages=struct.unpack_from("<I", self.dvdata, 8)[0]
        dvPixelType=struct.unpack_from("<I", self.dvdata, 12)[0]

        
        dvImageDataOffset=1024+dvExtendedHeaderSize
        rawSizeT = struct.unpack_from("<H", self.dvdata, 180)[0]
        if rawSizeT == 0:
            timepoints = 1
        else:
            timepoints = rawSizeT
        sequence =  struct.unpack_from("<H", self.dvdata, 182)[0]
        extSize  = struct.unpack_from("<I", self.dvdata, 92)[0]
        rawSizeC= struct.unpack_from("<H", self.dvdata, 196)[0]
        if rawSizeC == 0:
            num_channels = 1
        else:
            num_channels = rawSizeC
         
        dvExtendedHeaderNumInts=struct.unpack_from("<H", self.dvdata, 128)[0]
        dvExtendedHeaderNumFloats=struct.unpack_from("<H", self.dvdata, 130)[0]
        sectionSize = 4*(dvExtendedHeaderNumFloats+dvExtendedHeaderNumInts)
        sections = dvExtendedHeaderSize/sectionSize
        if (sections < dvNumOfImages):
            print "number of sections is less than the number of images"
            return
        self.maxFrames = dvNumOfImages
        self.numCH = rawSizeC
        self.num_of_tp = timepoints
        self.bitDepth = dvPixelType

        
        #elapsed_times = [[struct.unpack_from("<f", dvdata, i*sectionSize+k*4)[0] for k in range(sectionSize/4)][25] for i in range(sections)]

         
        #elapsed_times = [strftimefloat(s) for s in elapsed_times]
         
        self.offset = dvImageDataOffset
        self.size = dvImageWidth*dvImageHeight*4
        self.width = dvImageWidth
        self.height = dvImageHeight
    def get_frame(self,j):
        st = (j*self.size)+self.offset
        en = st+self.size
            
        im = np.frombuffer(self.dvdata[st:en], dtype=np.dtype(np.float32))
        
        return im.reshape(self.height,self.width)

def reset_parameters(par_obj):
    par_obj.frames_2_load ={}
    par_obj.left_2_calc =[]
    par_obj.saved_ROI =[]
    par_obj.saved_dots=[]
def processImgs(self,par_obj):
    """Loads images and calculates the features."""
    #Resets everything should this be another patch of images loaded.

    par_obj.height = par_obj.ori_height/par_obj.resize_factor
    par_obj.width = par_obj.ori_width/par_obj.resize_factor

    par_obj.initiate_data_store()
    par_obj.time_pt = par_obj.time_pt_list[0]
    par_obj.curr_z = par_obj.frames_2_load[0]
    par_obj.curr_file = 0
    '''
    #Now we commit our options to our imported files.
    if par_obj.file_ext == 'png':
        for i in range(0,par_obj.file_array.__len__()):
            par_obj.left_2_calc.append(i)
            par_obj.frames_2_load[i] = [0]
        self.image_status_text.showMessage('Status: Loading Images. Loading Image Num: '+str(par_obj.file_array.__len__()))

        ###v2.im_pred_inline_fn(par_obj, self)
    elif par_obj.file_ext =='tiff' or par_obj.file_ext =='tif':
            for i in range(0,par_obj.file_array.__len__()):
                par_obj.left_2_calc.append(i)

            self.image_status_text.showMessage('Status: Loading Images.')
            #v2.im_pred_inline_fn(par_obj, self)
            ###v2.im_pred_inline_fn_eval(par_obj, self,threaded=True) commenting these because calculation of features is now considered by later functions

    elif par_obj.file_ext =='oib'or  par_obj.file_ext == 'oif':
        if par_obj.test_im_end>1:
            for i in range(0,par_obj.file_array.__len__()):
                par_obj.left_2_calc.append(i)
            self.image_status_text.showMessage('Status: Loading Images.')
            ###v2.im_pred_inline_fn(par_obj, self)

        else:
            for i in range(0,par_obj.file_array.__len__()):
                par_obj.left_2_calc.append(i)
                par_obj.frames_2_load[i] = [0]
            v2.im_pred_inline_fn(par_obj, self)
    '''



          
        
