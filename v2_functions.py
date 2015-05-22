from PyQt4 import QtGui, QtCore, Qt
import PIL.Image
import numpy as np
import os
import vigra
import pylab
import csv
import time
from sklearn.ensemble import ExtraTreesRegressor
import cPickle as pickle
from scipy.ndimage import filters
from oiffile import OifFile
import itertools as itt
import struct
import copy
from matplotlib.lines import Line2D
from matplotlib.path import Path
"""QuantiFly3d Software v0.0

    Copyright (C) 2015  Dominic Waithe

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

def peak_local_max(image, min_distance=10, threshold_abs=0, threshold_rel=0.1,
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
    # In the case of labels, recursively build and return an output
    # operating on each label separately
    """
    if labels is not None:
        label_values = np.unique(labels)
        # Reorder label values to have consecutive integers (no gaps)
        if np.any(np.diff(label_values) != 1):
            mask = labels >= 1
            labels[mask] = 1 + rank_order(labels[mask])[0].astype(labels.dtype)
        labels = labels.astype(np.int32)

        # New values for new ordering
        label_values = np.unique(labels)
        for label in label_values[label_values != 0]:
            maskim = (labels == label)
            out += peak_local_max(image * maskim, min_distance=min_distance,
                                  threshold_abs=threshold_abs,
                                  threshold_rel=threshold_rel,
                                  exclude_border=exclude_border,
                                  indices=False, num_peaks=np.inf,
                                  footprint=footprint, labels=None,overlap=overlap)

        if indices is True:
            return np.transpose(out.nonzero())
        else:
            return out.astype(np.bool)
            """



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
import numpy as np


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

def calculateCI(data1,data2, test_value):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    
    x = np.array(data1)
    y = np.array(data2)

    z = np.polyfit(x,y,1)
    p = np.poly1d(z)
    fit = p(x)

    # get the coordinates for the fit curve    
    c_y = [np.min(fit),np.max(fit)]
    c_x = [np.min(x),np.max(x)]

    # predict y values of origional data using the fit
    p_y = z[0] * x + z[1] 

    # calculate the y-error (residuals)
    y_err = y -p_y 

    #Take the input prediction, convert to error, so is in format of x.
    #p_xx = (p_x*z[0])+z[1]
    
    # now calculate confidence intervals for new test x-series
    mean_x = np.mean(x)         # mean of x
    n = len(x)              # number of samples in origional fit
    t = 2.31                # appropriate t value (where n=9, two tailed 95%)
    s_err = np.sum(np.power(y_err,2))   # sum of the squares of the residuals

    #This calculates the error for one point, pxx
    
    confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((( test_value)+mean_x),2)/ ((np.sum(np.power(x,2)))-n*(np.power(mean_x,2))))))
    
    #We then modulate the point we the calculated correction +/- the Confidence interval.
    p_up =  test_value +abs(confs)
    p_dn =  test_value -abs(confs)
    

    return p_up,p_dn

def apply_correction(par_obj,withGT=False):
    par_obj.CC ={}
    par_obj.CC1 ={}
    par_obj.CC2 ={}
    par_obj.lowerCI ={}
    par_obj.upperCI ={}
        
    for i in range(par_obj.test_im_start,par_obj.test_im_end):
        test_value = ((par_obj.sum_pred[i]*par_obj.M)+par_obj.c)
        p_dn,p_up  = calculateCI(par_obj.gt_vec, par_obj.error_vec, test_value);
    
        
        
        par_obj.CC[i] = par_obj.sum_pred[i]-((par_obj.sum_pred[i]*par_obj.M)+par_obj.c)
        par_obj.lowerCI[i] = abs(par_obj.sum_pred[i]-p_dn)-par_obj.CC[i]
        par_obj.upperCI[i] = abs(par_obj.sum_pred[i]-p_up)-par_obj.CC[i]
        
        print('corrected value: '+str(par_obj.CC[i]))
        if withGT == True:
            par_obj.CC_absErr = np.abs(par_obj.CC[i]-par_obj.gt_sum[i])
            par_obj.CC_perErr = (np.abs(par_obj.CC[i]-par_obj.gt_sum[i])*100)/par_obj.gt_sum[i]
            print('Ground Truth count: '+str(par_obj.gt_sum[i]))
            print('corrected Absolute ERROR: '+str(np.abs(par_obj.CC[i]-par_obj.gt_sum[i])))
            print('corrected Percentage ERROR: '+str((np.abs(par_obj.CC[i]-par_obj.gt_sum[i])*100)/par_obj.gt_sum[i]))

def make_correction(par_obj,model_num,withGT=False):
    #Makes linear correction to model data.
    par_obj.error_vec =[];
    pred_vec=[];
    par_obj.gt_vec =[];
   
    

    for b in range(0,par_obj.saved_ROI.__len__()):
        #Iterates through saved ROI.
        rects = par_obj.saved_ROI[b]
        par_obj.gt_vec.append(np.sum(par_obj.dense_array[rects[0]][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3]])/255)
        pred_vec.append(np.sum(par_obj.pred_arr[rects[0]][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3]])/255)
        par_obj.error_vec.append(pred_vec[-1]-par_obj.gt_vec[-1])
        
    #Fits straight-line to data.
    M, c = np.polyfit(par_obj.gt_vec, par_obj.error_vec,1)
    par_obj.M = M
    par_obj.c = c
    apply_correction(par_obj,withGT)


    
    


    
    


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
        par_obj.rects = (par_obj.curr_img, int(s_ori_x), int(s_ori_y), int(abs(par_obj.rect_w)), int(abs(par_obj.rect_h)))
        return True
    
    return False
    

def update_training_samples_fn(par_obj,model_num):
    """Collects the pixels or patches which will be used for training and 
    trains the forest."""
    #Makes sure everything is refreshed for the training, encase any regions
    #were changed. May have to be rethinked for speed later on.
    par_obj.f_matrix =[]
    par_obj.o_patches=[]
    region_size = 0
    for b in range(0,par_obj.saved_ROI.__len__()):
        rects = par_obj.saved_ROI[b]
        region_size += rects[4]*rects[3]        
    
    calc_ratio = par_obj.limit_ratio_size
    
    #print 'calcratio',calc_ratio
    #print 'aftercratio',region_size/par_obj.limit_ratio_size

    for b in range(0,par_obj.saved_ROI.__len__()):

        #Iterates through saved ROI.
        rects = par_obj.saved_ROI[b]
        img2load = rects[0]



        #Loads necessary images only.
        try:
            par_obj.feat_arr[img2load]
        except:
            im_pred_inline_fn(par_obj,par_obj,True,img2load,0,img2load-1)

        if(par_obj.p_size == 1):
            #Finds and extracts the features and output density for the specific regions.
            mImRegion = par_obj.feat_arr[rects[0]][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3],:]
            denseRegion = par_obj.dense_array[rects[0]][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3]]
            #Find the linear form of the selected feature representation
            mimg_lin = np.reshape(mImRegion, (mImRegion.shape[0]*mImRegion.shape[1],mImRegion.shape[2]))
            #Find the linear form of the complementatory output region.
            dense_lin = np.reshape(denseRegion, (denseRegion.shape[0]*denseRegion.shape[1]))
            #Sample the input pixels sparsely or densely.
            if(par_obj.limit_sample == True):
                if(par_obj.limit_ratio == True):
                    par_obj.limit_size = round(mImRegion.shape[0]*mImRegion.shape[1]/calc_ratio,0)
                #Randomly sample from input ROI or im a certain number (par_obj.limit_size) patches. With replacement.
                indices =  np.random.choice(int(mImRegion.shape[0]*mImRegion.shape[1]), size=int(par_obj.limit_size), replace=True, p=None)
                #Add to feature vector and output vector.
                par_obj.f_matrix.extend(mimg_lin[indices])
                par_obj.o_patches.extend(dense_lin[indices])
            else:
                #Add these to the end of the feature Matrix, input patches
                par_obj.f_matrix.extend(mimg_lin)
                #And the the output matrix, output patches
                par_obj.o_patches.extend(dense_lin)
        if(par_obj.p_size >1):
            mgn = (win.p_size-1)/2
            #Finds the corresponding image.
            left_rect = rects[2]+1 -mgn
            right_rect = rects[2]+rects[4] +mgn+1
            top_rect = rects[1]+1 -mgn
            bot_rect = rects[1]+rects[3]+mgn+1
            if left_rect < 0:
                left_rect = 0
            if top_rect < 0:
                top_rect = 0
            if right_rect > par_obj.width - 1:
                right_rect = par_obj.width - 1
            if bot_rect > par_obj.height - 1:
                bot_rect = par_obj.height - 1
            mImRegion = win.par_obj.feat_arr[rects[0]][rects[2]+1:rects[2]+rects[4],rects[1]+1:rects[1]+rects[3]]

            mImRegion = win.par_obj.feat_arr[rects[0]][left_rect:right_rect,top_rect:bot_rect,:]
            denseRegion = win.dense_array[rects[0]][left_rect:right_rect,top_rect:bot_rect]

            mimg_linPatch,dense_linPatch, pos = v2.extractPatch(win.p_size,mImRegion,denseRegion, 'sparse')
            
            win.f_matrix.extend(mimg_linPatch)
            win.o_patches.extend(dense_linPatch)
    #Sets up extra trees regressor object.
    
    par_obj.RF[model_num] = ExtraTreesRegressor(par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1)
    


    #Fits the data.
    t3 = time.time()
    print 'fmatrix',np.array(par_obj.f_matrix).shape
    print 'o_patches',np.array(par_obj.o_patches).shape
    par_obj.RF[model_num].fit(np.asfortranarray(par_obj.f_matrix), np.asfortranarray(par_obj.o_patches))
    t4 = time.time()
    print 'actual training',t4-t3 
def update_density_fn(par_obj):
   
    for im in par_obj.im_for_train:
        #Construct empty array for current image.
        dots_im = np.zeros((par_obj.height,par_obj.width))
        #In array of all saved dots.

        for i in range(0,par_obj.saved_dots.__len__()):
            #Any ROI in the present image.
            #print 'iiiii',win.saved_dots.__len__()
            if(par_obj.saved_ROI[i][0] == im):
                #Save the corresponding dots.
                dots = par_obj.saved_dots[i]
                #Scan through the dots
                for b in range(0,dots.__len__()):
                   
                    #save the column and row 
                    c_dot = dots[b][2]
                    r_dot = dots[b][1]
                    #Set it to register as dot.
                    dots_im[c_dot, r_dot] = 255
        #Convolve the dots to represent density estimation.
        dense_im = filters.gaussian_filter(dots_im.astype(np.float32),   float(par_obj.sigma_data), order=0, output=None, mode='reflect', cval=0.0)
        #Replace member of dense_array with new image.
        
        par_obj.dense_array[im] = dense_im

def im_pred_inline_fn(par_obj, int_obj,inline=False,outer_loop=None,inner_loop=None,count=None):
    """Accesses TIFF file slice or opens png. Calculates features to indices present in par_obj.left_2_calc"""
    if inline == False:
        outer_loop = par_obj.left_2_calc
        inner_loop_arr = par_obj.frames_2_load
        count = -1
    else:
        #par_obj.feat_arr ={}
        inner_loop_arr ={outer_loop:[inner_loop]}
        outer_loop = [outer_loop]


    #Goes through the list of files.
    for b in outer_loop:
            
            imStr = str(par_obj.file_array[b])
            frames = inner_loop_arr[b]
            if par_obj.file_ext == 'tif' or par_obj.file_ext == 'tiff':
                temp = Tiff_Controller(imStr)
            if par_obj.file_ext == 'oib':
                par_obj.oib_file = OifFile(imStr).asarray()[:,:,::int(par_obj.resize_factor),::int(par_obj.resize_factor)]


            for i in frames:
                count = count+1
                if par_obj.file_ext == 'tif' or par_obj.file_ext == 'tiff':
                    
                    imRGB = temp.get_frame(i)
                elif par_obj.file_ext == 'png':
                    imRGB = pylab.imread(str(imStr))*255
                elif par_obj.file_ext == 'oib':
                    par_obj.height = par_obj.oib_file.shape[2]
                    par_obj.width = par_obj.oib_file.shape[3]
                    imRGB = np.zeros((int(par_obj.height),int(par_obj.width),par_obj.ch_active.__len__()))
                    for c in range(0,par_obj.ch_active.__len__()):
                        imRGB[:,:,c] = par_obj.oib_file[c,i,:,:]
                if par_obj.fresh_features == False:
                    try:
                        #Try loading features.
                        time1 = time.time()
                        feat = pickle.load(open(imStr[:-4]+'_'+str(i)+'.p', "rb"))
                        time2 = time.time()
                        int_obj.report_progress('Loading Features for Image: '+str(b+1)+' Frame: ' +str(i+1))
            
                    except:
                        #If don't exist create them.
                        int_obj.report_progress('Calculating Features for Image: '+str(b+1)+' Frame: ' +str(i+1))
                        feat = feature_create(par_obj,imRGB,imStr,i)
                else:
                    #If you want to ignore previous features which have been saved.
                    int_obj.report_progress('Calculating Features for Image: '+str(b+1)+' Frame: ' +str(i+1))
                    feat =feature_create(par_obj,imRGB,imStr,i)
                par_obj.num_of_feat = feat.shape[2]
                print 'thecountersays',count
                par_obj.feat_arr[count] = feat  
    
    return
def feature_create(par_obj,imRGB,imStr,i):
    time1 = time.time()
    if par_obj.crop_x2 ==0 and par_obj.crop_x1 ==0:
            par_obj.crop_x1 = 0
            par_obj.crop_x2=imRGB.shape[1]
            par_obj.crop_y1 = 0
            par_obj.crop_y2=imRGB.shape[0]
    par_obj.height = par_obj.crop_y2-par_obj.crop_y1
    par_obj.width = par_obj.crop_x2-par_obj.crop_x1 
    
    if (par_obj.feature_type == 'basic'):
        feat = np.zeros(((int(par_obj.crop_y2)-int(par_obj.crop_y1)),(int(par_obj.crop_x2)-int(par_obj.crop_x1)),13*par_obj.ch_active.__len__()))
    if (par_obj.feature_type == 'fine'):
        feat = np.zeros(((int(par_obj.crop_y2)-int(par_obj.crop_y1)),(int(par_obj.crop_x2)-int(par_obj.crop_x1)),21*par_obj.ch_active.__len__()))
    if (par_obj.feature_type == 'fineSpatial'):
        feat = np.zeros(((int(par_obj.crop_y2)-int(par_obj.crop_y1)),(int(par_obj.crop_x2)-int(par_obj.crop_x1)),23*par_obj.ch_active.__len__()))
    
    for b in range(0,par_obj.ch_active.__len__()):
        if (par_obj.feature_type == 'basic'):
            imG = imRGB[:,:,par_obj.ch_active[b]].astype(np.float32)
            feat[:,:,(b*13):((b+1)*13)] = local_shape_features_basic(imG,par_obj.feature_scale)   
        if (par_obj.feature_type == 'fine'):
            imG = imRGB[:,:,par_obj.ch_active[b]].astype(np.float32)
            feat[:,:,(b*21):((b+1)*21)] = local_shape_features_fine(imG,par_obj.feature_scale)
        if (par_obj.feature_type == 'fineSpatial'):
            imG = imRGB[:,:,par_obj.ch_active[b]].astype(np.float32)
            feat[:,:,(b*23):((b+1)*23)] = local_shape_features_fine_spatial(imG,par_obj.feature_scale,i)
    #pickle.dump(feat,open(imStr[:-4]+'_'+str(i)+'.p', "wb"),protocol=2)
    return feat
    
def evaluate_forest(par_obj,int_obj,withGT,model_num,inline=False,inner_loop=None,outer_loop=None,count=None):
    if inline == False:
        outer_loop = par_obj.left_2_calc
        inner_loop_arr = par_obj.frames_2_load
        count = -1
    else:
        inner_loop_arr ={outer_loop:[inner_loop]}
        outer_loop = [outer_loop]

    #Finds the current frame and file.
    
    for b in outer_loop:
        frames =inner_loop_arr[b]
        for i in frames:
            count = count+1
            
            

            if(par_obj.p_size >1):
                
                mimg_lin,dense_linPatch, pos = extractPatch(par_obj.p_size, par_obj.feat_arr[count], None, 'dense')
                tree_pred = par_obj.RF[model_num].predict(mimg_lin)
                linPred = v2.regenerateImg(par_obj.p_size, tree_pred, pos)
                    
            else:
                print 'the count is:',count
                mimg_lin = np.reshape(par_obj.feat_arr[count], (par_obj.height * par_obj.width, par_obj.feat_arr[count].shape[2]))
                t2 = time.time()
                linPred = par_obj.RF[model_num].predict(mimg_lin)
                t1 = time.time()
                


            par_obj.pred_arr[count] = linPred.reshape(par_obj.height, par_obj.width)

            maxPred = np.max(linPred)
            sum_pred =np.sum(linPred/255)
            par_obj.sum_pred[count] = sum_pred
            print 'prediction time taken',t1 - t2
            print 'Predicted count:',par_obj.sum_pred[count]
            int_obj.report_progress('Making Prediction for Image: '+str(b+1)+' Frame: ' +str(i+1))
                    

            if withGT == True:
                try:
                    #If it has already been opened.
                    a = par_obj.gt_sum[count]
                except:
                    #Else find the file.
                    gt_im =  pylab.imread(par_obj.gt_array[count])[:,:,0]
                    par_obj.gt_sum[count] = np.sum(gt_im)
                
                
                print('Ground Truth count: '+str(par_obj.gt_sum[count]))
                print('Absolute ERROR: '+str(np.abs(par_obj.sum_pred[count]-par_obj.gt_sum[count])))
                print('Percentage ERROR: '+str((np.abs(par_obj.sum_pred[count]-par_obj.gt_sum[count])*100)/par_obj.gt_sum[count]))
            

            

    
def regenerate_img(p_size,tree_pred,pos):
    outImg = np.zeros((evalImWin.par_obj.feat_arr[0].shape[0],evalImWin.par_obj.feat_arr[0].shape[1]))
    mgn = int((p_size-1)/2)
    norm = np.zeros((evalImWin.par_obj.feat_arr[0].shape[0],evalImWin.par_obj.feat_arr[0].shape[1]))
    zerot = np.ones((p_size,p_size))
    

    for i in range(0,tree_pred.shape[0]):
        y_pos = pos[i][0]
        x_pos = pos[i][1]
       
        outImg[y_pos-mgn:mgn+y_pos+1,x_pos-mgn:mgn+x_pos+1]  += np.array(tree_pred[i].reshape(p_size,p_size))
        norm[y_pos-mgn:mgn+y_pos+1,x_pos-mgn:mgn+x_pos+1]  += zerot
    

    
    ind2div = outImg > 0
    outImg[ind2div]= outImg[ind2div]*(1/norm[ind2div])
    
    return outImg


def extract_patch(p_size,mImRegion,denseRegion,sample):
    
    #patch margin.
    mgn = int((p_size-1)/2)
    
    #Active areas given patch margin.
    subImRegion =mImRegion[mgn:mImRegion.shape[0]-mgn-1,mgn:mImRegion.shape[1]-mgn-1,:]
    
    if denseRegion !=None:
        subDenseRegion = denseRegion[mgn:mImRegion.shape[0]-mgn-1,mgn:mImRegion.shape[1]-mgn-1]
    

    #Create meshgrid for quick index to position reference
    yR = np.arange(0,mImRegion.shape[0])
    xR = np.arange(0,mImRegion.shape[1])
    xvFull,yvFull = np.meshgrid(xR, yR)
    yv = yvFull[mgn:mImRegion.shape[0]-mgn-1,mgn:mImRegion.shape[1]-mgn-1]
    xv = xvFull[mgn:mImRegion.shape[0]-mgn-1,mgn:mImRegion.shape[1]-mgn-1]
    

    xvLin = xv.reshape(-1)
    yvLin = yv.reshape(-1)
    totalLocations = yv.shape[0]*yv.shape[1]
    
    if sample == 'sparse':
        #Samples non-densely.
        
        limit_size = np.floor(np.array((totalLocations/win.limit_ratio_size))).astype(np.int32)
        indices =  np.random.choice(totalLocations, size=limit_size, replace=True, p=None)
    elif(sample == 'dense'):
        indices = np.arange(0,totalLocations)

    #output containers
    mimgRegPatch =[]
    denseRegPatch = []
    pos =[]
    for i in range(0,indices.shape[0]):
        x_pos = xvLin[indices[i]]
        y_pos = yvLin[indices[i]]
        m_patch = mImRegion[y_pos-mgn:mgn+y_pos+1,x_pos-mgn:mgn+x_pos+1,:]
        if denseRegion !=None:
            dPatch = denseRegion[y_pos-mgn:mgn+y_pos+1,x_pos-mgn:mgn+x_pos+1]

        #if m_patch.shape[0]==2 and m_patch.shape[1]==2:
        mimgRegPatch.append(m_patch.reshape(-1))
        pos.append((y_pos,x_pos))
        if denseRegion !=None:
            denseRegPatch.append(dPatch.reshape(-1))
                
    
    return mimgRegPatch, denseRegPatch, pos


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
    f[:,:, 1]  = vigra.filters.gaussianGradientMagnitude(im, s)
    f[:,:, 2]  = st08[:,:,0]
    f[:,:, 3]  = st08[:,:,1]
    f[:,:, 4]  = vigra.filters.laplacianOfGaussian(im, s )
    f[:,:, 5]  = vigra.filters.gaussianGradientMagnitude(im, s*2) 
    f[:,:, 6]  =  st16[:,:,0]
    f[:,:, 7]  = st16[:,:,1]
    f[:,:, 8]  = vigra.filters.laplacianOfGaussian(im, s*2 )
    f[:,:, 9]  = vigra.filters.gaussianGradientMagnitude(im, s*4) 
    f[:,:, 10] =  st32[:,:,0]
    f[:,:, 11] =  st32[:,:,1]
    f[:,:, 12] = vigra.filters.laplacianOfGaussian(im, s*4 )
    f[:,:, 13]  = vigra.filters.gaussianGradientMagnitude(im, s*8) 
    f[:,:, 14] =  st64[:,:,0]
    f[:,:, 15] =  st64[:,:,1]
    f[:,:, 16] = vigra.filters.laplacianOfGaussian(im, s*8 )
    f[:,:, 17]  = vigra.filters.gaussianGradientMagnitude(im, s*16) 
    f[:,:, 18] =  st128[:,:,0]
    f[:,:, 19] =  st128[:,:,1]
    f[:,:, 20] = vigra.filters.laplacianOfGaussian(im, s*16 )
   
    
    
    return f
def local_shape_features_fine_spatial(im,scaleStart,im_num):
    #Exactly as in the Luca Fiaschi paper.
    s = scaleStart
    
    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC,imSizeR,23))

    xv, yv = np.meshgrid(range(0,imSizeC), range(0,imSizeR))
    
    st08 = vigra.filters.structureTensorEigenvalues(im,s*1,s*2)
    st16 = vigra.filters.structureTensorEigenvalues(im,s*2,s*4)
    st32 = vigra.filters.structureTensorEigenvalues(im,s*4,s*8)
    st64 = vigra.filters.structureTensorEigenvalues(im,s*8,s*16)
    st128 = vigra.filters.structureTensorEigenvalues(im,s*16,s*32)
    
    f[:,:, 0]  = np.ones((imSizeC,imSizeR))*im_num
    f[:,:, 1]  = vigra.filters.gaussianGradientMagnitude(im, s)
    f[:,:, 2]  = st08[:,:,0]
    f[:,:, 3]  = st08[:,:,1]
    f[:,:, 4]  = vigra.filters.laplacianOfGaussian(im, s )
    f[:,:, 5]  = vigra.filters.gaussianGradientMagnitude(im, s*2) 
    f[:,:, 6]  =  st16[:,:,0]
    f[:,:, 7]  = st16[:,:,1]
    f[:,:, 8]  = vigra.filters.laplacianOfGaussian(im, s*2 )
    f[:,:, 9]  = vigra.filters.gaussianGradientMagnitude(im, s*4) 
    f[:,:, 10] =  st32[:,:,0]
    f[:,:, 11] =  st32[:,:,1]
    f[:,:, 12] = vigra.filters.laplacianOfGaussian(im, s*4 )
    f[:,:, 13] = vigra.filters.gaussianGradientMagnitude(im, s*8) 
    f[:,:, 14] =  st64[:,:,0]
    f[:,:, 15] =  st64[:,:,1]
    f[:,:, 16] = vigra.filters.laplacianOfGaussian(im, s*8 )
    f[:,:, 17] = vigra.filters.gaussianGradientMagnitude(im, s*16) 
    f[:,:, 18] =  st128[:,:,0]
    f[:,:, 19] =  st128[:,:,1]
    f[:,:, 20] = vigra.filters.laplacianOfGaussian(im, s*16 )
    f[:,:, 21] = xv
    f[:,:, 22] = yv

   
    
    
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
    f[:,:, 1]  = vigra.filters.gaussianGradientMagnitude(im, s)
    f[:,:, 2]  = st08[:,:,0]
    f[:,:, 3]  = st08[:,:,1]
    f[:,:, 4]  = vigra.filters.laplacianOfGaussian(im, s )
    f[:,:, 5]  = vigra.filters.gaussianGradientMagnitude(im, s*2) 
    f[:,:, 6]  =  st16[:,:,0]
    f[:,:, 7]  = st16[:,:,1]
    f[:,:, 8]  = vigra.filters.laplacianOfGaussian(im, s*2 )
    f[:,:, 9]  = vigra.filters.gaussianGradientMagnitude(im, s*4) 
    f[:,:, 10] =  st32[:,:,0]
    f[:,:, 11] =  st32[:,:,1]
    f[:,:, 12] = vigra.filters.laplacianOfGaussian(im, s*4 )
    
    
    
    return f

def eval_goto_img_fn(im_num, par_obj, int_obj):
    """Loads up and converts image to correct format"""

    #Finds the current frame and file.
    count = -1
    for b in par_obj.left_2_calc:
        frames =par_obj.frames_2_load[b]
        for i in frames:
            count = count+1
            if par_obj.curr_img == count:
                break;
        else:
            continue 
        break 
    

    
    if ( par_obj.file_ext == 'png'):
        imStr = str(par_obj.file_array[b])
        imRGB = pylab.imread(imStr)*255
    if ( par_obj.file_ext == 'tif' or  par_obj.file_ext == 'tiff'):
        imStr = str(par_obj.file_array[b])
        temp = Tiff_Controller(imStr)
        imRGB = temp.get_frame(i)
    if ( par_obj.file_ext == 'oib'):
        print 'current index', i
        imRGB = np.zeros((int(par_obj.height),int(par_obj.width),par_obj.ch_active.__len__()))
        for c in range(0,par_obj.ch_active.__len__()):
            imRGB[:,:,c] = 0.99- (par_obj.oib_file[c,i,:,:].astype(np.float64)/np.max(par_obj.oib_file[c,i,:,:].astype(np.float64)))
        
    count = 0
    CH = [0]*par_obj.numCH
    for c in range(0,par_obj.numCH):
        name = 'a = int_obj.CH_cbx'+str(c)+'.checkState()'
        exec(name)
        if a ==2:
            count = count + 1
            CH[c] = 1

    newImg =np.zeros((par_obj.height,par_obj.width,3))
    if count == 1:
        ch = par_obj.ch_active[0]
        if imRGB.shape> 2:
            newImg[:,:,0] = imRGB[:,:,ch]
            newImg[:,:,1] = imRGB[:,:,ch]
            newImg[:,:,2] = imRGB[:,:,ch]
        else:
            newImg[:,:,0] = imRGB
            newImg[:,:,1] = imRGB
            newImg[:,:,2] = imRGB
    else:
        if CH[0] == 1:
            newImg[:,:,0] = imRGB[:,:,0]
        if CH[1] == 1:
            newImg[:,:,1] = imRGB[:,:,1]
        if CH[2] == 1:
            newImg[:,:,2] = imRGB[:,:,2]

    
    par_obj.save_im = imRGB
    for i in range(0,int_obj.plt1.lines.__len__()):
        int_obj.plt1.lines.pop(0)
    par_obj.newImg = newImg
    int_obj.plt1.cla()
    int_obj.plt1.imshow(254-newImg)
    int_obj.plt1.set_xlim([0,newImg.shape[0]])
    int_obj.plt1.set_ylim([newImg.shape[1],0])
    int_obj.draw_saved_dots_and_roi()
    int_obj.plt1.set_xticklabels([])
    int_obj.plt1.set_yticklabels([])
    int_obj.cursor.draw_ROI()
    int_obj.canvas1.draw()
    #del im
    
    
    int_obj.image_num_txt.setText('The Current image is No. ' + str(par_obj.curr_img+1)) # filename: ' +str(evalLoadImWin.file_array[im_num]))
    eval_pred_show_fn(im_num,par_obj,int_obj)

def eval_pred_show_fn(im_num,par_obj,int_obj):
    """Shows Prediction Image when forest is loaded"""
    if par_obj.eval_load_im_win_eval == True:
        print 'okyeah',str(im_num)
        int_obj.image_num_txt.setText('The Current Image is No. ' + str(par_obj.curr_img+1))

        if int_obj.count_maxima_plot_on.isChecked() == True:
            par_obj.show_pts = True
        else:
            par_obj.show_pts = False

        int_obj.plt2.cla()


        if par_obj.show_pts == True:
            
            pt_x = []
            pt_y = []
            for pt2d in par_obj.pts:
                if pt2d[2] == par_obj.curr_img:


                    

                        pt_x.append(pt2d[1])
                        pt_y.append(pt2d[0])


            int_obj.plt1.plot(pt_x,pt_y, 'go')
            int_obj.plt2.plot(pt_x,pt_y, 'go')
            string_2_show = 'The Predicted Count: ' + str(par_obj.pts.__len__())
        #if par_obj.upperCI[im_num] < 1000:
        #    string_2_show += ' with bias correction: '+str(round(par_obj.CC[im_num],1))+' +\- CI '+str(np.round(par_obj.upperCI[im_num],2))+''
            int_obj.output_count_txt.setText(string_2_show)
            int_obj.plt2.imshow(par_obj.maxi_arr[im_num].astype(np.float32))
        else:
            int_obj.plt2.imshow(par_obj.pred_arr[im_num].astype(np.float32))
        


        int_obj.plt2.set_xticklabels([])
        int_obj.plt2.set_yticklabels([])
        int_obj.canvas1.draw()
        int_obj.canvas2.draw()
        
 
def import_data_fn(par_obj,file_array):
    """Function which loads in Tiff stack or single png file to assess type."""
    prevExt = [] 
    prevBitDepth=[] 
    prevNumCH =[]
    for i in range(0,file_array.__len__()):
            n = str(i)
            imStr = str(file_array[i])
            par_obj.file_ext = imStr.split(".")[-1]
            
            if prevExt != [] and prevExt !=par_obj.file_ext:
                statusText = 'More than one file format present. Different number of image channels in the selected images'
                return False, statusText


            if par_obj.file_ext == 'tif' or par_obj.file_ext == 'tiff':
                par_obj.tiff_file = Tiff_Controller(imStr)
                par_obj.numCH = par_obj.tiff_file.im_sz[2]
                par_obj.bitDepth = par_obj.tiff_file.im.tag[0x102][0]

                if par_obj.tiff_file.im_sz[0] > par_obj.y_limit or par_obj.tiff_file.im_sz[1] > par_obj.x_limit:
                    statusText = 'Status: Your images are too large. Please reduce to less than 756x756.'
                    return False, statusText
                if par_obj.tiff_file.maxFrames > 8:
                    par_obj.uploadLimit = 8
                else:
                    par_obj.uploadLimit = par_obj.tiff_file.maxFrames


                 
                
                par_obj.test_im_end = par_obj.tiff_file.maxFrames
                imRGB = par_obj.tiff_file.get_frame(0)
            elif par_obj.file_ext == 'oib':
                par_obj.oib_file = OifFile(imStr).asarray()
                par_obj.numCH = par_obj.oib_file.shape[0]
                

                par_obj.bitDepth = 16
                imRGB = par_obj.oib_file[0,0,:,:]
                par_obj.test_im_end = par_obj.oib_file.shape[1]
                if par_obj.oib_file.shape[1] > 8:
                    par_obj.uploadLimit = 8
                else:
                    par_obj.uploadLimit = par_obj.oib_file.shape[1]
                
            elif par_obj.file_ext =='png':
                 
                 imRGB = pylab.imread(imStr)*255
                 par_obj.test_im_end = file_array.__len__()
                 par_obj.numCH =imRGB.shape.__len__()
                 par_obj.bitDepth = 8

                 if imRGB.shape[0] > par_obj.y_limit or imRGB.shape[1] > par_obj.x_limit:
                    statusText = 'Status: Your images are too large. Please reduce to less than 756x756.'
                    return False, statusText
                
            else:
                 statusText = 'Status: Image format not-recognised. Please choose either png or TIFF files.'
                 return False, statusText

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
    
    par_obj.test_im_start = 0
    par_obj.height = imRGB.shape[0]
    par_obj.width = imRGB.shape[1]
    par_obj.im_num_range = range(par_obj.test_im_start, par_obj.test_im_end)
    par_obj.num_of_train_im = par_obj.test_im_end - par_obj.test_im_start
    
    
    if imRGB.shape.__len__() > 2:
    #If images have more than three channels. 
        if imRGB.shape[2]>1:
            #If the shape of the third dimension is greater than 2.
            par_obj.ex_img = imRGB[:,:,:]
        else:
            #If the size of the third dimenion is just 1, this is invalid for imshow show we have to adapt.
            par_obj.ex_img = imRGB[:,:,0]
    elif imRGB.shape.__len__() ==2:
        par_obj.ex_img = imRGB[:,:]
    

    
    statusText= str(file_array.__len__())+' Files Loaded.'
    return True, statusText

def save_output_data_fn(par_obj,int_obj):
    local_time = time.asctime( time.localtime(time.time()) )

    with open(par_obj.csvPath+'outputData.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([str(par_obj.selectedModel)]+[str('Filename: ')]+[str('Frame number: ')]+[str('Predicted count: ')]+[str('Corrected count: ')]+[str('CI ')])
    
    count = -1
    for b in par_obj.left_2_calc:
        frames =par_obj.frames_2_load[b]
        imStr = str(par_obj.file_array[count])
        for i in frames:
            count = count+1
            n = str(count)
            string = par_obj.csvPath+'output' + n.zfill(3)+'.tif'
            im_to_save= PIL.Image.fromarray(par_obj.pred_arr[count].astype(np.float32))
            im_to_save.save(string)
               
            with open(par_obj.csvPath+'outputData.csv', 'a') as csvfile:
                spamwriter = csv.writer(csvfile,  dialect='excel')
                spamwriter.writerow([local_time]+[str(imStr)]+['-'+str(i)]+[par_obj.sum_pred[count]]+[par_obj.CC[count]]+[par_obj.upperCI[count]])
                

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
                print self.ppt_x.__len__()
                print 'line',self.line.__len__()
                self.ppt_x[i] = x
                self.ppt_y[i] = y
                
                if i  == self.ppt_x.__len__()-1:
                    
                    self.line[0].set_data([self.ppt_x[i], self.ppt_x[0]],[self.ppt_y[i], self.ppt_y[0]])
                    self.line[i].set_data([self.ppt_x[i], self.ppt_x[i-1]],[self.ppt_y[i], self.ppt_y[i-1]])   
                else:
                    self.line[i+1].set_data([self.ppt_x[i], self.ppt_x[i+1]],[self.ppt_y[i], self.ppt_y[i+1]])
                    self.line[i].set_data([self.ppt_x[i], self.ppt_x[i-1]],[self.ppt_y[i], self.ppt_y[i-1]])   
                self.int_obj.canvas1.draw()
                self.par_obj.roi_stk_x[self.par_obj.curr_img] = copy.deepcopy(self.ppt_x)
                self.par_obj.roi_stk_y[self.par_obj.curr_img] = copy.deepcopy(self.ppt_y)
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
                            print 'here ',i
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
                        self.par_obj.roi_stk_x[self.par_obj.curr_img] = copy.deepcopy(self.ppt_x)
                        self.par_obj.roi_stk_y[self.par_obj.curr_img] = copy.deepcopy(self.ppt_y)
    def button_release_callback(self, event):
        self.flag = False
    def complete_roi(self):
        print 'ROI completed.'
        self.complete = True
        self.draw_ROI()
        self.reparse_ROI(self.par_obj.curr_img)
        #self.find_the_inside()
    def draw_ROI(self):
        #redraws the regions in the current slice.
        drawn = False
        for bt in self.par_obj.roi_stk_x:
            if bt == self.par_obj.curr_img:
                cppt_x = self.par_obj.roi_stk_x[self.par_obj.curr_img]  
                cppt_y = self.par_obj.roi_stk_y[self.par_obj.curr_img]
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
            for bt in self.par_obj.roi_stkint_x:
                if bt == self.par_obj.curr_img:
                    cppt_x = self.par_obj.roi_stkint_x[self.par_obj.curr_img]  
                    cppt_y = self.par_obj.roi_stkint_y[self.par_obj.curr_img]
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
        for bt in self.par_obj.roi_stkint_x:
            #If there is a matching hand-drawn one we 
            to_approve.append(bt)
                
        
        for cv in to_approve:
            del self.par_obj.roi_stkint_x[cv]
            del self.par_obj.roi_stkint_y[cv]

        for cd in self.par_obj.roi_stk_x:
                
            #First we make a local copy. We make a deep copy because python uses pointers and we don't want to change the original.
            cppt_x = copy.deepcopy(self.par_obj.roi_stk_x[cd])
            cppt_y = copy.deepcopy(self.par_obj.roi_stk_y[cd])

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
            self.par_obj.roi_stkint_x[cd]  = nppt_x
            self.par_obj.roi_stkint_y[cd]  = nppt_y

        
        #self.int_obj.plt1.plot(nppt_x,nppt_y,'-')
        #self.int_obj.canvas1.draw()

    def interpolate_ROI(self):
        
        

        #We want to interpolate between frames. So we make sure the frames are in order.
        tosort = []
        for bt in self.par_obj.roi_stkint_x:
            tosort.append(bt)
        sortd = np.sort(np.array(tosort))

        #For each of the slices which have been drawn in
        for b in range(0,sortd.shape[0]-1):
            ab = sortd[b]
            ac = sortd[b+1]
            #We assess if there are any slices which are empty within the range.
            if (ac-ab) > 1:
                #We then copy the points. (we use deepcopy because python uses pointers by defualt and we don't want to edit the original)
                lrx = copy.deepcopy(self.par_obj.roi_stkint_x[ab])
                lry = copy.deepcopy(self.par_obj.roi_stkint_y[ab])
                upx = copy.deepcopy(self.par_obj.roi_stkint_x[ac])
                upy = copy.deepcopy(self.par_obj.roi_stkint_y[ac])
                
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
                lrx = copy.deepcopy(self.par_obj.roi_stkint_x[ab])[::-1]
                lry = copy.deepcopy(self.par_obj.roi_stkint_y[ab])[::-1]

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
                    lrx = copy.deepcopy(self.par_obj.roi_stkint_x[ab])
                    lry = copy.deepcopy(self.par_obj.roi_stkint_y[ab])
                else:  
                    min_dis = np.argmin(np.array(minfn2))
                    lrx = copy.deepcopy(self.par_obj.roi_stkint_x[ab])[::-1]
                    lry = copy.deepcopy(self.par_obj.roi_stkint_y[ab])[::-1]

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
                    self.par_obj.roi_stkint_x[b] = nppt_x
                    self.par_obj.roi_stkint_y[b] = nppt_y
                    
            self.int_obj.canvas1.draw()
    def find_the_inside(self):
            ppt_x =    self.par_obj.roi_stkint_x[self.par_obj.curr_img]
            ppt_y  =   self.par_obj.roi_stkint_y[self.par_obj.curr_img]
            imRGB = np.array(self.par_obj.dv_file.get_frame(par_obj.curr_img))
            
            pot = [] 
            for i in range(0,ppt_x.__len__()):
                pot.append([ppt_x[i],ppt_y[i]])

            p = Path(pot)
            for y in range(0,imRGB.shape[0]):
                for x in range(0,imRGB.shape[1]):
                    if p.contains_point([x,y]) == True:
                        imRGB[y,x] = 255
                        
                    
            self.int_obj.plt1.imshow(imRGB)
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

        print 'self.maxFrames', self.maxFrames
        print 'self.numCH', self.numCH
        print 'self.num_of_tp', self.num_of_tp
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


          
        

    

class Tiff_Controller:
    def __init__(self,fname):
        '''fname is the full path '''
        self.im  = PIL.Image.open(fname)
        self.fname = fname
        self.im.seek(0)
        self.im_sz = [self.im.tag[0x101][0], self.im.tag[0x100][0],self.im.tag[0x102].__len__()]
        self.cur = self.im.tell()
        num = 0
        self.maxFrames =1
        while True:
            num = num+1
            try:
                self.im.seek(num)
            except EOFError:
                return None
            self.maxFrames = num
    def get_frame(self,j):
        '''Extracts the jth frame from the image sequence.
        if the frame does not exist return None'''
        try:
            self.im.seek(j)
        except EOFError:
            return None

        self.cur = self.im.tell()
        return np.reshape(self.im.getdata(),self.im_sz)
    def __iter__(self):
        self.im.seek(0)
        self.old = self.cur
        self.cur = self.im.tell()
        return self

    def next(self):
        try:
            self.im.seek(self.cur)
            self.cur = self.im.tell()+1
        except EOFError:
            self.im.seek(self.old)
            self.cur = self.im.tell()
            raise StopIteration
        return np.reshape(self.im.getdata(),self.im_sz)    