"""CytoCensus Software v0.1

    Copyright (C) 2016-2018  Dominic Waithe Martin Hailstone

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
from __future__ import division
#from PyQt4 import QtGui, QtCore, Qt
from PyQt5 import QtCore
#import PIL.Image
import os
import csv
import time
import pickle
from multiprocessing.dummy import Pool as ThreadPool

import functools
import itertools as itt
import copy
import numpy as np


import skimage
import scipy as sp

from sklearn import ensemble

from scipy.ndimage import filters, measurements
from tifffile import imsave #Install with pip install tifffile.

from matplotlib.path import Path
import matplotlib.image as pylab
from features import local_features as lf
from fileio.file_handler import File_handler
from functions.maxima import count_maxima



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
        par_obj.rects = (par_obj.curr_z, int(s_ori_x), int(s_ori_y), int(abs(par_obj.rect_w)), \
        int(abs(par_obj.rect_h)), par_obj.curr_t, \
        par_obj.curr_file, par_obj.file_array[par_obj.curr_file])
        return True

    return False

def stratified_sample(binlength, samples_indices, imhist, samples_at_tiers, denseRegion):
    #preallocate array rather than extend, size corrects for rounding errors
    indices = np.zeros(samples_indices[-1], 'uint32')
    #Randomly sample from input ROI or im a certain number (par_obj.limit_size) patches.
    #With replacement.
    for it in range(binlength):
        if samples_at_tiers[it] > 0:
            bin1 = imhist[1][it]
            bin2 = imhist[1][it+1]

            stratified_indices = np.nonzero(((denseRegion >= bin1) & (denseRegion < bin2)).flat)
            if stratified_indices[0].__len__() != 0:
                stratified_sampled_indices = np.random.choice(stratified_indices[0], size=samples_at_tiers[it], replace=True, p=None)
            else:
                stratified_sampled_indices = []
            indices[range(samples_indices[it], samples_indices[it+1])] = stratified_sampled_indices
    return indices

def update_training_samples_fn_new_only(par_obj, int_obj, rects, arr='feat_arr'):
    """Collects the pixels or patches which will be used for training and
    trains the forest."""
    #Makes sure everything is refreshed for the training, encase any regions
    #were changed. May have to be rethinked for speed later on.
    '''
    for b in range(0,par_obj.saved_ROI.__len__()):
        rects = par_obj.saved_ROI[b]
        region_size += rects[4]*rects[3]
    '''
    calc_ratio = par_obj.limit_ratio_size

    STRATIFY = False
    sigma = par_obj.sigma_data
    dot_im = np.pad(np.ones((1, 1)), (int(sigma)*6, int(sigma)*4), mode='constant')
    dot_im = filters.gaussian_filter(dot_im, float(par_obj.sigma_data), mode='constant', cval=0)
    dot_im /= dot_im.max()
    binlength = 10
    imhist = np.histogram(dot_im, bins=binlength, range=(0, 1), density=True)
    imhist[1][binlength] = 5 # adjust top bin to make sure we include everthing if we have overlapping gaussians-try to avoid though-if very common will distort bins
    samples_at_tiers = (imhist[0]/binlength*par_obj.limit_size).astype('int')
    #print samples_at_tiers
    samples_indices = [0]+(np.cumsum(samples_at_tiers)).tolist() #to allow preallocation of array


    #for b in range(0,par_obj.saved_ROI.__len__()):
        #TODO check this works for edge cases- with very sparse sampling, and with v small bin sizes

    zslice = rects[0]
    tpt = rects[5]
    imno = rects[6]
    if par_obj.p_size == 1:

        #if rects[5] == tpt and rects[0] == zslice and rects[6] == imno:

        #Finds and extracts the features and output density for the specific regions.
        im_region = par_obj.data_store[arr][imno][tpt][zslice][rects[2]+1:rects[2]+rects[4], rects[1]+1:rects[1]+rects[3], :]
        p_region = par_obj.data_store['dense_arr'][imno][tpt][zslice][rects[2]+1:rects[2]+rects[4], rects[1]+1:rects[1]+rects[3]]
        #Find the linear form of the selected feature representation
        mimg_lin = np.reshape(im_region, (im_region.shape[0]*im_region.shape[1], im_region.shape[2]))
        #Find the linear form of the complementatory output region.
        dense_lin = np.reshape(p_region, (p_region.shape[0]*p_region.shape[1]))
        #Sample the input pixels sparsely or densely.
        if par_obj.limit_sample is True:

            if par_obj.limit_ratio is True:

                par_obj.limit_size = round(im_region.shape[0]*im_region.shape[1]/calc_ratio, 0)

                if STRATIFY is True:
                    indices = stratified_sample(binlength, samples_indices, imhist, samples_at_tiers, p_region)
                else:
                    indices = np.random.choice(int(im_region.shape[0]*im_region.shape[1]), size=int(par_obj.limit_size), replace=True, p=None)
            else:
                #this works because the first n indices refer to the full first two dimensions, and np indexing takes slices
                indices = np.random.choice(int(im_region.shape[0]*im_region.shape[1]), size=int(par_obj.limit_size), replace=True, p=None)
            #Add to feature vector and output vector.
            par_obj.f_matrix.extend(mimg_lin[indices])
            par_obj.o_patches.extend(dense_lin[indices])
        else:
            #Add these to the end of the feature Matrix, input patches
            par_obj.f_matrix.extend(mimg_lin)
            #And the the output matrix, output patches
            par_obj.o_patches.extend(dense_lin)

    if par_obj.p_size == 2:

        im_region = par_obj.data_store[arr][imno][tpt][zslice][rects[2]+1:rects[2]+rects[4], rects[1]+1:rects[1]+rects[3], :]
        par_obj.f_matrix.append(im_region)
        p_region = par_obj.data_store['dense_arr'][imno][tpt][zslice][rects[2]+1:rects[2]+rects[4], rects[1]+1:rects[1]+rects[3]]
        par_obj.o_patches.append(p_region)

def update_training_samples_fn_auto(par_obj, int_obj, rects):
    """Collects the pixels or patches which will be used for training and
    trains the forest."""
    #Makes sure everything is refreshed for the training, encase any regions
    #were changed. May have to be rethinked for speed later on.
    '''
    for b in range(0,par_obj.saved_ROI.__len__()):
        rects = par_obj.saved_ROI[b]
        region_size += rects[4]*rects[3]
    '''
    calc_ratio = par_obj.limit_ratio_size
    #print 'calcratio',calc_ratio
    STRATIFY = False
    dot_im = np.pad(np.ones((1, 1)), (int(par_obj.sigma_data)*6, int(par_obj.sigma_data)*4), mode='constant')
    dot_im = filters.gaussian_filter(dot_im, float(par_obj.sigma_data), mode='constant', cval=0)
    dot_im /= dot_im.max()
    binlength = 10
    imhist = np.histogram(dot_im, bins=binlength, range=(0, 1), density=True)
    # adjust top bin to make sure we include everthing if we have overlapping gaussians-try to avoid though-if very common will distort bins
    imhist[1][binlength] = 5
    samples_at_tiers = (imhist[0]/binlength*par_obj.limit_size).astype('int')
    #print samples_at_tiers
    samples_indices = [0]+(np.cumsum(samples_at_tiers)).tolist() #to allow preallocation of array


    #for b in range(0,par_obj.saved_ROI.__len__()):
        #TODO check this works for edge cases- with very sparse sampling, and with v small bin sizes
        #Iterates through saved ROI.
        #rects = par_obj.saved_ROI[b]

    zslice = rects[0]
    tpt = rects[5]
    imno = rects[6]
    if par_obj.p_size == 1:
        #if rects[5] == tpt and rects[0] == zslice and rects[6] == imno:
        #Finds and extracts the features and output density for the specific regions.
        mImRegion = par_obj.data_store['double_feat_arr'][imno][tpt][zslice][rects[2]+1:rects[2]+rects[4], rects[1]+1:rects[1]+rects[3], :]
        denseRegion = par_obj.data_store['dense_arr'][imno][tpt][zslice][rects[2]+1:rects[2]+rects[4], rects[1]+1:rects[1]+rects[3]]
        #Find the linear form of the selected feature representation
        mimg_lin = np.reshape(mImRegion, (mImRegion.shape[0]*mImRegion.shape[1], mImRegion.shape[2]))
        #Find the linear form of the complementatory output region.
        dense_lin = np.reshape(denseRegion, (denseRegion.shape[0]*denseRegion.shape[1]))
        #Sample the input pixels sparsely or densely.
        if par_obj.limit_sample is True:
            if par_obj.limit_ratio is True:
                par_obj.limit_size = round(mImRegion.shape[0]*mImRegion.shape[1]/calc_ratio, 0)

                if STRATIFY is True:
                    indices = stratified_sample(binlength, samples_indices, imhist, samples_at_tiers, denseRegion)
                else:
                    indices = np.random.choice(int(mImRegion.shape[0]*mImRegion.shape[1]), size=int(par_obj.limit_size), replace=True, p=None)
            else:
                #this works because the first n indices refer to the full first two dimensions, and np indexing takes slices
                indices = np.random.choice(int(mImRegion.shape[0]*mImRegion.shape[1]), size=int(par_obj.limit_size), replace=True, p=None)
            #Add to feature vector and output vector.
            par_obj.f_matrix.extend(mimg_lin[indices])
            par_obj.o_patches.extend(dense_lin[indices])
        else:
            #Add these to the end of the feature Matrix, input patches
            par_obj.f_matrix.extend(mimg_lin)
            #And the the output matrix, output patches
            par_obj.o_patches.extend(dense_lin)

def train_forest(par_obj, int_obj, model_num):

    #if par_obj.max_features > par_obj.num_of_feat[model_num]:
    par_obj.max_features = int(par_obj.num_of_feat[model_num]/3)

    #assigns model type
    par_obj.RF[model_num] = lf.RF(par_obj, 'ETR')

    #Fits the data.
    X = np.array(par_obj.f_matrix)
    Y = np.array(par_obj.o_patches)
    par_obj.RF[model_num].fit(X, Y)
    
    
    """
    importances = par_obj.RF[model_num].method.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(len(importances)):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    t4 = time.time()
    print 'actual training',t4-t3
    """
    par_obj.o_patches = []
    par_obj.f_matrix = []

def refresh_all_density(par_obj):
    number_of_saved_roi = range(0, len(par_obj.saved_ROI))
    for it in number_of_saved_roi:
        tpt = int(par_obj.saved_ROI[it][5])
        zslice = int(par_obj.saved_ROI[it][0])
        fileno = int(par_obj.saved_ROI[it][6])
        update_com_fn(par_obj, tpt, zslice, fileno)

def update_com_fn(par_obj, tpt, zslice, fileno):

    #Construct empty array for current image.
    dots_im = np.zeros((par_obj.height, par_obj.width))
    #In array of all saved dots.
    for i in range(0, par_obj.saved_dots.__len__()):
        #Any ROI in the present image.
        #print 'iiiii',win.saved_dots.__len__()

        if(par_obj.saved_ROI[i][0] == zslice and par_obj.saved_ROI[i][5] == tpt and par_obj.saved_ROI[i][6] == fileno):
            #Save the corresponding dots.
            dots = par_obj.saved_dots[i]
            #Scan through the dots
            for b in range(0, dots.__len__()):

                #save the column and row
                c_dot = dots[b][2]
                r_dot = dots[b][1]
                #Set it to register as dot.
                dots_im[c_dot, r_dot] = 1 #change from 255
    #Convolve the dots to represent density estimation.
    #print 'Using template matching to generate C-O-M representation'
    dense_im = np.zeros(dots_im.shape).astype(np.float64)

    size_of_kernel = np.round(np.ceil(par_obj.sigma_data * 5)).astype('int') #At least the 3-sigma rule.
    if size_of_kernel % 2 == 0:
        size_of_kernel = int(size_of_kernel + 1)

    patch = np.zeros((size_of_kernel, size_of_kernel))
    m_p = int((size_of_kernel-1)/2)
    patch[m_p, m_p] = 1

    kernel = filters.gaussian_filter(patch.astype(np.float32), float(par_obj.sigma_data), order=0, output=None, mode='constant', cval=0.0)

    #Replace member of dense_array with new image.
    r_arr, c_arr = np.where(dots_im > 0)
    for r, c in zip(r_arr, c_arr):
        p1 = 0
        p2 = patch.shape[0]
        p3 = 0
        p4 = patch.shape[1]

        r1 = int(r-m_p)
        r2 = int(r+m_p+1)
        c1 = int(c-m_p)
        c2 = int(c+m_p+1)

        if r1 < 0:
            p1 = abs(r1-0)
            r1 = 0
        if r2 > dots_im.shape[0]:
            p2 = (patch.shape[0]-1)- (abs(r2-dots_im.shape[0]))+1
            r2 = dots_im.shape[0]
        if c1 < 0:
            p3 = abs(c1-0)
            c1 = 0
        if c2 > dots_im.shape[1]:
            p4 = (patch.shape[1]-1)- (abs(c2-dots_im.shape[1]))+1
            c2 = dots_im.shape[1]



        dense_im[r1:r2, c1:c2] = np.max([kernel[p1:p2, p3:p4], dense_im[r1:r2, c1:c2]], 0)

    '''NORMALISE GAUSSIANS. THIS MAKES IT USELESS FOR DOING 2D DENSITY ESTIMATION,
 but super useful if you want a probability output between 0 and 1 at the end of the day
for thresholding and the like'''
    dense_im = dense_im/kernel.max()*256
    par_obj.gaussian_im_max = kernel.max()

    par_obj.data_store['dense_arr'][fileno][tpt][zslice] = dense_im

def im_pred_inline_fn_new(par_obj, int_obj, zsliceList, tptList, imnoList, threaded=False):
    """Accesses TIFF file slice (from open tiffarray. Calculates features to slices specified"""
    # consider cropping
    if par_obj.to_crop is False:
        par_obj.crop_x1 = 0
        par_obj.crop_x2 = par_obj.width
        par_obj.crop_y1 = 0
        par_obj.crop_y2 = par_obj.height
    par_obj.height = par_obj.crop_y2-par_obj.crop_y1
    par_obj.width = par_obj.crop_x2-par_obj.crop_x1

    if par_obj.FORCE_nothreading is not False:
        threaded = par_obj.FORCE_nothreading

    if threaded == 0:

        for imno in imnoList:
            for tpt in tptList:
                for zslice in zsliceList:
                    #checks if features already in array

                    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:
                        #imRGB=return_rgb_slice(par_obj,zslice,tpt,imno)
                        #imRGB/=par_obj.tiffarraymax
                        width = range(0, par_obj.ori_width, int(par_obj.resize_factor))
                        height = range(0, par_obj.ori_height, int(par_obj.resize_factor))
                        imRGB = par_obj.filehandlers[imno].get_tiff_slice([tpt], [zslice], width, height, par_obj.ch_active)

                        #imRGBlist.append(imRGB)
                        imRGB = imRGB.astype('float32')/ par_obj.tiffarray_typemax
                        #If you want to ignore previous features which have been saved.
                        int_obj.report_progress('Calculating Features for File:'+str(imno+1)+ ' Timepoint: '+str(tpt+1) +' Z: '+str(zslice+1))

                        feat = lf.feature_create_threadable(par_obj, imRGB)

                        par_obj.num_of_feat[0] = feat.shape[2]
                        par_obj.data_store['feat_arr'][imno][tpt][zslice] = feat

    elif threaded == 'auto':
        #threaded version
        imRGBlist = []
        for tpt in tptList:
            for imno in imnoList:
                featlist = []
                tee1 = time.time()
                pool = ThreadPool(8)
                featlist = pool.map(functools.partial(lf.feature_create_threadable_auto, par_obj, imno, tpt), zsliceList)
                pool.close()
                pool.join()
                tee2 = time.time()
                #feat =feature_create(par_obj,imRGB,imStr,i)
                print (tee2-tee1)
                lcount = -1

                for zslice in zsliceList:
                    lcount = lcount+1
                    feat = featlist[lcount]
                    int_obj.report_progress('Calculating Features for File:'+str(imno+1)+ ' Timepoint: '+str(tpt+1) +' Z: '+str(zslice+1))

                    par_obj.num_of_feat[1] = feat.shape[2]+par_obj.num_of_feat[0]

                    par_obj.data_store['double_feat_arr'][imno][tpt][zslice] = np.concatenate((feat, par_obj.data_store['feat_arr'][imno][tpt][zslice]), axis=2)

    else:
        #threaded version
        imRGBlist = []
        for tpt in tptList:
            for imno in imnoList:
                for zslice in zsliceList:
                    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:

                        width = range(0, par_obj.ori_width, int(par_obj.resize_factor))
                        height = range(0, par_obj.ori_height, int(par_obj.resize_factor))
                        imRGB = par_obj.filehandlers[imno].get_tiff_slice([tpt], [zslice], width, height, par_obj.ch_active)
                        imRGBlist.append(imRGB.astype('float32')/ par_obj.tiffarray_typemax)
                #initiate pool and start caclulating features
                int_obj.report_progress('Calculating Features for File:'+str(imno+1)+ ' Timepoint: '+str(tpt+1) +' Z: '+'ALL')
                featlist = []
                tee1 = time.time()
                pool = ThreadPool(8)
                featlist = pool.map(functools.partial(lf.feature_create_threadable, par_obj), imRGBlist)
                pool.close()
                pool.join()
                tee2 = time.time()
                #feat =feature_create(par_obj,imRGB,imStr,i)
                print (tee2-tee1)
                lcount = -1

                for zslice in zsliceList:
                    if zslice not in par_obj.data_store['feat_arr'][imno][tpt]:
                        lcount = lcount+1
                        feat = featlist[lcount]
                        int_obj.report_progress('Calculating Features for File:'+str(imno+1)+ ' Timepoint: '+str(tpt+1) +' Z: '+str(zslice+1))
                        if 0==1:
                            featwithzt = np.zeros((feat.shape[0],feat.shape[1],8+feat.shape[2]))
                            featwithzt[:,:,8:]=feat
                            if zslice>1:
                                featwithzt[:,:,0:2] = par_obj.filehandlers[imno].get_tiff_slice([tpt], [zslice-2], width, height, par_obj.ch_active)+par_obj.filehandlers[imno].get_tiff_slice([tpt], [zslice-1], width, height, par_obj.ch_active)-2*par_obj.filehandlers[imno].get_tiff_slice([tpt], [zslice], width, height, par_obj.ch_active)
                            if zslice<(par_obj.max_z-1):
                                featwithzt[:,:,2:4] = par_obj.filehandlers[imno].get_tiff_slice([tpt], [zslice+2], width, height, par_obj.ch_active)+par_obj.filehandlers[imno].get_tiff_slice([tpt], [zslice+1], width, height, par_obj.ch_active)-2*par_obj.filehandlers[imno].get_tiff_slice([tpt], [zslice], width, height, par_obj.ch_active)
                            if tpt>5:
                                featwithzt[:,:,4:6] = par_obj.filehandlers[imno].get_tiff_slice([tpt-5], [zslice], width, height, par_obj.ch_active)+par_obj.filehandlers[imno].get_tiff_slice([tpt-4], [zslice], width, height, par_obj.ch_active)-2*par_obj.filehandlers[imno].get_tiff_slice([tpt], [zslice], width, height, par_obj.ch_active)
                            if tpt<(par_obj.max_t-5):
                                featwithzt[:,:,6:8] = par_obj.filehandlers[imno].get_tiff_slice([tpt+5], [zslice], width, height, par_obj.ch_active)+par_obj.filehandlers[imno].get_tiff_slice([tpt+4], [zslice], width, height, par_obj.ch_active)-2*par_obj.filehandlers[imno].get_tiff_slice([tpt], [zslice], width, height, par_obj.ch_active)
                            feat=featwithzt

                        par_obj.num_of_feat[0] = feat.shape[2]
                        #print 'for test'
                        par_obj.data_store['feat_arr'][imno][tpt][zslice] = feat
        int_obj.report_progress('Features calculated')
    return

def return_rgb_slice(par_obj, zslice, tpt, imno):
    """Fetches slice zslice of timepoint tpt and puts in RGB format for display"""
    width = range(0, par_obj.ori_width, int(par_obj.resize_factor))
    height = range(0, par_obj.ori_height, int(par_obj.resize_factor))
    imfile = par_obj.filehandlers[imno]
    clim = par_obj.clim
    imRGB = np.zeros((int(par_obj.height), int(par_obj.width), 3), 'float32')

    if len(par_obj.ch_display) > 1 or (len(par_obj.ch_display) == 1 and par_obj.numCH > 1):

        if par_obj.numCH > 3:
            for i, ch in enumerate(par_obj.ch_display):
                if i == 3: break #can only display 3 channels

                input_im = imfile.get_tiff_slice([tpt], [zslice], width, height, [ch])
                imRGB[:, :, i] = (input_im.astype('float32')/par_obj.filehandlers[imno].tiffarraymax)*clim[ch][1]-clim[ch][0]
        else:
            for i, ch in enumerate(par_obj.ch_display):

                input_im = imfile.get_tiff_slice([tpt], [zslice], width, height, [ch])
                imRGB[:, :, ch] = (input_im.astype('float32')/par_obj.filehandlers[imno].tiffarraymax)*clim[ch][1]-clim[ch][0]

    elif par_obj.numCH == 1 and len(par_obj.ch_display) == 1:
        input_im = imfile.get_tiff_slice([tpt], [zslice], width, height)

        im = ((input_im.astype('float32')/par_obj.filehandlers[imno].tiffarraymax)*clim[0][1]-clim[0][0])
        imRGB[:, :, 0] = im
        imRGB[:, :, 1] = im
        imRGB[:, :, 2] = im


    imRGB = np.clip(imRGB, 0, 1)
    return imRGB


def evaluate_forest_new(par_obj, int_obj, withGT, model_num, zsliceList, tptList, curr_file, arr='feat_arr',threaded=False):

    #Finds the current frame and file.

    for imno in curr_file:
        for tpt in tptList:
            for zslice in zsliceList:
                '''
                if(par_obj.p_size >1):

                    mimg_lin,dense_linPatch, pos = extractPatch(par_obj.p_size, par_obj.feat_arr[zslice], None, 'dense')
                    tree_pred = par_obj.RF[model_num].predict(mimg_lin)

                    linPred = v2.regenerateImg(par_obj.p_size, tree_pred, pos)

                else:
                '''
                if par_obj.p_size==1:
                    mimg_lin = np.reshape(par_obj.data_store[arr][imno][tpt][zslice], (par_obj.height * par_obj.width, par_obj.data_store[arr][imno][tpt][zslice].shape[2]))
                if par_obj.p_size==2:
                    mimg_lin= par_obj.data_store[arr][imno][tpt][zslice]
                t2 = time.time()
                linPred = par_obj.RF[model_num].predict(mimg_lin).astype('uint16')
                #linPred=linPred[:,1]-linPred[:,0]
                t1 = time.time()

                par_obj.data_store['pred_arr'][imno][tpt][zslice] = linPred.reshape(par_obj.height, par_obj.width)

                maxPred = np.max(linPred)
                minPred = np.min(linPred)

                par_obj.maxPred = max(par_obj.maxPred, maxPred)
                par_obj.minPred = min(par_obj.minPred, minPred)

                sum_pred = np.sum(linPred/255)
                par_obj.data_store['sum_pred'][imno][tpt][zslice] = sum_pred

                print ('prediction time taken', t1 - t2,  ' Predicted i:', par_obj.data_store['sum_pred'][imno][tpt][zslice])
                int_obj.report_progress('Making Prediction for File: '+str(imno+1)+' T: '+str(tpt+1)+' Z: ' +str(zslice+1))


                if withGT == True:
                    try:
                        #If it has already been opened.
                        a = par_obj.data_store['gt_sum'][imno][tpt][zslice]
                    except:
                        #Else find the file.
                        gt_im = pylab.imread(par_obj.data_store['gt_array'][imno][tpt][zslice])[:, :, 0]
                        par_obj.data_store['gt_sum'][imno][tpt][zslice] = np.sum(gt_im)

def evaluate_forest_auto(par_obj, int_obj, withGT, model_num, zsliceList, tptList, curr_file):

    par_obj.maxPred = 0 #resets scaling for display between models
    par_obj.minPred = 100
    for imno in curr_file:
        for tpt in tptList:
            for zslice in zsliceList:
                '''
                if(par_obj.p_size >1):

                    mimg_lin,dense_linPatch, pos = extractPatch(par_obj.p_size, par_obj.feat_arr[zslice], None, 'dense')
                    tree_pred = par_obj.RF[model_num].predict(mimg_lin)

                    linPred = v2.regenerateImg(par_obj.p_size, tree_pred, pos)

                else:
                '''
                mimg_lin = np.reshape(par_obj.data_store['double_feat_arr'][imno][tpt][zslice], (par_obj.height * par_obj.width, par_obj.data_store['double_feat_arr'][imno][tpt][zslice].shape[2]))
                t2 = time.time()
                linPred = par_obj.RF[model_num].predict(mimg_lin)


                t1 = time.time()



                par_obj.data_store['pred_arr'][imno][tpt][zslice] = linPred.reshape(par_obj.height, par_obj.width)

                maxPred = np.max(linPred)
                minPred = np.min(linPred)
                par_obj.maxPred = max([par_obj.maxPred, maxPred])
                par_obj.minPred = min([par_obj.minPred, minPred])
                sum_pred = np.sum(linPred/255)
                par_obj.data_store['sum_pred'][imno][tpt][zslice] = sum_pred

                #print ('prediction time taken', t1 - t2)
                #print ('Predicted i:', par_obj.data_store['sum_pred'][imno][tpt][zslice])
                int_obj.report_progress('Making Prediction for Image: '\
                +str(imno+1)+' Frame: ' +str(zslice+1)+' Timepoint: '+str(tpt+1))

def goto_img_fn_new(par_obj, int_obj,keep_roi=False):
    """Loads up current image and displays it"""

    tpt = par_obj.curr_t
    zslice = par_obj.curr_z
    imno = par_obj.curr_file
    t0 = time.time()
    #Finds the current frame and file.
    newImg = return_rgb_slice(par_obj, zslice, tpt, imno)
    par_obj.save_im = newImg
    #deals with displaying different channels
    if par_obj.overlay and zslice in par_obj.data_store['pred_arr'][imno][tpt]:
        pred_img=(par_obj.data_store['pred_arr'][imno][tpt][zslice])/par_obj.maxPred
        pred_img[pred_img<0]=0
        newImg[:, :, 2] = pred_img

    #sets image for display
    int_obj.plt1.images[0].set_data(newImg)

    #remove rois
    if not keep_roi:
        int_obj.plt1.lines = []

    #update image text
    int_obj.image_num_txt.setText('Current File is : ' + str(par_obj.curr_file+1)\
    +'/'+str(par_obj.max_file)+', Current Time: '+str(par_obj.curr_t+1)+'/'+str(par_obj.max_t+1)\
    +', Current Z: '+str(par_obj.curr_z+1)+'/'+str(par_obj.max_z+1))
    # filename: ' +str(evalLoadImWin.file_array[im_num]))

    """Deals with displaying Kernel/Prediction/Counts"""
    #im2draw = None

    if par_obj.show_pts == 0:
        if zslice in par_obj.data_store['dense_arr'][imno][tpt]:

            im2draw = par_obj.data_store['dense_arr'][imno][tpt][zslice]

            int_obj.plt2_is_clear = False
            int_obj.plt2.images[0].set_data(im2draw)
            int_obj.plt2.images[0].autoscale()
            int_obj.canvas2.draw()

        elif int_obj.plt2_is_clear is not True:

            int_obj.plt2_is_clear = True
            im2draw = np.zeros((par_obj.height, par_obj.width))
            int_obj.plt2.images[0].set_data(im2draw)
            int_obj.canvas2.draw()

        else: # don't update
            pass

    elif par_obj.show_pts == 1:
        if zslice in par_obj.data_store['pred_arr'][imno][tpt]:
            im2draw = par_obj.data_store['pred_arr'][imno][tpt][zslice].astype(np.float32)
        else:
            im2draw = np.zeros((par_obj.height, par_obj.width))

        int_obj.plt2.images[0].set_clim(0, par_obj.maxPred)
        int_obj.plt2.images[0].set_data(im2draw)
        #int_obj.plt2.autoscale()
        int_obj.canvas2.draw()

    elif par_obj.show_pts == 2:
        #show det(hessian) array, and the green circles?
        pt_x = []
        pt_y = []
        pts = par_obj.data_store['pts'][imno][tpt]

        ind = np.where(np.array(range(par_obj.filehandlers[imno].max_z+1)) == zslice)

        for pt2d in pts:
            #if pt2d[3] == 0:
            #    break
            for i1 in ind:
                z_range=range(int(i1)-int(par_obj.min_distance[2]),int(i1)+int(par_obj.min_distance[2])+1)
            
                if pt2d[2] in z_range or par_obj.z_project:
                    pt_x.append(pt2d[1])
                    pt_y.append(pt2d[0])

        int_obj.plt2.lines = []
        int_obj.plt2.axes.plot(pt_x, pt_y, 'wo',markersize=max(2,2*par_obj.min_distance[0]))
        int_obj.plt2.autoscale_view(tight=True)
        int_obj.plt1.axes.plot(pt_x, pt_y, 'wo',markersize=max(2,2*par_obj.min_distance[0]))
        int_obj.plt1.autoscale_view(tight=True)

        if zslice in par_obj.data_store['maxi_arr'][imno][tpt]:
            im2draw = par_obj.data_store['maxi_arr'][imno][tpt][zslice].astype(np.float32)
            int_obj.plt2.images[0].set_clim(0, 1)
            int_obj.plt2.images[0].set_data(im2draw)
            int_obj.canvas2.draw()

    #set data and draw canvas


    int_obj.draw_saved_dots_and_roi()
    int_obj.cursor.draw_ROI()
    buffereddraw(int_obj)
    print (time.time() -t0)

def buffereddraw(int_obj):

    if int_obj.threadpool.activeThreadCount()>1:
        return
    else:

        worker = Worker(int_obj.canvas1.draw)
        worker.run()
        return


def load_and_initiate_plots(par_obj, int_obj):
    """prepare plots and data for display"""

    newImg = np.zeros((int(par_obj.height), int(par_obj.width), 3), 'uint8')
    #newImg[:,:,3]=1
    int_obj.plt1.cla()
    int_obj.plt1.imshow(newImg, interpolation='nearest', vmin=0, vmax=255)
    #int_obj.plt1.imshow(newImg,interpolation='nearest')
    int_obj.plt1.axis("off")
    newImg = np.zeros((int(par_obj.height), int(par_obj.width), 3))
    int_obj.plt2.cla()
    int_obj.plt2.imshow(newImg, interpolation='none',cmap='jet')

    #modest_image.imshow(int_obj.plt2.axes,newImg,interpolation='none')
    int_obj.plt2.axis("off")
    int_obj.plt2.autoscale()

    int_obj.cursor.draw_ROI()
    int_obj.image_num_txt.setText('Current File is : ' + str(par_obj.curr_file+1)+'/'+str(par_obj.max_file+1)+' ,Current Time: '+str(par_obj.curr_t+1)+'/'+str(par_obj.max_t+1)+' ,Current Z: '+str(par_obj.curr_z+1)+'/'+str(par_obj.max_z+1))
    # filename: ' +str(evalLoadImWin.file_array[im_num]))

    goto_img_fn_new(par_obj, int_obj)

def eval_pred_show_fn(par_obj, int_obj, zslice, tpt):
    """Shows Prediction Image when forest is loaded"""
    if par_obj.eval_load_im_win_eval == True:

        int_obj.image_num_txt.setText('The Current Image is No. ' + str(zslice+1)+' and the time point is: '+str(tpt+1))

        #if int_obj.count_maxima_plot_on.isChecked() == True:
        #    par_obj.show_pts = True
        #else:
    #        par_obj.show_pts = False
        imno = par_obj.curr_file
        int_obj.plt2.cla()
        if par_obj.show_pts == 0:
            im2draw = np.zeros((par_obj.height, par_obj.width))
            for ind in par_obj.data_store['dense_arr'][imno][tpt]:
                if ind == zslice:
                    im2draw = par_obj.data_store[tpt]['dense_arr'][zslice].astype(np.float32)
            int_obj.plt2.imshow(im2draw)
        if par_obj.show_pts == 1:
            im2draw = np.zeros((par_obj.height, par_obj.width))
            for ind in par_obj.data_store['pred_arr'][imno][tpt]:
                if ind == zslice:
                    im2draw = par_obj.data_store['pred_arr'][imno][tpt][zslice].astype(np.float32)
            int_obj.plt2.imshow(im2draw, vmax=par_obj.maxPred)
        if par_obj.show_pts == 2:

            pt_x = []
            pt_y = []
            pts = par_obj.data_store['pts'][imno][tpt]

            ind = np.where(np.array(par_obj.frames_2_load[0]) == zslice)[0][0]

        for pt2d in pts:
            if pt2d[3] == 0:
                break
            elif pt2d[2] == ind:
                pt_x.append(pt2d[1])
                pt_y.append(pt2d[0])


            int_obj.plt1.plot(pt_x, pt_y, 'wo')
            int_obj.plt2.plot(pt_x, pt_y, 'wo')
            string_2_show = 'The Predicted Count: ' + str(pts.__len__())
            int_obj.output_count_txt.setText(string_2_show)
            im2draw = np.zeros((par_obj.height, par_obj.width))
            for ind in par_obj.data_store['maxi_arr'][imno][tpt]:
                if ind == zslice:
                    im2draw = par_obj.data_store['maxi_arr'][imno][tpt][zslice].astype(np.float32)
            d = int_obj.plt2.imshow(im2draw)
            d.set_clim(0, 255)

        int_obj.plt2.set_xticklabels([])
        int_obj.plt2.set_yticklabels([])
        int_obj.canvas1.draw()
        int_obj.canvas2.draw()


def import_data_fn(par_obj, file_array, file_array_offset=0):
    """Function which loads in list of Tiff stacks and checks for consistency"""
    #careful with use of non-zero offset. Intended primarily for use in validation

    par_obj.max_file = file_array.__len__()

    par_obj.filehandlers = {}

    for im_n, imfile in enumerate(file_array):
        imno=im_n+file_array_offset

        par_obj.filehandlers[imno] = File_handler(str(imfile))
        #currently doesn't check if multiple filetypes, on the basis only loads tiffs
        #check number of channels is consistent
        if imno == 0:
            par_obj.numCH = max(par_obj.filehandlers[imno].numCH,1)

            par_obj.clim = [[0,1] for x in range(par_obj.numCH)]
        elif par_obj.numCH == max(par_obj.filehandlers[imno].numCH,1):
            pass
        else:
            status_text = 'Different number of image channels in the selected images'
            raise Exception(status_text)# if this isn't true then something is wrong
        #check height and width match
        if imno == 0:
            par_obj.ori_height = par_obj.filehandlers[imno].height
        elif par_obj.ori_height == par_obj.filehandlers[imno].height:
            pass
        else:
            status_text = 'Different image size in the selected images'
            raise Exception(status_text)# if this isn't true then something is wrong

        if imno == 0:
            par_obj.ori_width = par_obj.filehandlers[imno].width
        elif par_obj.ori_width == par_obj.filehandlers[imno].width:
            pass
        else:
            status_text = 'Different image size in the selected images'
            raise Exception(status_text)# if this isn't true then something is wrong
        #check bit depth matches
        if imno == 0:
            par_obj.bitDepth = par_obj.filehandlers[imno].bitDepth
            par_obj.tiffarray_typemax = par_obj.filehandlers[imno].tiffarray_typemax
            if par_obj.bitDepth in ['float32']:
                par_obj.tiffarray_typemax=par_obj.filehandlers[imno].tiffarraymax

        elif par_obj.bitDepth == par_obj.filehandlers[imno].bitDepth:
            pass
        else:
            status_text = 'Different image bit depth in the selected images'
            raise Exception(status_text)# if this isn't true then something is wrong

    #set max z and t for current time
    par_obj.max_z = par_obj.filehandlers[par_obj.curr_file].max_z
    par_obj.max_t = par_obj.filehandlers[par_obj.curr_file].max_t

    #Prepare RGB example image
    par_obj.height = len(range(0, par_obj.ori_height, int(par_obj.resize_factor)))
    par_obj.width = len(range(0, par_obj.ori_width, int(par_obj.resize_factor)))
    par_obj.ch_display = list(range(0, min(par_obj.numCH, 3)))
    par_obj.ex_img = return_rgb_slice(par_obj, 0, 0, 0)
    
    statusText = str(file_array.__len__())+' Files Loaded.'
    return True, statusText

def filter_prediction_fn(par_obj, int_obj):
    for fileno, imfile in par_obj.filehandlers.items():
        #funky ordering TZCYX
        image = np.zeros([imfile.max_t+1, imfile.max_z+1, 1, par_obj.height, par_obj.width], 'float32')
        for tpt in range(imfile.max_t+1):
            for zslice in range(imfile.max_z+1):
                image[tpt, zslice, 0, :, :] = par_obj.data_store['pred_arr'][fileno][tpt][zslice].astype(np.float32)
        image = image - filters.uniform_filter1d(image,15,axis=0)
        for tpt in range(imfile.max_t+1):
            for zslice in range(imfile.max_z+1):
                par_obj.data_store['pred_arr'][fileno][tpt][zslice] = np.squeeze(image[tpt, zslice, 0, :, :])
        int_obj.report_progress('Prediction filtered')

def save_output_prediction_fn(par_obj, int_obj,subtract_background=False):
    """Saves prediction to tiff file using tiffiles imsave"""
    for fileno, imfile in par_obj.filehandlers.items():
        #funky ordering TZCYX
        image = np.zeros([imfile.max_t+1, imfile.max_z+1, 1, par_obj.height, par_obj.width], 'uint16')
        for tpt in range(imfile.max_t+1):
            for zslice in range(imfile.max_z+1):
                image[tpt, zslice, 0, :, :] = par_obj.data_store['pred_arr'][fileno][tpt][zslice].astype(np.float32)
        if subtract_background:
            image = image - filters.uniform_filter1d(image,10,axis=0)
            imsave(imfile.path+'/'+imfile.name+'_'+par_obj.modelName+'_Prediction.tif', image, imagej=True)
        else:
            imsave(imfile.path+'/'+imfile.name+'_'+par_obj.modelName+'_Prediction.tif', image, imagej=True)

        print ('Prediction written to disk')
        int_obj.report_progress('Prediction written to disk '+ imfile.path)

def save_kernels_fn(par_obj, int_obj):
    """Saves prediction to tiff file using tiffiles imsave"""
    for fileno, imfile in par_obj.filehandlers.items():
        #funky ordering TZCYX

        image = np.zeros([imfile.max_t+1, imfile.max_z+1, 1, par_obj.height, par_obj.width], 'uint16')
        for tpt in range(imfile.max_t):
            for zslice in range(imfile.max_z+1):
                if zslice in par_obj.data_store['dense_arr'][fileno][tpt]:
                    image[tpt, zslice, 0, :, :] = par_obj.data_store['dense_arr'][fileno][tpt][zslice].astype(np.float32)
        imsave(imfile.path+'/'+imfile.name+'_'+par_obj.modelName+'_Kernels.tif', image, imagej=True)

        print ('Prediction written to disk')
        int_obj.report_progress('Prediction written to disk '+ imfile.path)

def save_output_hess_fn(par_obj, int_obj):
    """Saves hessian map to tiff file using tiffiles imsave"""
    for fileno, imfile in par_obj.filehandlers.items():
        #funky ordering TZCYX
        image = np.zeros([imfile.max_t+1, imfile.max_z+1, 1, par_obj.height, par_obj.width], 'float32')
        for tpt in range(imfile.max_t+1):
            for zslice in range(imfile.max_z+1):
                image[tpt, zslice, 0, :, :] = par_obj.data_store['maxi_arr'][fileno][tpt][zslice].astype('float32')

        print ('Saving Hessian image to disk')
        imsave(imfile.path+'/'+imfile.name+'_'+par_obj.modelName+'_Hess.tif', image, imagej=True)
        int_obj.report_progress('Hessian written to disk '+ imfile.path)

def save_output_mask_fn(par_obj,int_obj):
    #funky ordering TZCYX
    for fileno,imfile in par_obj.filehandlers.items():
        filename = imfile.full_name
        image = np.zeros([imfile.max_t+1,imfile.max_z+1,1,par_obj.height,par_obj.width], 'uint8')
        for tpt in range(imfile.max_t+1):

            for i in range(0,par_obj.data_store['pts'][fileno][tpt].__len__()):
                [x,y,z,W]=par_obj.data_store['pts'][fileno][tpt][i]
                if W:
                    image[tpt,z,0,x,y]=255
        image = filters.maximum_filter(image,size=(0,3,0,3,3))
        imsave(filename+'_'+par_obj.modelName+'_Points.tif',image, imagej=True)
        int_obj.report_progress('Point mask written to disk '+ imfile.path)

def save_output_ROI(par_obj, int_obj):
    #funky ordering TZCYX
    for fileno, imfile in par_obj.filehandlers.items():
        filename = imfile.base_name
        with open(imfile.path+filename+'_outputROI.pickle', 'wb') as afile:
            pickle.dump([par_obj.data_store['roi_stkint_x'][fileno], par_obj.data_store['roi_stkint_y'][fileno]], afile)

    with open(par_obj.filehandlers[0].path+par_obj.modelName+'_outputROI.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([str('Filename: ')]+[str('Time point: ')]+[str('Z: ')]+[str('Regions(x): ')]+[str('Regions(y): ')])

        for fileno, imfile in par_obj.filehandlers.items():
            filename = imfile.full_name
            for tpt in range(imfile.max_t+1):
                ppt_x = par_obj.data_store['roi_stkint_x'][fileno][tpt]
                ppt_y = par_obj.data_store['roi_stkint_y'][fileno][tpt]
                for i in ppt_x.keys():
                    spamwriter.writerow([str(filename)]+[tpt+1]+[i]+[ppt_x[i]]+[ppt_y[i]])
    int_obj.report_progress('Data exported to '+ imfile.path)

def save_output_data_fn(par_obj, int_obj):
    local_time = time.asctime(time.localtime(time.time()))

    with open(par_obj.filehandlers[0].path+'/_'+par_obj.modelName+'outputData.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        spamwriter.writerow([str(par_obj.selectedModel)]+[str('Filename: ')]+[str('Time point: ')]+[str('Predicted count: ')])

        for fileno, imfile in par_obj.filehandlers.items():

            for tpt in par_obj.tpt_list:
                filename = str(par_obj.file_array[fileno])
                spamwriter.writerow([local_time]+[str(filename)]+[tpt+1]+[par_obj.data_store['pts'][fileno][tpt].__len__()])


    for fileno, imfile in par_obj.filehandlers.items():
        with open(imfile.path+'/'+imfile.name+'_'+par_obj.modelName+'_outputPoints.csv', 'a') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow([str(par_obj.selectedModel)]+[str('Filename: ')]+[str('Time point: ')]+[str('X: ')]+[str('Y: ')]+[str('Z: ')])

            for tpt in par_obj.tpt_list:
                pts = par_obj.data_store['pts'][fileno][tpt]
                for i in range(0, par_obj.data_store['pts'][fileno][tpt].__len__()):
                    spamwriter.writerow([local_time]+[str(filename)]+[tpt+1]+[pts[i][0]]+[pts[i][1]]+[pts[i][2]])
        int_obj.report_progress('File '+str(fileno)+' Data exported to '+ imfile.path)

def save_user_ROI(par_obj, int_obj):
    #funky ordering TZCYX
    for fileno, imfile in par_obj.filehandlers.items():
        filename = imfile.base_name
        with open(imfile.path+'/'+filename+'_outputROI.pickle', 'wb') as afile:
            data = [par_obj.data_store['roi_stkint_x'][fileno], par_obj.data_store['roi_stkint_y'][fileno],\
                    par_obj.data_store['roi_stk_x'][fileno], par_obj.data_store['roi_stk_y'][fileno]]
            pickle.dump(data, afile, par_obj.resize_factor)

    with open(imfile.path+'/'+par_obj.modelName+'_outputROI.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([str('Filename: ')]+[str('Time point: ')]+[str('Z: ')]+[str('Regions(x): ')]+[str('Regions(y): ')])

        for fileno, imfile in par_obj.filehandlers.items():
            filename = imfile.full_name
            for tpt in range(imfile.max_t+1):
                ppt_x = par_obj.data_store['roi_stk_x'][fileno][tpt]
                ppt_y = par_obj.data_store['roi_stk_y'][fileno][tpt]
                for i in ppt_x.keys():
                    spamwriter.writerow([str(filename)]+[tpt+1]+[i]+[ppt_x[i]]+[ppt_y[i]])
    int_obj.report_progress('Data exported to '+ imfile.path)

def save_ROI_area(par_obj, int_obj):

    with open(par_obj.filehandlers[0].path+'/'+par_obj.modelName+'_outputROI_area.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([str('Filename: ')]+[str('Time point: ')]+[str('Area')])

        for fileno, imfile in par_obj.filehandlers.items():
            filename = imfile.full_name
            for tpt in range(imfile.max_t+1):
                ppt_x = par_obj.data_store['roi_stkint_x'][fileno][tpt]
                area = 0
                for i in ppt_x.keys():
                    area += ROI_area(par_obj, fileno, tpt, i)
                spamwriter.writerow([str(filename)]+[tpt+1]+[area])

    int_obj.report_progress('Data exported to '+ par_obj.filehandlers[0].path)

def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

def ROI_area(par_obj, fileno, tpt, zslice):
    'Calculates area of interpolated ROI'
    ppt_x = par_obj.data_store['roi_stkint_x'][fileno][tpt][zslice]
    ppt_y = par_obj.data_store['roi_stkint_y'][fileno][tpt][zslice]
    imfile = par_obj.filehandlers[fileno]
    counter = 0

    counter += imfile.z_calibration * PolyArea(np.array(ppt_x), np.array(ppt_y))

    return counter

def load_user_ROI(par_obj, int_obj):
    #funky ordering TZCYX
    #fileName = QtGui.QFileDialog.getOpenFileName(None, "Load ROIs", filter="QuantiFly ROI files (*.outputROI)")
    #filename, file_ext = os.path.splitext(os.path.basename(fileName[0:-18]))

    for fileno, imfile in par_obj.filehandlers.items():
        name = imfile.full_name +'_outputROI.pickle'
        if os.path.isfile(name):

            with open(name, 'rb') as afile:
                rois = pickle.load(afile)
                par_obj.data_store['roi_stkint_x'][fileno] = rois[0]
                resizef = 1

                roi_f = par_obj.data_store['roi_stkint_x'][fileno]
                for t in roi_f:
                    for z in roi_f[t]:
                        roi_f[t][z] = [x*resizef for x in roi_f[t][z]]

                par_obj.data_store['roi_stkint_y'][fileno] = rois[1]
                roi_f = par_obj.data_store['roi_stkint_y'][fileno]
                for t in roi_f:
                    for z in roi_f[t]:
                        roi_f[t][z] = [x*resizef for x in roi_f[t][z]]



                if len(rois) > 2:
                    par_obj.data_store['roi_stk_x'][fileno] = rois[2]
                    roi_f = par_obj.data_store['roi_stk_x'][fileno]
                    for t in roi_f:
                        for z in roi_f[t]:
                            roi_f[t][z] = [x*resizef for x in roi_f[t][z]]
                    par_obj.data_store['roi_stk_y'][fileno] = rois[3]
                    roi_f = par_obj.data_store['roi_stk_y'][fileno]
                    for t in roi_f:
                        for z in roi_f[t]:
                            roi_f[t][z] = [x*resizef for x in roi_f[t][z]]

    int_obj.report_progress('ROIs imported')

def setup_parameters(self, par_obj):
    """Loads parameters and initiates data structure"""
    #Resets everything should this be another patch of images loaded.

    par_obj.height = len(range(0, par_obj.ori_height, int(par_obj.resize_factor)))
    par_obj.width = len(range(0, par_obj.ori_width, int(par_obj.resize_factor)))

    par_obj.curr_t = par_obj.tpt_list[0]
    par_obj.curr_z = par_obj.user_min_z
    par_obj.curr_file = 0

    par_obj.initiate_data_store()

    par_obj.frames_2_load = {}
    par_obj.left_2_calc = []
    par_obj.saved_ROI = []
    par_obj.saved_dots = []

#if __name__ == '__main__':
    #multiprocessing.freeze_support()
    #multiprocessing.spawn.freeze_support()
class Worker(QtCore.QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @QtCore.pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.fn(*self.args, **self.kwargs)

