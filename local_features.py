"""
Created on Wed Mar 30 14:25:35 2016

@author: martin

Feature calculation methods (moved from v2_functions for clarity)
 makes adding new features more straightforward
 might be nice to add subclasses to this
"""
import vigra
import scipy

from skimage import feature as skfeat

from skimage import exposure

import skimage
import numpy as np

#from v2_functions import get_tiff_slice
from scipy.ndimage.interpolation import shift

from sklearn import ensemble
from sklearn import tree
from sklearn.pipeline import Pipeline


def RF(par_obj, RF_type='ETR'):
    """Choose regression method. Must implement fit and predict. Can use sklearn pipeline"""
    if RF_type == 'ETR':
        method = ensemble.ExtraTreesRegressor(par_obj.num_of_tree, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split,
                                              min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1)
    elif RF_type == 'GBR':
        method = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.01, n_estimators=par_obj.num_of_tree, max_depth=par_obj.max_depth,
                                                    min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features)
    elif RF_type == 'GBR2':
        method = ensemble.GradientBoostingRegressor(loss='lad', learning_rate=0.1, n_estimators=par_obj.num_of_tree, max_depth=par_obj.max_depth,
                                                    min_samples_split=par_obj.min_samples_split, min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features)
    elif RF_type == 'ABR':
        method = ensemble.AdaBoostRegressor(base_estimator=ensemble.ExtraTreesRegressor(10, max_depth=par_obj.max_depth, min_samples_split=par_obj.min_samples_split,
                                                                                        min_samples_leaf=par_obj.min_samples_leaf, max_features=par_obj.max_features, bootstrap=True, n_jobs=-1), n_estimators=3, learning_rate=1.0, loss='square')

    return method


def get_feature_lengths(feature_type):
    """Choose feature sets, accepts feature types 'basic' 'pyramid' 'imhist' 'dual'"""
    # dictionary of feature sets
    # can add arbitrary feature sets by defining a name, length, and function
    # that accepts two arguments
    feature_dict = {'basic': [13, local_shape_features_basic],
                    'fine': [21, local_shape_features_fine],
                    'pyramid': [26, local_shape_features_pyramid],
                    'histeq': [26, local_shape_features_fine_imhist]}

    if feature_dict.has_key(feature_type):
        feat_length = feature_dict[feature_type][0]
        feat_func = feature_dict[feature_type][1]
    else:
        raise Exception('Feature set not found')

    return feat_length, feat_func


def feature_create_threadable(par_obj, imRGB):
    """Creates features based on input image"""
    # get number of features
    [feat_length, feat_func] = get_feature_lengths(par_obj.feature_type)

    # preallocate array
    feat = np.zeros(((int(par_obj.crop_y2) - int(par_obj.crop_y1)), (int(par_obj.crop_x2) -
    int(par_obj.crop_x1)), feat_length * (par_obj.ch_active.__len__())))

    if par_obj.numCH == 1:
        imG = imRGB[:, :].astype(np.float32)

        feat = feat_func(imG, par_obj.feature_scale)
    else:
        for b in range(0, par_obj.ch_active.__len__()):
            imG = imRGB[:, :, b].astype(np.float32)
            feat[:, :, (b * feat_length):((b + 1) * feat_length)] = feat_func(imG, par_obj.feature_scale)
    '''        if b==1:#dirty hack to test
                imG = imRGB[:,:,b].astype(np.float32)*imRGB[:,:,b].astype(np.float32)
                feat[:,:,(2*feat_length):((2+1)*feat_length)]=feat_func(imG,par_obj.feature_scale)'''

    return feat


def feature_create_threadable_auto(par_obj, imno, tpt, zslice):
    # allows auto-context based features to be included
    # currently slow, could optimise to calculate more efficiently
    [feat_length, feat_func] = get_feature_lengths(par_obj.feature_type)
    feat = feat_func(
        par_obj.data_store['pred_arr'][imno][tpt][zslice], par_obj.feature_scale)
    return feat


def auto_context_features(feat_array):
    # pattern=[1,2,3,5,7,10,12,15,20,25,30,35,40,45,50,60,70,80,90,100]
    pattern = [1, 2, 4, 8, 16, 32, 64, 128]
    feat_list = []
    for xshift in pattern:
        for yshift in pattern:
            a = shift(feat_array, (xshift, 0), order=0, cval=0)
            b = shift(feat_array, (-xshift, 0), order=0, cval=0)
            c = shift(feat_array, (yshift, 0), order=0, cval=0)
            d = shift(feat_array, (-yshift, 0), order=0, cval=0)
            feat_list.append(a)
            feat_list.append(b)
            feat_list.append(c)
            feat_list.append(d)
    return feat_list


def local_shape_features_fine(im, scaleStart):
    """ Creates features. Exactly as in the Luca Fiaschi paper but on 5 scales, and a truncated gaussian"""
    s = scaleStart

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 21))

    st08 = vigra.filters.structureTensorEigenvalues(im, s * 1, s * 2)
    st16 = vigra.filters.structureTensorEigenvalues(im, s * 2, s * 4)
    st32 = vigra.filters.structureTensorEigenvalues(im, s * 4, s * 8)
    st64 = vigra.filters.structureTensorEigenvalues(im, s * 8, s * 16)
    st128 = vigra.filters.structureTensorEigenvalues(im, s * 16, s * 32)

    f[:, :, 0] = im
    f[:, :, 1] = vigra.filters.gaussianGradientMagnitude(
        im, s, window_size=2.5)
    f[:, :, 2] = st08[:, :, 0]
    f[:, :, 3] = st08[:, :, 1]
    f[:, :, 4] = vigra.filters.laplacianOfGaussian(im, s, window_size=2.5)
    f[:, :, 5] = vigra.filters.gaussianGradientMagnitude(
        im, s * 2, window_size=2.5)
    f[:, :, 6] = st16[:, :, 0]
    f[:, :, 7] = st16[:, :, 1]
    f[:, :, 8] = vigra.filters.laplacianOfGaussian(im, s * 2, window_size=2.5)
    f[:, :, 9] = vigra.filters.gaussianGradientMagnitude(
        im, s * 4, window_size=2.5)
    f[:, :, 10] = st32[:, :, 0]
    f[:, :, 11] = st32[:, :, 1]
    f[:, :, 12] = vigra.filters.laplacianOfGaussian(im, s * 4, window_size=2.5)
    f[:, :, 13] = vigra.filters.gaussianGradientMagnitude(
        im, s * 8, window_size=2.5)
    f[:, :, 14] = st64[:, :, 0]
    f[:, :, 15] = st64[:, :, 1]
    f[:, :, 16] = vigra.filters.laplacianOfGaussian(im, s * 8, window_size=2.5)
    f[:, :, 17] = vigra.filters.gaussianGradientMagnitude(
        im, s * 16, window_size=2.5)
    f[:, :, 18] = st128[:, :, 0]
    f[:, :, 19] = st128[:, :, 1]
    f[:, :, 20] = vigra.filters.laplacianOfGaussian(
        im, s * 16, window_size=2.5)
    return f


def local_shape_features_pyramid(im, scaleStart):
    """ Creates features based on those in the Luca Fiaschi paper but on 5 scales independent of object size,
    but using a gaussian pyramid to calculate at multiple scales more efficiently
    and then linear upsampling to original image scale
    also includes gaussian smoothed image
    """
    # Smoothing and scale parameters chosen to approximate those in 'fine'
    # features

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 26))
    f[:, :, 0] = im

    # create pyramid structure
    pyr = skimage.transform.pyramid_gaussian(
        im, sigma=1.5, max_layer=5, downscale=2)

    a = im
    for layer in range(0, 5):

        # calculate scale
        scale = [float(im.shape[0]) / float(a.shape[0]),
                 float(im.shape[1]) / float(a.shape[1])]

        # create features
        lap = scipy.ndimage.filters.laplace(a)

        [m, n] = np.gradient(a)
        ggm = np.hypot(m, n)

        x, y, z = skfeat.structure_tensor(a, 1)
        st = skfeat.structure_tensor_eigvals(x, y, z)

        # upsample features to original image
        lap = scipy.ndimage.interpolation.zoom(lap, scale, order=1)
        ggm = scipy.ndimage.interpolation.zoom(ggm, scale, order=1)
        st0 = scipy.ndimage.interpolation.zoom(st[0], scale, order=1)
        st1 = scipy.ndimage.interpolation.zoom(st[1], scale, order=1)
        upsampled = scipy.ndimage.interpolation.zoom(a, scale, order=1)

        f[:, :, layer * 5 + 1] = lap
        f[:, :, layer * 5 + 2] = ggm
        f[:, :, layer * 5 + 3] = st0
        f[:, :, layer * 5 + 4] = st1
        f[:, :, layer * 5 + 5] = upsampled

        # get next layer
        a = pyr.next()

    return f


def local_shape_features_fine_imhist(im, scaleStart):
    """As per pyramid features but with histogram equalisation"""

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 26))
    im = exposure.equalize_hist(im)
    #im=exposure.equalize_adapthist(im, kernel_size=5)
    f[:, :, 0] = im

    # set up pyramid
    pyr = skimage.transform.pyramid_gaussian(
        im, sigma=1.5, max_layer=5, downscale=2)
    a = im

    for layer in range(0, 5):

        scale = [float(im.shape[0]) / float(a.shape[0]),
                 float(im.shape[1]) / float(a.shape[1])]

        lap = scipy.ndimage.filters.laplace(a)

        [m, n] = np.gradient(a)
        ggm = np.hypot(m, n)

        x, y, z = skfeat.structure_tensor(a, 1)
        st = skfeat.structure_tensor_eigvals(x, y, z)

        lap = scipy.ndimage.interpolation.zoom(lap, scale, order=1)
        ggm = scipy.ndimage.interpolation.zoom(ggm, scale, order=1)
        st0 = scipy.ndimage.interpolation.zoom(st[0], scale, order=1)
        st1 = scipy.ndimage.interpolation.zoom(st[1], scale, order=1)
        up = scipy.ndimage.interpolation.zoom(a, scale, order=1)

        f[:, :, layer * 5 + 1] = lap
        f[:, :, layer * 5 + 2] = ggm
        f[:, :, layer * 5 + 3] = st0
        f[:, :, layer * 5 + 4] = st1
        f[:, :, layer * 5 + 5] = up
        a = pyr.next()
    return f


def local_shape_features_basic(im, scaleStart):
    """Exactly as in the Luca Fiaschi paper. Calculates features at 3 scales dependent on scale parameter"""
    s = scaleStart

    imSizeC = im.shape[0]
    imSizeR = im.shape[1]
    f = np.zeros((imSizeC, imSizeR, 13))

    st08 = vigra.filters.structureTensorEigenvalues(im, s * 1, s * 2)
    st16 = vigra.filters.structureTensorEigenvalues(im, s * 2, s * 4)
    st32 = vigra.filters.structureTensorEigenvalues(im, s * 4, s * 8)

    f[:, :, 0] = im
    f[:, :, 1] = vigra.filters.gaussianGradientMagnitude(
        im, s, window_size=2.5)

    f[:, :, (2, 3)] = st08
    f[:, :, 4] = vigra.filters.laplacianOfGaussian(im, s, window_size=2.5)

    f[:, :, 5] = vigra.filters.gaussianGradientMagnitude(
        im, s * 2, window_size=2.5)

    f[:, :, (6, 7)] = st16

    f[:, :, 8] = vigra.filters.laplacianOfGaussian(im, s * 2, window_size=2.5)

    f[:, :, 9] = vigra.filters.gaussianGradientMagnitude(
        im, s * 4, window_size=2.5)

    f[:, :, (10, 11)] = st32

    f[:, :, 12] = vigra.filters.laplacianOfGaussian(im, s * 4, window_size=2.5)
    return f
