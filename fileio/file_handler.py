#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import numpy as np
from tifffile import TiffFile, imsave, TiffWriter  # Install with pip install tifffile.

"""
Created on Fri Feb 17 19:48:58 2017
@author: martin
Since it's clear that we want to handle files with all their own data in a
coherent way
And at the moment we rely on 'Parameter Object' effectively global parameters
Which contain lists or dictionaries of the relevant parameters
This means that it is possible to have some global file-related parameters
Along with some non-global ones
This is potentially neccessary for things such as the Z-calibration, which
really shouldn't differ between images.
But is not a particularly good way to handle the images along with their associated data
Because it requires specialised parameters and logic for each.
A more coherent way to do this would be to have a class that holds all the file specific
information, and then is simply accessed when needed
The parameter object then need only hold relevant parameters to the
user interface, for instance those locating the current Z, Time, and File number

These can then be passed to the File handler class, which will return the requested
image data, or z_cal, etc.

Special thought should be extended to the behaviour of the handler at the limits
of the file extents if this information is no longer directly accessible to the UI

For instance, it may be necessary to compare the desired file extent to the
possible ones in the UI, thus preventing the need for an actual refresh

But equally, it may be better for the File handler to return the 'new' extent
even when it is the same because we have gone off the end of the file

The former seems neater.

The file handlers themselves can be a list as part of par_obj, thus allowing
easy pass of information between them.
Theoretically it might be nice to separate them entirely, but I suspect this
would cause problems without some significant rewrites"""


class File_handler(object):
    """create new File_handler object for each file"""

    def __init__(self, file_path):
        # store high level file data and metadata
        self.full_name = file_path

        self.base_name = os.path.basename(file_path)
        self.path = os.path.dirname(file_path)

        self.ext = self.base_name.split(".")[-1]
        self.name = self.base_name.split(".")[0]

        self.array = []  # memmap object l
        self.z_calibration = 1
        self.order = {}  # ordering of tiff objects

        # default file extents
        self.height = 0
        self.width = 0
        self.max_t = 0
        self.max_z = 0
        self.numCH = 0
        self.tiff = None
        self.tiffarray_typemax = 0  # dtype max
        self.tiffarraymax = 0  # actual data maximum
        self.bitDepth = 0
        self.import_file()

    def get_tiff_slice(self, tpt=[0], zslice=[0], x=[0], y=[0], c=[0]):
        """Get image cube using numpy views, dealing with different TXYZC orderings. Always return TZYXC"""
        # handles lists and ints nicely
        if not isinstance(zslice, list):  # type(zslice) is not list:
            zslice = [zslice]
        if isinstance(zslice[0], list):
            zslice = zslice[0]

        alist = []
        blist = []
        for n, axis in enumerate(self.order):
            if axis == "T":
                alist.append(tpt)
                blist.append(n)
        for n, axis in enumerate(self.order):
            if axis == "Z" or axis == "Q":
                alist.append(zslice)
                blist.append(n)
        for n, axis in enumerate(self.order):
            if axis == "Y":
                alist.append(y)
                blist.append(n)
        for n, axis in enumerate(self.order):
            if axis == "X":
                alist.append(x)
                blist.append(n)

        for n, axis in enumerate(self.order):
            if axis == "C" or axis == "S":
                alist.append(c)
                blist.append(n)

        tiff2 = self.array.transpose(blist)

        if self.order.__len__() == 5:
            tiff = np.squeeze(tiff2[np.ix_(alist[0], alist[1], alist[2], alist[3], alist[4])])

        elif self.order.__len__() == 4:
            tiff = np.squeeze(tiff2[np.ix_(alist[0], alist[1], alist[2], alist[3])])

        elif self.order.__len__() == 3:
            tiff = np.squeeze(tiff2[np.ix_(alist[0], alist[1], alist[2])])

        elif self.order.__len__() == 2:
            tiff = np.squeeze(tiff2[np.ix_(alist[0], alist[1])])

        return tiff

    def close(self):
        self.array = []
        self.tiff.close()

    def import_file(self):
        """ loads in File image data for subsequent use. Currently only supports TIFF"""
        # reworked to separate file logic from UI related logic
        # TODO add directory separation logic

        if self.ext == "tif" or self.ext == "tiff":
            self.import_tiff()
            return True, "Image loaded"
        else:
            status_text = "Status: Image format not-recognised. Please choose TIF/TIFF files."
            return False, status_text

    def import_tiff(self):
        tiff = TiffFile(self.full_name)
        self.tiff = tiff

        meta = tiff.series[0]
        if tiff.is_imagej:
            try:  # if an imagej file, we know where, and can extract the x,y,z resolutions
                x_res = tiff.pages[0].tags["XResolution"].value
                y_res = tiff.pages[0].tags["YResolution"].value

                # X resolution stored as Fraction
                x_res = float(x_res[1]) / float(x_res[0])

                z_res = tiff.imagej_metadata["spacing"]  # .pages[0].imagej_tags['spacing']

                # We're interested in X resolution relative to Z
                self.z_calibration = z_res / x_res

                print("z_scale_factor", self.z_calibration)
            except (AttributeError, KeyError) as ex:
                print("tiff resolution not recognised")
        elif self.tiff.is_ome:
            try:
                x_res = tiff.ome_metadata["Image"]["Pixels"]["PhysicalSizeX"]
                print(x_res)
                y_res = tiff.ome_metadata["Image"]["Pixels"]["PhysicalSizeY"]
                z_res = tiff.ome_metadata["Image"]["Pixels"]["PhysicalSizeZ"]
                # We're interested in X resolution relative to Z
                self.z_calibration = z_res / x_res

                print("z_scale_factor", self.z_calibration)
            except (AttributeError, KeyError) as ex:
                print("tiff resolution not recognised")

        self.order = meta.axes
        for n, axis in enumerate(self.order):
            if axis == "T":
                self.max_t = meta.shape[n] - 1
            if axis == "Z":
                self.max_z = meta.shape[n] - 1
            if axis == "Y":
                self.height = meta.shape[n]
            if axis == "X":
                self.width = meta.shape[n]
            if axis == "S":  # RGB image colour channel
                self.numCH = meta.shape[n]
                # par_obj.tiff_reorder=False
            if axis == "C":
                self.numCH = meta.shape[n]

        self.bitDepth = meta.dtype

        self.array = self.tiff.asarray()  # memmap=True)

        self.tiffarraymax = self.array.max()

        if self.bitDepth in ["uint8", "uint16"]:
            self.tiffarray_typemax = np.iinfo(self.bitDepth).max
            # 12 bit depth
            if self.bitDepth in ["uint16"] and self.tiffarraymax < 4096:
                self.tiffarray_typemax = 4095
        else:
            self.tiffarray_typemax = np.finfo(self.bitDepth).max


class Intermediate_handler:
    """ Wrapper for saving intermediate results to file and keeping track of them """

    def __init__(self, filename):
        try:
            os.remove(filename)
        except:
            pass
        self.plane = 0
        self.tif = None
        self.filename = filename
        self.refs = []
        self.array = None

    def reader(self):
        self.read = True
        self.write = False
        return self

    def writer(self):
        self.write = True
        self.read = False
        return self

    def __enter__(self):
        if self.write:
            self.tif = TiffWriter(self.filename, bigtiff=True, append=True)
        if self.read:
            self.tif = TiffFile(self.filename)
            self.array = self.tif.asarray()  # memmap=True)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.array = None
        self.tif.close()

    def close(self):
        self.array = None

        if self.tif is not None:
            self.tif.close()

    def write_plane(self, data, t, z, f):
        # works plane by plane
        # print 'writing plane' + str((t,z,f))
        # print data.shape

        self.tif.save(data, compress=0, contiguous=True)

        if len(data.shape) > 2:
            for i in range(data.shape[0]):
                self.plane += 1
                self.refs = self.refs + [(t, z + i)]
        else:
            self.plane += 1
            self.refs = self.refs + [(t, z)]

    def read_plane(self, t, z, f):
        # works plane by plane
        planeno = self.refs.index((t, z))
        return np.squeeze(self.tif.asarray(planeno))
