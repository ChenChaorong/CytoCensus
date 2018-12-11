#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from tifffile import TiffFile, imsave, TiffWriter#Install with pip install tifffile.
import os
"""
Created on Fri Feb 17 19:48:58 2017

@author: martin
"""
'''
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
would cause problems without some significant rewrites
'''


#create new file_handler for each file
class File_handler(object):
    def __init__(self,file_path):
        #store high level file data and metadata
        self.full_name=file_path

        self.base_name=os.path.basename(file_path)
        self.path=os.path.dirname(file_path)

        self.ext = self.base_name.split(".")[-1]
        self.name = self.base_name.split(".")[0]

        self.array = [] #memmap object l
        self.z_calibration = 1
        self.order={} #ordering of tiff objects
        #default file extents

        self.max_t = 0
        self.max_z = 0
        self.numCH = 0

        self.import_file()

    def get_tiff_slice(self,tpt=[0],zslice=[0],x=[0],y=[0],c=[0]):
        #deal with different TXYZC orderings. Always return TZYXC
        #handles lists and ints nicely
        if type(zslice) is not list: zslice = [zslice]
        if type(zslice[0]) is list: zslice = zslice[0]

        alist=[]
        blist=[]
        for n,b in enumerate(self.order):
            if b=='T':
                alist.append(tpt)
                blist.append(n)
        for n,b in enumerate(self.order):
            if b=='Z':
                alist.append(zslice)
                blist.append(n)
        for n,b in enumerate(self.order):
            if b=='Y':
                alist.append(y)
                blist.append(n)
        for n,b in enumerate(self.order):
            if b=='X':
                alist.append(x)
                blist.append(n)

        for n,b in enumerate(self.order):
            if b=='C' or b=='S':
                alist.append(c)
                blist.append(n)

        tiff2=self.array.transpose(blist)

        if self.order.__len__()==5:
            #tiff=np.squeeze(tiff2[alist[0],:,:,:,:][:,alist[1],:,:,:][:,:,alist[2],:,:][:,:,:,alist[3],:][:,:,:,:,alist[4]])
            tiff=np.squeeze(tiff2[np.ix_(alist[0],alist[1],alist[2],alist[3],alist[4])])
        elif self.order.__len__()==4:
            #tiff=np.squeeze(tiff2[alist[0],:,:,:][:,alist[1],:,:][:,:,alist[2],:][:,:,:,alist[3]])
            tiff=np.squeeze(tiff2[np.ix_(alist[0],alist[1],alist[2],alist[3])])

        elif self.order.__len__()==3:
            #tiff=np.squeeze(tiff2[alist[0],:,:][:,alist[1],:][:,:,alist[2]])
            tiff=np.squeeze(tiff2[np.ix_(alist[0],alist[1],alist[2])])
        elif self.order.__len__()==2:
            #tiff=np.squeeze(tiff2[alist[0],:][:,alist[1]])
            tiff=np.squeeze(tiff2[np.ix_(alist[0],alist[1])])
        return tiff

    def close(self):
        self.array =[]
        self.Tiff.close()

    def import_file(self):

        #loads in Tiff image data for subsequent use
        #reworked to separate file logic from UI related logic
        #TODO add directory separation logic


        if self.ext == 'tif' or self.ext == 'tiff':
            self.import_tiff()
            return True, 'Image loaded'
        else:
            statusText = 'Status: Image format not-recognised. Please choose either png or TIFF files.'
            return False, statusText

    def import_tiff(self):
            self.Tiff = TiffFile(self.full_name)

            meta = self.Tiff.series[0]

            try: #if an imagej file, we know where, and can extract the x,y,z
            #if 1==1:
                x = self.Tiff.pages[0].tags.x_resolution.value
                y = self.Tiff.pages[0].tags.y_resolution.value
                if x!=y: raise Exception('x resolution different to y resolution')# if this isn't true then something is wrong
                x_res=float(x[1])/float(x[0])

                z=self.Tiff.pages[0].imagej_tags['spacing']

                self.z_calibration = z/x_res

                print('z_scale_factor', self.z_calibration)
            except:
                #might need to modify this to work with OME-TIFFs
                print 'tiff resolution not recognised'

            self.order = meta.axes
            for n,b in enumerate(self.order):
                    if b == 'T':
                        self.max_t = meta.shape[n]-1
                    if b == 'Z':
                        self.max_z = meta.shape[n]-1
                    if b == 'Y':
                        self.height = meta.shape[n]
                    if b == 'X':
                        self.width = meta.shape[n]
                    if b == 'S':
                        self.numCH = meta.shape[n]
                        #par_obj.tiff_reorder=False
                    if b == 'C':
                        self.numCH = meta.shape[n]

            self.bitDepth = meta.dtype

            self.array=self.Tiff.asarray(memmap=True)

            self.tiffarraymax = self.array.max()

            if self.bitDepth in ['uint8','uint16']:
                self.tiffarray_typemax=np.iinfo(self.bitDepth).max
                #12 bit depth
                if self.bitDepth in ['uint16'] and self.tiffarraymax<4096:
                    self.tiffarray_typemax=4095          
            else:
                self.tiffarray_typemax=np.finfo(self.bitDepth).max

class Intermediate_handler():
    """ Wrapper for saving intermediate results to file and keeping track of them """
    def  __init__(self,filename):
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
            self.tif = TiffWriter(self.filename, bigtiff=True,append=True)
        if self.read:
            self.tif = TiffFile(self.filename)
            self.array = self.tif.asarray(memmap=True)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
            self.array = None
            self.tif.close()
                

    def close(self):
            self.array = None
            
            if self.tif is not None:
                self.tif.close()

    def write_plane(self, data,t,z,f):
        #works plane by plane
        #print 'writing plane' + str((t,z,f))
        #print data.shape
            
        self.tif.save(data, compress = 0, contiguous = True)
        
        if len(data.shape)>2:
            for i in range (data.shape[0]):
                self.plane += 1
                self.refs = self.refs + [(t,z+i)]
        else:
            self.plane += 1
            self.refs = self.refs + [(t,z)]
    
    def read_plane(self,t,z,f):
        #works plane by plane
        planeno = self.refs.index((t,z))
        return np.squeeze(self.tif.asarray(planeno))
        
        
