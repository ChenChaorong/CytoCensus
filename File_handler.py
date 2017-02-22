#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from tifffile import TiffFile, imsave #Install with pip install tifffile.

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
        self.file_name=file_path
        self.file_array = []
        self.tiffarray = [] #memmap object list
        self.z_calibration = 1
        self.order={} #ordering of tiff objects
        #default file extents

        self.total_time_pt = 1
        self.max_zslices = 0

        self.curr_file=0
        self.curr_z = 0
        self.time_pt = 0 #TODO make these names consistent
        self.numCH = 0
        
        import_file(self, file_path)
        
    def get_tiff_slice(par_obj,tpt=[0],zslice=[0],x=[0],y=[0],c=[0]):
        #deal with different TXYZC orderings. Always return TZYXC
        #handles lists and ints nicely
        if type(zslice) is not list: zslice = [zslice]
        if type(zslice[0]) is list: zslice = zslice[0]

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
        
        tiff2=par_obj.tiffarray.transpose(blist)
        
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
        
        return tiff
    def close(self, file_path):
        self.array =[]
        self.Tiff.close()
        
    def import_file(self, file_path):

        #loads in Tiff image data for subsequent use
        #reworked to separate file logic from UI related logic
        #TODO add directory separation logic
        self.path=file_path
        self.ext = self.path.split(".")[-1]
        self.name = self.path.split(".")[0].split("/")[-1]

        if self.file_ext == 'tif' or self.file_ext == 'tiff':
            self.import_file.import_tiff()
            return True, 'Image loaded'
        else:
            statusText = 'Status: Image format not-recognised. Please choose either png or TIFF files.'
            return False, statusText
        
        def import_tiff(self):
                self.Tiff = TiffFile(self.path)
                self.tiffarraymax
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
                            self.total_time_pt = meta.shape[n]
                        if b == 'Z':
                            self.max_zslices = meta.shape[n]
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
                
                if self.bitDepth in ['uint8','uint16','uint32']:
                    self.tiffarray_typemax=np.iinfo(self.bitDepth).max
                    if self.bitDepth in ['uint16'] and self.tiffarraymax<4096:
                        self.tiffarray_typemax=4095
                else:
                    self.tiffarray_typemax=np.finfo(self.bitDepth).max

                
    def import_data_fn(par_obj,file_array,file_array_offset=0):
        """Function which loads in Tiff stack or single png file to assess type."""
        
        #careful with use of non-zero offset. Intended primarily for use in validation
        prevExt = [] 
        prevBitDepth=[] 
        prevNumCH =[]
        par_obj.numCH = 0
        #par_obj.total_time_pt = 0
        
        par_obj.max_file= file_array.__len__()
            
        par_obj.filehandlers=[None]*(par_obj.max_file+file_array_offset)
        
        for imno,imfile in enumerate(file_array):
                
            par_obj.filehandlers[imno] = File_handler(imfile)
            #currently doesn't check if multiple filetypes, on the basis only loads tiffs
            #check number of channels is consistent
            if imno==0:
                par_obj.numCH = par_obj.filehandlers[imno].numCH
            elif par_obj.numCH == par_obj.filehandlers[imno].numCH:
                pass
            else:
                statusText = 'Different number of image channels in the selected images'
                raise Exception(statusText)# if this isn't true then something is wrong
            #check height and width match
            if imno==0:
                par_obj.ori_height = par_obj.filehandlers[imno].height
            elif par_obj.ori_height == par_obj.filehandlers[imno].height:
                pass
            else:
                statusText = 'Different image size in the selected images'
                raise Exception(statusText)# if this isn't true then something is wrong
                
            if imno==0:
                par_obj.ori_width = par_obj.filehandlers[imno].width
            elif par_obj.ori_width == par_obj.filehandlers[imno].width:
                pass
            else:
                statusText = 'Different image size in the selected images'
                raise Exception(statusText)# if this isn't true then something is wrong
            #check bit depth matches
            if imno==0:
                par_obj.bitDepth = par_obj.filehandlers[imno].bitDepth
                par_obj.tiffarray_typemax=par_obj.filehandlers[imno].tiffarray_typemax
                                                  
            elif par_obj.bitDepth == par_obj.filehandlers[imno].bitDepth:
                pass
            else:
                statusText = 'Different image bit depth in the selected images'
                raise Exception(statusText)# if this isn't true then something is wrong

        #Prepare RGB example image
        x = range(0,par_obj.ori_width,par_obj.resize_factor)
        y = range(0,par_obj.ori_height,par_obj.resize_factor)
        imRGB = par_obj.filehandlers[0].get_tiff_slice([0],[0],x,y,range(par_obj.numCH))
        
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