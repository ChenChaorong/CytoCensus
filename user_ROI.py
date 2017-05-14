# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:00:58 2017

@author: martin
"""
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
import struct
import copy

class ROI:
    
    def __init__(self, int_obj,par_obj):
        self.ppt_x = []
        self.ppt_y = []
        self.line = [None]
        self.int_obj = int_obj
        self.complete = False
        self.flag = False
        self.par_obj = par_obj
        self.roi_active =False
        
    def motion_notify_callback(self, event):
        #Mouse moving.
        if event.inaxes and self.roi_active: 
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
                self.par_obj.data_store['roi_stk_x'][self.par_obj.curr_file][self.par_obj.curr_t][self.par_obj.curr_z] = copy.deepcopy(self.ppt_x)
                self.par_obj.data_store['roi_stk_y'][self.par_obj.curr_file][self.par_obj.curr_t][self.par_obj.curr_z] = copy.deepcopy(self.ppt_y)
                #self.flag = False

        
    def button_press_callback(self, event):
        #Mouse clicking
        if event.inaxes and self.roi_active: 
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
                            #if there is a line, check if it is interpolated or not
                            if len(self.ppt_x)==0:
                                self.line = [None] #clear existing lines
                                self.line[0] = Line2D([x,  x],[y, y], marker = 'o')
                                self.ppt_x.append(x)
                                self.ppt_y.append(y) 
                                self.int_obj.plt1.add_line(self.line[0])
                                self.int_obj.canvas1.draw()
                            else:
                                self.line.append(Line2D([self.ppt_x[-1], x], [self.ppt_y[-1], y],marker = 'o'))
                                self.ppt_x.append(x)
                                self.ppt_y.append(y)
                                self.int_obj.plt1.add_line(self.line[-1])
                                self.int_obj.canvas1.draw()
                        self.par_obj.data_store['roi_stk_x'][self.par_obj.curr_file][self.par_obj.curr_t][self.par_obj.curr_z] = copy.deepcopy(self.ppt_x)
                        self.par_obj.data_store['roi_stk_y'][self.par_obj.curr_file][self.par_obj.curr_t][self.par_obj.curr_z] = copy.deepcopy(self.ppt_y)
                        
    def button_release_callback(self, event):
        self.flag = False
    def complete_roi(self):
        print 'ROI completed.'
        self.complete = True
        self.draw_ROI()
        self.reparse_ROI(self.par_obj.curr_z)
        #self.find_the_inside()
    def clear_ROI(self):
        self.complete = False
        imno=self.par_obj.curr_file
        tpt=self.par_obj.curr_t
        zslice=self.par_obj.curr_z
        self.par_obj.data_store['roi_stk_x'][imno][tpt].pop(zslice)
        self.line = [None]
        self.draw_ROI()
        
    def draw_ROI(self):
        #redraws the regions in the current slice.
        drawn = False
        imno=self.par_obj.curr_file
        tpt=self.par_obj.curr_t
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
        for bt in self.par_obj.data_store['roi_stkint_x'][self.par_obj.curr_file][self.par_obj.curr_t]:
            #If there is a matching hand-drawn one we 
            to_approve.append(bt)
                
        for cv in to_approve:
            del self.par_obj.data_store['roi_stkint_x'][self.par_obj.curr_file][self.par_obj.curr_t][cv]
            del self.par_obj.data_store['roi_stkint_y'][self.par_obj.curr_file][self.par_obj.curr_t][cv]

        for cd in self.par_obj.data_store['roi_stk_x'][self.par_obj.curr_file][self.par_obj.curr_t]:
                
            #First we make a local copy. We make a deep copy because python uses pointers and we don't want to change the original.
            cppt_x = copy.deepcopy(self.par_obj.data_store['roi_stk_x'][self.par_obj.curr_file][self.par_obj.curr_t][cd])
            cppt_y = copy.deepcopy(self.par_obj.data_store['roi_stk_y'][self.par_obj.curr_file][self.par_obj.curr_t][cd])

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
            self.par_obj.data_store['roi_stkint_x'][self.par_obj.curr_file][self.par_obj.curr_t][cd] = nppt_x
            self.par_obj.data_store['roi_stkint_y'][self.par_obj.curr_file][self.par_obj.curr_t][cd] = nppt_y
            

        
        #self.int_obj.plt1.plot(nppt_x,nppt_y,'-')
        #self.int_obj.canvas1.draw()

    def interpolate_ROI(self):
        imno=self.par_obj.curr_file
        #We want to interpolate between frames. So we make sure the frames are in order.
        tosort = []
        for bt in self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.curr_t]:
            tosort.append(bt)
        sortd = np.sort(np.array(tosort))

        #For each of the slices which have been drawn in
        for b in range(0,sortd.shape[0]-1):
            ab = sortd[b]
            ac = sortd[b+1]
            #We assess if there are any slices which are empty within the range.
            if (ac-ab) > 1:
                #We then copy the points. (we use deepcopy because python uses pointers by defualt and we don't want to edit the original)
                lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.curr_t][ab])
                lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.curr_t][ab])
                upx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.curr_t][ac])
                upy = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.curr_t][ac])
                
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


                #The list can be reversed depending on how the ROI was drawn (clockwise or anti-clockwise)
                lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.curr_t][ab])[::-1]
                lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.curr_t][ab])[::-1]

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
                    lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.curr_t][ab])
                    lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.curr_t][ab])
                else:  
                    min_dis = np.argmin(np.array(minfn2))
                    lrx = copy.deepcopy(self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.curr_t][ab])[::-1]
                    lry = copy.deepcopy(self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.curr_t][ab])[::-1]

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
                    self.par_obj.data_store['roi_stkint_x'][imno][self.par_obj.curr_t][b] = nppt_x
                    self.par_obj.data_store['roi_stkint_y'][imno][self.par_obj.curr_t][b] = nppt_y
                    
            self.int_obj.canvas1.draw()
    def interpolate_ROI_in_time(self):
        imno=self.par_obj.curr_file

        time_pts_with_roi = []
        for it_time_pt in self.par_obj.tpt_list:
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
            ppt_x = self.par_obj.data_store['roi_stkint_x'][self.par_obj.curr_file][self.par_obj.curr_t][self.par_obj.curr_z]
            ppt_y = self.par_obj.data_store['roi_stkint_y'][self.par_obj.curr_file][self.par_obj.curr_t][self.par_obj.curr_z]
            imRGB = np.array(self.par_obj.dv_file.get_frame(self.par_obj.curr_z))
            
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
            
    def area(self):
            ppt_x = self.par_obj.data_store['roi_stkint_x'][self.par_obj.curr_file][self.par_obj.curr_t][self.par_obj.curr_z]
            ppt_y = self.par_obj.data_store['roi_stkint_y'][self.par_obj.curr_file][self.par_obj.curr_t][self.par_obj.curr_z]
            
            pot = [] 
            for i in range(0,ppt_x.__len__()):
                pot.append([ppt_x[i],ppt_y[i]])

            p = Path(pot)
            counter=0
            for y in range(0,self.par_obj.height):
                for x in range(0,self.par_obj.width):
                    if p.contains_point([x,y]) == True:
                        counter+=1
            return counter

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
