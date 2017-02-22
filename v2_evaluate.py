#Script for running evaluation mode
import pylab
from PyQt4 import QtGui, QtCore, Qt, QtWebKit
try:
    from PyQt4.QtCore import QString as QString
except:
    QString=str
import time
import vigra
import scipy
from scipy.ndimage import filters
from scipy.sparse.csgraph import _validation
from sklearn.ensemble import ExtraTreesRegressor
import thread
import random
import csv
import cPickle as pickle
import datetime
import errno
import os
import numpy as np
import copy
import datetime
import functools
from scipy.special import _ufuncs_cxx
import sklearn.utils.lgamma
from gnu import return_license
from matplotlib.path import Path
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import pdb
from common_navigation import *
from parameter_object import parameterClass

#Apparently matplotlib slows the loading drammatically due to a font cache issue. This resolves it.
try:
    #mac location.
    os.remove(os.path.expanduser('~')+'/.matplotlib/fontList.cache')
    print 'removing matplotlib cache'
except:
    pass
try:
    #Alternate location.
    os.remove(os.path.expanduser('~')+'/.cache/matplotlib')
    print 'removing matplotlib cache'
except:
    pass

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import v2_functions as v2
"""QBrain Software v0.1

    Copyright (C) 2016  Dominic Waithe Martin Hailstone

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
   

class Eval_load_im_win(QtGui.QWidget):
    def __init__(self,par_obj):
        super(Eval_load_im_win, self).__init__()
        """Setups the load image interface"""

        #Stops reinitalise of text which can produce buggy text.
        #if par_obj.evalLoadImWin_loaded !=True:
        vbox0 = QtGui.QVBoxLayout()
        self.setLayout(vbox0)

        #Load images button
        self.loadImages_button_panel = QtGui.QHBoxLayout()
        self.loadImages_button = QtGui.QPushButton("Add Images")
        self.loadImages_button.move(20, 20)
        self.loadImages_button_panel.addWidget(self.loadImages_button)
        self.loadImages_button_panel.addStretch()

        about_btn = QtGui.QPushButton('About')
        about_btn.clicked.connect(lambda: on_about(self))
        self.loadImages_button_panel.addWidget(about_btn)


        #Table widget which displays
        self.modelTabIm_panel = QtGui.QHBoxLayout()
        self.modelTabIm = QtGui.QTableWidget()
        self.modelTabIm.setRowCount(1)
        self.modelTabIm.setColumnCount(4)
        self.modelTabIm.setColumnWidth(0,125)
        self.modelTabIm.resize(550,200)
        self.modelTabIm.setHorizontalHeaderLabels(QString(",Image name, Range, Path").split(","))
        self.modelTabIm.hide()
        self.modelTabIm.setEditTriggers( QtGui.QTableWidget.NoEditTriggers )
        self.modelTabIm_panel.addWidget(self.modelTabIm)
        self.modelTabIm_panel.addStretch()

        #Starts file dialog calss
        self.imLoad = File_Dialog(par_obj,self,self.modelTabIm)
        self.imLoad.type = 'im'
        self.loadImages_button.clicked.connect(self.imLoad.showDialog)
        
        vbox0.addLayout(self.loadImages_button_panel)
        vbox0.addLayout(self.modelTabIm_panel)
        vbox0.addStretch()
        
        #Move to training button.
        self.selIntButton_panel = QtGui.QHBoxLayout()
        self.selIntButton = QtGui.QPushButton("Goto Model Import")
        self.selIntButton.clicked.connect(self.goto_model_import)
        self.selIntButton.move(20, 520)
        self.selIntButton.setEnabled(False)
        self.selIntButton_panel.addWidget(self.selIntButton)
        self.selIntButton_panel.addStretch()
        vbox0.addLayout(self.selIntButton_panel)
        

        #Status bar to report whats going on.
        
        self.image_status_text = QtGui.QStatusBar()
        self.image_status_text.showMessage('Status: Highlight training images in folder. ')
        vbox0.addWidget(self.image_status_text)

    def goto_model_import(self):
            win_tab.setCurrentWidget(evalLoadModelWin)
            #Checks the correct things are disabled.
            evalLoadModelWin.gotoEvalButton.setEnabled(False)
            evalImWin.prev_im_btn.setEnabled(False)
            evalImWin.next_im_btn.setEnabled(False)
            evalImWin.eval_im_btn.setEnabled(False)
            par_obj.eval_load_im_win_eval = False

class Eval_load_model_win(QtGui.QWidget):
    """Interface which allows selection of model."""
    def __init__(self,par_obj):
        
        super(Eval_load_model_win,self).__init__()
        
        #The main layout
        box = QtGui.QVBoxLayout()
        self.setLayout(box)
        
        hbox0 = QtGui.QHBoxLayout()
        box.addLayout(hbox0)
        #The two principle columns
        vbox0 = QtGui.QVBoxLayout()
        vbox1 = QtGui.QVBoxLayout()
        hbox0.addLayout(vbox0)
        hbox0.addLayout(vbox1)

        #Display available models.
        #Find all files in folder with serielize prefix.
        
        files = os.listdir(par_obj.forPath)
        filesRF =[]
        for b in range(0,files.__len__()):
            
            if(os.path.splitext(files[b])[1] == '.pkl'):
                filesRF.append(os.path.splitext(files[b])[0] )
            if(os.path.splitext(files[b])[1] == '.mdla'):
                filesRF.append(os.path.splitext(files[b])[0] )
        

        filesLen =filesRF.__len__()

      
        self.modelTabFor = QtGui.QTableWidget()
        self.modelTabFor.setRowCount(1)
        self.modelTabFor.setColumnCount(3)
        self.modelTabFor.setColumnWidth(2,200)
        self.modelTabFor.resize(600,500)
        self.modelTabFor.setHorizontalHeaderLabels(QString(",model name, date and time saved").split(","))
        self.modelTabFor.setEditTriggers( QtGui.QTableWidget.NoEditTriggers )

        vbox0.addWidget(self.modelTabFor)

        #Button for going to model evaluation.
        self.gotoEvalButton = QtGui.QPushButton("Goto Image Evaluation")
        self.gotoEvalButton.setEnabled(False)
        self.gotoEvalButton.clicked.connect(self.gotoEvalButton_fn)
        
        vbox0.addWidget(self.gotoEvalButton)

        #Status text.
        self.image_status_text = QtGui.QStatusBar()
        self.image_status_text.resize(300,20)
        self.image_status_text.setStyleSheet("QLabel {  color : green }")
        self.image_status_text.showMessage('Status: Please click a model from above and then click \'Load Model\'. ')

        #The second column.
        self.modelIm_panel = QtGui.QHBoxLayout()

        self.figure1 = Figure(figsize=(3, 3), dpi=100)
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure1.patch.set_facecolor('grey')
        
        
        self.plt1 = self.figure1.add_subplot(1, 1, 1)
        im_RGB = np.zeros((300, 300))
        #Makes sure it spans the whole figure.
        self.figure1.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)
        self.plt1.imshow(im_RGB)

        #Removes the tick labels
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])

        

        vbox1.addWidget(self.canvas1)
        


        self.modelImTxt1_panel = QtGui.QHBoxLayout()
        self.modelImTxt2_panel = QtGui.QHBoxLayout()
        self.modelImTxt3_panel = QtGui.QHBoxLayout()
        self.modelImTxt4_panel = QtGui.QHBoxLayout()
        self.modelImTxt5_panel = QtGui.QHBoxLayout()
        self.modelImTxt6_panel = QtGui.QHBoxLayout()

        self.modelImTxt1 = QtGui.QLabel()
        self.modelImTxt2 = QtGui.QLabel()
        self.modelImTxt3 = QtGui.QLabel()
        self.modelImTxt4 = QtGui.QLabel()
        self.modelImTxt5 = QtGui.QLabel()
        self.modelImTxt6 = QtGui.QLabel()

        self.modelImTxt1.setText('Name: ')
        self.modelImTxt1.resize(400,25)
        self.modelImTxt1_panel.addWidget(self.modelImTxt1)
        self.modelImTxt1_panel.addStretch()
        
        self.modelImTxt2.setText('Description: ')
        self.modelImTxt2.resize(400,25)
        self.modelImTxt2_panel.addWidget(self.modelImTxt2)
        self.modelImTxt2_panel.addStretch()
        
        self.modelImTxt3.setText('Sigma Data: ')
        self.modelImTxt3.resize(400,25)
        self.modelImTxt3_panel.addWidget(self.modelImTxt3)
        self.modelImTxt3_panel.addStretch()
        
        self.modelImTxt4.setText('Feature Scale: ')
        self.modelImTxt4.resize(400,25)
        self.modelImTxt4_panel.addWidget(self.modelImTxt4)
        self.modelImTxt4_panel.addStretch()
        
        self.modelImTxt5.setText('Feature Type: ')
        self.modelImTxt5.resize(400,25)
        self.modelImTxt5_panel.addWidget(self.modelImTxt5)
        self.modelImTxt5_panel.addStretch()
        
        self.modelImTxt6.setText('Channels: ')
        self.modelImTxt6.resize(400,25)
        self.modelImTxt6_panel.addWidget(self.modelImTxt6)
        self.modelImTxt6_panel.addStretch()

        vbox1.addLayout(self.modelImTxt1_panel)
        vbox1.addLayout(self.modelImTxt2_panel)
        vbox1.addLayout(self.modelImTxt3_panel)
        vbox1.addLayout(self.modelImTxt4_panel)
        vbox1.addLayout(self.modelImTxt5_panel)
        vbox1.addLayout(self.modelImTxt6_panel)
        vbox1.addStretch()

        
        
        
        
        c =0
        for i in range(0,filesRF.__len__()):
                
                strFn = filesRF[i].split('_')
                
                
                
                if(str(strFn[0]) == 'pv1.3'):


                    self.modelTabFor.setRowCount(c+1)
                    btn = loadModelBtn(par_obj,self,self.modelTabFor,i,filesRF[i])

                    btn.setText('Click to View')
                    self.modelTabFor.setCellWidget(c, 0, btn)

                    text1 = QtGui.QLabel(self.modelTabFor)
                    text1.setText(str(' '+strFn[1]))
                    self.modelTabFor.setCellWidget(c,1,text1)

                    text2 = QtGui.QLabel(self.modelTabFor)
                    text2.setText(str(' '+strFn[2]))
                    self.modelTabFor.setCellWidget(c,2,text2)
                    c= c+1
                if str(strFn[0]) == 'pv20':
                    

                    self.modelTabFor.setRowCount(c+1)
                    btn = loadModelBtn(par_obj,self,self.modelTabFor,i,filesRF[i])

                    btn.setText('Click to View')
                    self.modelTabFor.setCellWidget(c, 0, btn)

                    text1 = QtGui.QLabel(self.modelTabFor)
                    text1.setText(str(' '+strFn[2]))
                    self.modelTabFor.setCellWidget(c,1,text1)

                    text2 = QtGui.QLabel(self.modelTabFor)
                    text2.setText(str(' '+datetime.datetime.fromtimestamp(float(strFn[1])).strftime('%Y-%m-%d %H:%M:%S')))
                    self.modelTabFor.setCellWidget(c,2,text2)
                    c= c+1

        box.addWidget(self.image_status_text)
    def loadModelFn(self,par_obj, fileName):
        """Shows details of the model when loaded"""

        par_obj.selectedModel = par_obj.forPath+fileName
        par_obj.evaluated = False
        self.image_status_text.showMessage('Status: Loading previously trained model. ')
        app.processEvents()

        ver = par_obj.selectedModel.split('/')[-1][0:5]
        if(ver =='pv20_'):
            save_file = pickle.load(open(par_obj.selectedModel+str('.mdla'), "rb"))
            print(par_obj.selectedModel+str('.mdla'))

            par_obj.modelName = save_file["name"]
            par_obj.modelDescription = save_file["description"]
            par_obj.RF = save_file["model"]
            local_time = save_file["date"]
            par_obj.M = save_file["M"]
            par_obj.c = save_file["c"]
            par_obj.feature_type = save_file["feature_type"]
            par_obj.feature_scale = save_file["feature_scale"]
            par_obj.sigma_data = save_file["sigma_data"]
            par_obj.ch_active = save_file["ch_active"]
            par_obj.limit_ratio_size = save_file["limit_ratio_size"]
            par_obj.max_depth = save_file["max_depth"]
            par_obj.min_samples_split = save_file["min_samples"]
            par_obj.min_samples_leaf = save_file["min_samples_leaf"]
            par_obj.max_features = save_file["max_features"]
            par_obj.num_of_tree = save_file["num_of_tree"]
            
            par_obj.resize_factor = save_file["resize_factor"]
            par_obj.min_distance = save_file["min_distance"]
            par_obj.abs_thr = save_file["abs_thr"]
            par_obj.rel_thr = save_file["rel_thr"]


            #par_obj.gt_vec = save_file["gt_vec"]
            #par_obj.error_vec = save_file["error_vec"]
            save_im = save_file["imFile"]
            self.image_status_text.showMessage('Status: Model loaded. ')
            success = True

            #Some basic image checking.
            if 1==0:#par_obj.numCH <par_obj.ch_active.__len__():
                success = False
                self.image_status_text.showMessage('Status: Model is incompatible. There were more channels in the original images on which the training was performed than in the loaded images. ')
            if par_obj.file_ext != save_file["file_ext"]:
                success = False
                self.image_status_text.showMessage('Status: Model is incompatible. The original images on which the training was performed are of a different format to the loaded images. ')
            

        if(ver =='pv1.3'):
            #Load in parameters from file.
            par_obj.file_ext,par_obj.RF,par_obj.sigma_data,par_obj.feature_scale,par_obj.feature_type,par_obj.ch_active,par_obj.modelName,par_obj.modelDescription = pickle.load(open(par_obj.selectedModel+str('.pkl'), "rb"))
            
            #save_im =cv2.imread(par_obj.selectedModel+"im.png")
            self.image_status_text.showMessage('Status: Model loaded. ')
            #Display details about model.
            success, statusText = v2.import_data_fn(par_obj,par_obj.file_array)

        if success == True:
            self.modelImTxt1.setText('Name: '+str(par_obj.modelName))
            self.modelImTxt2.setText('Description: '+str(par_obj.modelDescription ))
            self.modelImTxt3.setText('Sigma Data: '+str(par_obj.sigma_data))
            self.modelImTxt4.setText('Feature Scale: '+str(par_obj.feature_scale))
            self.modelImTxt5.setText('Feature Type: '+str(par_obj.feature_type))
            self.modelImTxt6.setText('Channels: '+str(par_obj.ch_active))
            self.plt1.imshow(save_im/np.max(save_im))
            self.canvas1.draw()
            self.gotoEvalButton.setEnabled(True)
            evalImWin.eval_im_btn.setEnabled(True)
            par_obj.eval_load_im_win_eval = False
    '''
    def process_imgs(self):
        imgs =[]
        gt_im_sing_chgs =[]
        #fmStr = self.linEdit_Frm.text()
        
        par_obj.height = par_obj.ori_height/par_obj.resize_factor
        par_obj.width = par_obj.ori_width/par_obj.resize_factor
        
        par_obj.frames_2_load ={}
        par_obj.left_2_calc =[]
        par_obj.saved_ROI =[]
        par_obj.saved_dots=[]
        par_obj.curr_z = 0
        par_obj.eval_load_im_win_eval = False
        
        
        #Now we commit our options to our imported files.
        if par_obj.file_ext == 'png':
            for i in range(0,par_obj.file_array.__len__()):
                par_obj.left_2_calc.append(i)
                par_obj.frames_2_load[i] = [0]
            
        elif par_obj.file_ext =='tiff' or par_obj.file_ext =='tif':
            if par_obj.max_zslices>1:
                for i in range(0,par_obj.file_array.__len__()):
                    par_obj.left_2_calc.append(i)
                    par_obj.frames_2_load[i] = range(0,par_obj.max_zslices)
                    
                    if par_obj.total_time_pt> 0:
                        tmStr = par_obj.input_range.text()
                        par_obj.time_pt_list = np.array(list(self.hyphen_range(tmStr)))-1
                    else:
                        par_obj.time_pt_list = [0]
                    for time_pt in par_obj.time_pt_list:
                        par_obj.data_store[time_pt] ={}
                        par_obj.data_store[time_pt]['dense_arr'] = {}
                        par_obj.data_store[time_pt]['feat_arr'] = {}
                        par_obj.data_store[time_pt]['pred_arr'] = {}
                        par_obj.data_store[time_pt]['sum_pred'] = {}
                        par_obj.data_store[time_pt]['maxi_arr'] = {}
                        par_obj.data_store[time_pt]['pts'] ={}
                        par_obj.data_store[time_pt]['roi_stkint_x'] ={}
                        par_obj.data_store[time_pt]['roi_stkint_y'] ={}
                        par_obj.data_store[time_pt]['roi_stk_x'] ={}
                        par_obj.data_store[time_pt]['roi_stk_y'] ={}
                    #try:
                    #    np.array(list(self.hyphen_range(fmStr)))-1
                    #    par_obj.frames_2_load[i] = np.array(list(self.hyphen_range(fmStr)))-1
                    #except:
                    #    self.image_status_text.showMessage('Status: The supplied range of image frames is in the wrong format. Please correct and click confirm images.')
                    #    return
            else:
                for i in range(0,par_obj.file_array.__len__()):
                    par_obj.left_2_calc.append(i)
                    par_obj.frames_2_load[i] = [0]
        elif par_obj.file_ext =='oib'or  par_obj.file_ext == 'oif':
            if par_obj.max_zslices>1:
                for i in range(0,par_obj.file_array.__len__()):
                    par_obj.left_2_calc.append(i)
                    par_obj.frames_2_load[i] = range(0,par_obj.max_zslices)
                    
                    if par_obj.total_time_pt> 0:
                        tmStr = par_obj.input_range.text()
                        par_obj.time_pt_list = np.array(list(self.hyphen_range(tmStr)))-1
                    else:
                        par_obj.time_pt_list = [0]
                    for time_pt in par_obj.time_pt_list:
                        par_obj.data_store[time_pt] ={}
                        par_obj.data_store[time_pt]['dense_arr'] = {}
                        par_obj.data_store[time_pt]['feat_arr'] = {}
                        par_obj.data_store[time_pt]['pred_arr'] = {}
                        par_obj.data_store[time_pt]['sum_pred'] = {}
                        par_obj.data_store[time_pt]['maxi_arr'] = {}
                        par_obj.data_store[time_pt]['pts'] ={}
                        par_obj.data_store[time_pt]['roi_stkint_x'] ={}
                        par_obj.data_store[time_pt]['roi_stkint_y'] ={}
                        par_obj.data_store[time_pt]['roi_stk_x'] ={}
                        par_obj.data_store[time_pt]['roi_stk_y'] ={}
                #self.image_status_text.showMessage('Status: Loading Images.')
                #v2.im_pred_inline_fn(par_obj, self)
               
            else:
                for i in range(0,par_obj.file_array.__len__()):
                    par_obj.left_2_calc.append(i)
                    par_obj.frames_2_load[i] = [0]
                v2.im_pred_inline_fn(par_obj, self)
                
        count = 0
        for b in par_obj.left_2_calc:
            frames =par_obj.frames_2_load[b]
            for i in frames:
                count = count+1
                
        par_obj.test_im_start= 0
        par_obj.test_im_end= count
        

        win_tab.setCurrentWidget(evalImWin)
        evalImWin.prev_im_btn.setEnabled(True)
        evalImWin.next_im_btn.setEnabled(True)

        #Now that we have correct resize factor from the model import.
        v2.import_data_fn(par_obj,par_obj.file_array)
        
        evalImWin.loadTrainFn()
        v2.load_and_initiate_plots(par_obj, evalImWin)
        #v2.eval_goto_img_fn(par_obj.curr_z,par_obj,evalImWin)
    '''
    def gotoEvalButton_fn(self):
        for i in range(par_obj.file_array.__len__()):
            par_obj.frames_2_load = range(0,par_obj.max_zslices) #TODO make max_zslices  a list
            if par_obj.total_time_pt==[]:
                par_obj.total_time_pt=1
            par_obj.time_pt_list = range(0,par_obj.total_time_pt) #TODO make sure this matches up and don't have a +1 error

        v2.processImgs(self,par_obj)
        par_obj.eval_load_im_win_eval = False
        win_tab.setCurrentWidget(evalImWin)
        evalImWin.prev_im_btn.setEnabled(True)
        evalImWin.next_im_btn.setEnabled(True)
        

        #Now that we have correct resize factor from the model import.
        v2.import_data_fn(par_obj,par_obj.file_array)
        
        evalImWin.loadTrainFn()
        v2.load_and_initiate_plots(par_obj, evalImWin)
    def hyphen_range(self,s):
        """ yield each integer from a complex range string like "1-9,12, 15-20,23"

        >>> list(hyphen_range('1-9,12, 15-20,23'))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 23]

        >>> list(hyphen_range('1-9,12, 15-20,2-3-4'))
        Traceback (most recent call last):
            ...
        ValueError: format error in 2-3-4
        """
        for x in s.split(','):
            elem = x.split('-')
            if len(elem) == 1: # a number
                yield int(elem[0])
            elif len(elem) == 2: # a range inclusive
                start, end = map(int, elem)
                for i in xrange(start, end+1):
                    yield i
            else: # more than one hyphen
                raise ValueError('format error in %s' % x)
class Eval_disp_im_win(QtGui.QWidget):
    """ Arranges widget to visualise the input images and ouput prediction. """
    def __init__(self,par_obj):
        super(Eval_disp_im_win, self).__init__()
        #Sets up the figures for displaying images.
        self.figure1 = Figure(figsize=(8, 8), dpi=100)
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure1.patch.set_facecolor('grey')
        toolbar = NavigationToolbar(self.canvas1,self)
        
        self.plt1 = self.figure1.add_subplot(1, 1, 1)
        im_RGB = np.zeros((512, 512))
        #Makes sure it spans the whole figure.
        self.figure1.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)
        
        self.plt1.imshow(im_RGB)

        #Removes the tick labels
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])
        self.cursor = v2.ROI(self,par_obj)
        
        #Initialises the second figure.
        self.figure2 = Figure(figsize=(8, 8), dpi=100)
        self.canvas2 = FigureCanvas(self.figure2)
        self.figure2.patch.set_facecolor('grey')
        self.plt2 = self.figure2.add_subplot(1, 1, 1)
        
        #Makes sure it spans the whole figure.
        self.figure2.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)
        self.plt2.imshow(im_RGB)
        self.plt2.set_xticklabels([])
        self.plt2.set_yticklabels([])
        
        #The ui for training
        self.count_txt = QtGui.QLabel()
        self.image_num_txt = QtGui.QLabel()
        box = QtGui.QVBoxLayout()
        self.setLayout(box)
        
        #Widget containing the top panel.
        top_panel = QtGui.QHBoxLayout()
        
        #Top left and right widget panels
        top_left_panel = QtGui.QGroupBox('Basic Controls')
        top_middle_panel = QtGui.QGroupBox('ROI Controls')
        top_right_panel = QtGui.QGroupBox('Advanced Controls')

        
        #Grid layouts for the top and left panels.
        self.top_left_grid = QtGui.QGridLayout()
        self.top_right_grid = QtGui.QGridLayout()
        
        self.top_left_grid.setSpacing(2)
        self.top_right_grid.setSpacing(2)
        
        
        #Widgets for the top panel.
        top_panel.addWidget(top_left_panel)
        top_panel.addWidget(top_middle_panel)
        top_panel.addWidget(top_right_panel)
        top_panel.addStretch()
        
        #Set the layout of the panels to be the grids.
        top_left_panel.setLayout(self.top_left_grid)
        
        
        navigation_setup(self,par_obj)
        #Sets the current text.
        self.image_num_txt.setText('The Current Image is: ' + str(par_obj.curr_z +1)+' and the time point is: '+str(par_obj.time_pt+1))
        self.count_txt = QtGui.QLabel()

        #Populates the grid with the different widgets.
        self.top_left_grid.addLayout(self.panel_buttons, 0, 0,1,4)
        
        self.top_left_grid.addWidget(self.image_num_txt, 2, 0, 2, 3)

        top_right_panel.setLayout(self.top_right_grid)

        self.eval_im_btn = QtGui.QPushButton('Evaluate Images')
        self.eval_im_btn.clicked.connect(self.evaluate_images)

        self.count_all_btn = QtGui.QPushButton('Count Maxima at all time')
        self.count_all_btn.clicked.connect(self.count_all_fn)

        self.save_output_data_btn = QtGui.QPushButton('Save Output Data')
        self.save_output_data_btn.clicked.connect(self.save_output_data)

        self.save_output_prediction_btn = QtGui.QPushButton('Save Prediction')
        self.save_output_prediction_btn.clicked.connect(self.save_output_prediction)
        
        self.save_output_mask_btn = QtGui.QPushButton('Save Point Mask')
        self.save_output_mask_btn.clicked.connect(self.save_output_mask)  
  
        self.save_output_link = QtGui.QLabel()
        self.save_output_link.setText('''<p><a href="'''+str(par_obj.csvPath)+'''">Goto output folder</a></p>
        <p><span style="font-size: 17px;"><br /></span></p>''')

        self.kernel_show_btn = QtGui.QPushButton('Showing Prediction')
        #self.kernel_show_btn.setMinimumWidth(170)
        

        self.top_right_grid.addWidget(self.kernel_show_btn, 1, 1)

        #Populates the grid on the right with the different widgets.
        self.top_right_grid.addWidget(self.eval_im_btn, 0, 0)
        self.top_right_grid.addWidget(self.save_output_data_btn, 1, 0)
        self.top_right_grid.addWidget(self.save_output_prediction_btn,1,1)
        self.top_right_grid.addWidget(self.save_output_mask_btn,0,1)
        self.top_right_grid.addWidget(self.count_all_btn, 2,0)
        self.top_right_grid.addWidget(self.output_count_txt,2,1,1,4)
        #self.top_right_grid.addWidget(self.save_output_link, 2, 0)
        
        self.count_maxima_btn = QtGui.QPushButton('Count Maxima')
        self.count_maxima_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.count_maxima_btn, 4, 0)
        self.count_maxima_btn.clicked.connect(self.count_maxima_btn_fn)
        self.kernel_show_btn.clicked.connect(self.kernel_btn_fn)

        

        self.count_replot_btn = QtGui.QPushButton('Replot')
        self.count_replot_btn_all = QtGui.QPushButton('Replot All')

        
        

        self.count_txt_1 = QtGui.QLineEdit(str(par_obj.min_distance[0]))
        self.count_txt_1.setFixedWidth(20)
        self.count_txt_2 = QtGui.QLineEdit(str(par_obj.min_distance[1]))
        self.count_txt_2.setFixedWidth(20)
        self.count_txt_3 = QtGui.QLineEdit(str(par_obj.min_distance[2]))
        self.count_txt_3.setFixedWidth(20)

        abs_thr_lbl = QtGui.QLabel('Abs Thr:')
        self.abs_thr_txt = QtGui.QLineEdit(str(par_obj.abs_thr))
        self.abs_thr_txt.setFixedWidth(25)
        rel_thr_lbl = QtGui.QLabel('Rel Thr:')
        self.rel_thr_txt = QtGui.QLineEdit(str(par_obj.rel_thr))
        self.rel_thr_txt.setFixedWidth(25)
        

        self.min_distance_panel = QtGui.QHBoxLayout()
        self.min_distance_panel.addStretch()
        self.min_distance_panel.addWidget(QtGui.QLabel("x:"))
        self.min_distance_panel.addWidget(self.count_txt_1)
        self.min_distance_panel.addWidget(QtGui.QLabel("y:"))
        self.min_distance_panel.addWidget(self.count_txt_2 )
        self.min_distance_panel.addWidget(QtGui.QLabel("z:"))
        self.min_distance_panel.addWidget(self.count_txt_3 )
        self.min_distance_panel.addWidget(abs_thr_lbl)
        self.min_distance_panel.addWidget(self.abs_thr_txt)
        self.min_distance_panel.addWidget(rel_thr_lbl)
        self.min_distance_panel.addWidget(self.rel_thr_txt)
        
        self.top_right_grid.addLayout(self.min_distance_panel,4,1)
        
        #self.top_right_grid.addWidget(self.count_replot_btn_all,0,1)
        #self.top_right_grid.addWidget(self.count_replot_btn,1,1)

        self.top_right_grid.setRowStretch(4,2)
        
        #self.top_panel = QtGui.QHBoxLayout()
        self.compl_btn = QtGui.QPushButton('save roi to frame')
        self.compl_btn.clicked.connect(self.cursor.complete_roi)
        
        self.interpolate_btn = QtGui.QPushButton('interpolate between z slices')

        self.interpolate_btn.clicked.connect(self.interpolate_roi_fn)

        self.interpolate_time_btn = QtGui.QPushButton('interpolate between time frames')

        self.interpolate_time_btn.clicked.connect(self.interpolate_roi_in_time_fn)
        

        roi_panel = QtGui.QVBoxLayout()

        roi_panel.addWidget(self.compl_btn)
        roi_panel.addWidget(self.interpolate_btn)
        roi_panel.addWidget(self.interpolate_time_btn)
        roi_panel.addStretch()


        top_middle_panel.setLayout(roi_panel)


        #Sets up the image panel splitter.
        image_panel = QtGui.QSplitter(QtCore.Qt.Horizontal)
        image_panel.addWidget(self.canvas1)
        image_panel.addWidget(self.canvas2)

        

        #Splitter which separates the controls at the top and the images below.
        splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        hbox1 = QtGui.QWidget()
        hbox1.setLayout(top_panel)
        splitter.addWidget(hbox1)
        splitter.addWidget(image_panel)
        box.addWidget(splitter)

        #Status bar which is located beneath images.
        self.image_status_text = QtGui.QStatusBar()
        box.addWidget(toolbar)
        box.addWidget(self.image_status_text)
        self.image_status_text.showMessage('Status: Please Select a Region and Click \'Save ROI\'. ')
        
    

    
        self.modelLoadedText = QtGui.QLabel(self)
        
        self.imageNumText = QtGui.QLabel(self)
        
        self.evalStatusText = QtGui.QLabel(self)
        self.canvas1.mpl_connect('motion_notify_event', self.cursor.motion_notify_callback)
        self.canvas1.mpl_connect('button_press_event', self.cursor.button_press_callback)
        self.canvas1.mpl_connect('button_release_event', self.cursor.button_release_callback)
    def interpolate_roi_fn(self):
        self.cursor.interpolate_ROI()
        self.goto_img_fn()
        
        #v2.eval_pred_show_fn(par_obj.curr_z, par_obj,self)
    def interpolate_roi_in_time_fn(self):
        self.cursor.interpolate_ROI_in_time()
        self.goto_img_fn()
        #v2.eval_pred_show_fn(par_obj.curr_z, par_obj,self)
    def count_all_fn(self):
        print '1'
        for fileno in range(par_obj.max_file):
            for tpt in par_obj.time_pt_list:
                v2.count_maxima(par_obj,tpt,fileno)
        par_obj.show_pts = 1
        self.kernel_btn_fn()
    def kernel_btn_fn(self):
        """Shows the kernels on the image."""
        
        par_obj.show_pts = par_obj.show_pts+ 1
        if par_obj.show_pts ==3:
             par_obj.show_pts = 1
        
        if par_obj.show_pts == 1:
            self.kernel_show_btn.setText('Showing Probability')
            #self.goto_img_fn(par_obj.curr_z,par_obj)
            self.goto_img_fn()
            #v2.goto_img_fn(par_obj, int_obj,par_obj.curr_z,par_obj.time_pt)
            #v2.eval_goto_img_fn(par_obj.curr_z,par_obj,self) ###needs correcting??###
        elif par_obj.show_pts == 2:
            self.kernel_show_btn.setText('Showing Counts')
            #self.goto_img_fn(par_obj.curr_z,par_obj)
            self.goto_img_fn()
            #v2.goto_img_fn(par_obj, int_obj,par_obj.curr_z,par_obj.time_pt)
            #v2.eval_goto_img_fn(par_obj.curr_z,par_obj,self)
    def count_maxima_btn_fn(self):
        par_obj.min_distance[0]= float(self.count_txt_1.text())
        par_obj.min_distance[1]= float(self.count_txt_2.text())
        par_obj.min_distance[2]= float(self.count_txt_3.text())
        par_obj.abs_thr =float(self.abs_thr_txt.text())
        par_obj.rel_thr =float(self.rel_thr_txt.text())
        
        v2.count_maxima(par_obj,par_obj.time_pt,par_obj.curr_file)
        par_obj.show_pts= 1
        self.kernel_btn_fn()
        #v2.eval_pred_show_fn(par_obj.curr_z, par_obj,self)
        #self.goto_img_fn(par_obj.curr_z,par_obj)
        #self.goto_img_fn(par_obj.curr_z,par_obj.time_pt)
        return

        
    def evaluate_images(self):
        par_obj.double_feat_arr = {}
        par_obj.feat_arr = {}
        par_obj.pred_arr = {}
        par_obj.sum_pred = {}
        '''
        for fileno in range(par_obj.max_file):
            for tpt in par_obj.time_pt_list:
                count = -1
                frames =par_obj.frames_2_load
                #try to make it threadable
                #v2.im_pred_inline_fn_eval(par_obj, self,outer_loop=b,inner_loop=frames,threaded=True)
                v2.im_pred_inline_fn_new(par_obj, self,frames,[tpt],[fileno],True)
                for i in frames:
                    #v2.im_pred_inline_fn(par_obj, self,inline=True,outer_loop=b,inner_loop=frames,count=count)
                    #v2.evaluate_forest(par_obj,self, False, 0,inline=True,outer_loop=b,inner_loop=i,count=count)
                    v2.evaluate_forest_new(par_obj,self,False,0,[i],[tpt],[fileno],False)
                    count = count+1
                par_obj.data_store['feat_arr'][fileno][tpt] = {}
                v2.count_maxima(par_obj,tpt,fileno)
                time.sleep(0.001)
        '''
        for fileno in range(par_obj.max_file):
            for tpt in par_obj.time_pt_list:

                frames =par_obj.frames_2_load
                #try to make it threadable
                #v2.im_pred_inline_fn_eval(par_obj, self,outer_loop=b,inner_loop=frames,threaded=True)
                v2.im_pred_inline_fn_new(par_obj, self,frames,[tpt],[fileno],True)
                
                for i in frames:
                    v2.evaluate_forest_new(par_obj,self,False,0,[i],[tpt],[fileno],False)



                if par_obj.double_train==True:
                    v2.im_pred_inline_fn_new(par_obj, self,frames,[tpt],[fileno],threaded='auto')
                    par_obj.data_store['feat_arr'][fileno][tpt] = {}
                    for i in frames:
                        v2.evaluate_forest_new(par_obj,self,False,1,[i],[tpt],[fileno],False,arr='double_feat_arr')
                    par_obj.data_store['double_feat_arr'][fileno][tpt] = {}
                else:
                    par_obj.data_store['feat_arr'][fileno][tpt] = {}
                v2.count_maxima(par_obj,tpt,fileno)
                time.sleep(0.001)
        #set count maxima parameters from model
        self.count_txt_1.setText(str(par_obj.min_distance[0]))
        self.count_txt_2.setText(str(par_obj.min_distance[1]))
        self.count_txt_3.setText(str(par_obj.min_distance[2]))
        self.abs_thr_txt.setText(str(par_obj.abs_thr))
        self.rel_thr_txt.setText(str(par_obj.rel_thr))
        


        self.count_maxima_btn.setEnabled(True)
        self.count_all_btn.setEnabled(True)
        self.save_output_data_btn.setEnabled(True)
        self.save_output_prediction_btn.setEnabled(True)
        self.save_output_mask_btn.setEnabled(True)
        self.image_status_text.showMessage('Status: evaluation finished.')
        par_obj.eval_load_im_win_eval = True
        par_obj.time_pt = 0
        #TODO check why this is not using the local version of goto_img
        par_obj.show_pts=1
        v2.goto_img_fn_new(par_obj, self)
        
        #v2.eval_pred_show_fn(par_obj,self,par_obj.curr_z,par_obj.time_pt)
    def save_output_data(self):
        v2.save_output_data_fn(par_obj,self)
    def save_output_prediction(self):
        v2.save_output_prediction_fn(par_obj,self)
    def save_output_mask(self):
        v2.save_output_mask_fn(par_obj,self)
    def report_progress(self,message):
        self.image_status_text.showMessage('Status: ' + message)
        app.processEvents()
    def draw_saved_dots_and_roi(self):
        pass
    def loadTrainFn(self):
        #Win_fn()
        channel_wid = QtGui.QWidget()
        channel_lay = QtGui.QHBoxLayout()
        ButtonGroup=create_channel_objects(self,par_obj,par_obj.numCH)
        for x in ButtonGroup:
            channel_lay.addWidget(x)
            x.show()
            x.setChecked(True)
        channel_lay.addStretch()
        channel_wid.setLayout(channel_lay)
        self.top_left_grid.addWidget(channel_wid,1,0,1,3)

    def goto_img_fn(self,zslice=None,tpt=None,imno=None):
        if zslice!=None:
            par_obj.curr_z=zslice
        if tpt!=None:
            par_obj.time_pt=tpt
        if imno!=None:
            par_obj.curr_file=imno
            
        self.cursor.ppt_x = []
        self.cursor.ppt_y = []
        self.cursor.line = [None]
        #self.cursor.draw_ROI()
        #self.canvas1.draw()
        
        self.cursor.complete = False
        self.cursor.flag = False
        try: #TODO not really sure what went wrong here, fix later
            for bt in par_obj.data_store['roi_stk_x'][imno][tpt]:
                if bt == zslice:
                    self.cursor.complete = True
                    self.cursor.ppt_x = copy.deepcopy(par_obj.data_store['roi_stk_x'][imno][tpt][bt])
                    self.cursor.ppt_y = copy.deepcopy(par_obj.data_store['roi_stk_y'][imno][tpt][bt])
                    
                    break;
        except KeyError:
            pass
        #v2.eval_goto_img_fn(im_num,par_obj,self)
        v2.goto_img_fn_new(par_obj, self)
                  
class checkBoxCH(QtGui.QCheckBox):
    def __init__(self):
        QtGui.QCheckBox.__init__(self)
        self.stateChanged.connect(self.stateChange)
        self.type = None;
    def stateChange(self):
        
        if self.type == 'visual_ch':
            #v2.eval_goto_img_fn(par_obj.curr_z,par_obj,evalImWin)
            Eval_disp_im_win.goto_img_fn(evalImWin)

            
class File_Dialog(QtGui.QMainWindow):
    
    def __init__(self,par_obj,int_obj,modelTabObj):
        super(File_Dialog, self).__init__()
       
        
        self.int_obj = int_obj
        self.par_obj = par_obj
        self.type = 'im'
        self.modelTabObj = modelTabObj
        self.initUI()
        
    def initUI(self):      

        self.textEdit = QtGui.QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.statusBar()

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open', self)

        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)       
        
        self.setGeometry(300, 300, 350, 500)
        self.setWindowTitle('File dialog')
        self.par_obj.config ={}
        try:
            self.par_obj.config = pickle.load(open(os.path.expanduser('~')+'/.densitycount/config.p', "rb" ));
            self.par_obj.filepath = self.par_obj.config['evalpath']
        except:
            self.par_obj.filepath = os.path.expanduser('~')+'/'
        #self.show()
        
    def showDialog(self):
        self.int_obj.selIntButton.setEnabled(False)
        #filepath = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        par_obj.file_array =[]
        for path in QtGui.QFileDialog.getOpenFileNames(self, 'Open file', self.par_obj.filepath,'Images(*.tif *.tiff *.png *.oib *.oif);;'):
            par_obj.file_array.append(path)
        
        self.par_obj.config['evalpath'] = str(QtCore.QFileInfo(path).absolutePath())+'/'
        pickle.dump(self.par_obj.config, open(str(os.path.expanduser('~')+'/.densitycount/config.p'), "w" ))
        self.par_obj.csvPath = self.par_obj.config['evalpath']




        v2.import_data_fn(par_obj,par_obj.file_array)

    
        self.int_obj.image_status_text.showMessage('Status: Loading Images. ')                
        if self.type == 'im':
            if(self.par_obj.file_array.__len__()>0):
                self.int_obj.selIntButton.setEnabled(True)
            

        self.refreshTable()
    def refreshTable(self):
        self.int_obj.image_status_text.showMessage(str(self.par_obj.file_array.__len__())+' Files Selected.')
        filesLen =self.par_obj.file_array.__len__()

        self.modelTabObj.show()
        
        c = 0
        
        for i in range(0,self.par_obj.file_array.__len__()):
                
                self.modelTabObj.setRowCount(c+1)
                btn = removeImBtn(self.int_obj,self.par_obj,self,i)
                btn.setText('Click to Remove')
                self.modelTabObj.setCellWidget(c, 0, btn)

                text1 = QtGui.QLabel(self.modelTabObj)
                text1.setText(str(self.par_obj.file_array[i]).split('/')[-1])
                self.modelTabObj.setCellWidget(c,1,text1)
                self.par_obj.input_range = QtGui.QLineEdit(self.modelTabObj)
                self.par_obj.input_range.setText('1-'+str(self.par_obj.total_time_pt))

                self.modelTabObj.setCellWidget(c,2,self.par_obj.input_range)
                text3 = QtGui.QLabel(self.modelTabObj)
                text3.setText(str(self.par_obj.file_array[i]))
                self.modelTabObj.setCellWidget(c,3,text3)

                
                c= c+1
        self.modelTabObj.show()
        if self.type == 'im':
            if self.par_obj.file_array.__len__() ==0:
                self.modelTabObj.hide()               
                
                

class loadModelBtn(QtGui.QPushButton):
    def __init__(self,par_obj,int_obj,parent,idnum,fileName):
        QtGui.QPushButton.__init__(self,parent)
        self.par_obj = par_obj
        self.int_obj = int_obj
        self.modelNum = idnum
        self.fileName = fileName
        self.clicked.connect(self.onClick)
        self.type = []
    def onClick(self):
        self.int_obj.loadModelFn(self.par_obj,self.fileName)


class removeImBtn(QtGui.QPushButton):
    def __init__(self,parent,par_obj,table,idnum):
        QtGui.QPushButton.__init__(self,parent)
        self.modelNum = idnum
        self.par_obj = par_obj
        self.clicked.connect(self.onClick)
        self.table = table
    def onClick(self):
        self.par_obj.file_array.pop(self.modelNum)
        self.table.refreshTable()
 
#Creates win, an instance of QWidget
class widgetSP(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.par_obj =[]
    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Period:
            im_num = self.par_obj.curr_z + 1
        if ev.key() == QtCore.Qt.Key_Comma:
            im_num = self.par_obj.curr_z - 1
        v2.evalGotoImgFn(im_num,self.par_obj, self)
    def wheelEvent(self, event):
        super(widgetSP, self).wheelEvent(event)
        if event.delta() < 0:
            im_num = self.par_obj.curr_z + 1
        if event.delta() > 0:
            im_num = self.par_obj.curr_z - 1
        v2.evalGotoImgFn(im_num,self.par_obj, self)



#generate layout
app = QtGui.QApplication([])

# Create and display the splash screen
splash_pix = QtGui.QPixmap('splash_loading.png')
splash = QtGui.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
splash.setMask(splash_pix.mask())
splash.show()
app.processEvents()


#Creates tab widget.
win_tab = QtGui.QTabWidget()

#Intialises counter object.
par_obj = parameterClass()
#Main widgets.
evalLoadImWin = Eval_load_im_win(par_obj)
evalLoadModelWin = Eval_load_model_win(par_obj)
evalImWin = Eval_disp_im_win(par_obj)


#Adds win tab and places button in win.
win_tab.addTab(evalLoadImWin, "Select Images")
win_tab.addTab(evalLoadModelWin, "Load Model")
win_tab.addTab(evalImWin, "Evaluate Images")

#Defines size of the widget.
win_tab.resize(1200, 800)

time.sleep(2.0)
splash.finish(win_tab)

#Initalises load screen.
#eval_load_im_win_fn(par_obj,evalLoadImWin)
#evalLoadModelWinFn(par_obj,evalLoadModelWin)
#evalDispImWinFn(par_obj,evalImWin)
win_tab.show()
# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
