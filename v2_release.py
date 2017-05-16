#Main script for running QuantiFly training.
import time
import numpy as np
from PyQt4 import QtGui, QtCore,QtWebKit
import errno
import os
import os.path
import re
import cPickle as pickle
import sys

#from scipy.special import _ufuncs_cxx
import sklearn.utils.lgamma

import matplotlib.lines as lines
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
import v2_functions as v2
#import numdifftools as ndt
from common_navigation import navigation_setup,create_channel_objects,btn_fn, on_about
from parameter_object import parameterClass
from user_ROI import ROI
"""QBrain Software v0.1

    Copyright (C) 2017  Dominic Waithe Martin Hailstone

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


class fileDialog(QtGui.QMainWindow):
    """The dialog for loading images"""
    def __init__(self,parent):
        super(fileDialog, self).__init__()
        self.parent = parent
        self.initUI()
        self.parent.config ={}

        try:
            self.parent.config = pickle.load(open(os.path.expanduser('~')+'/.densitycount/config.p', "rb" ));
            self.parent.filepath = self.parent.config['filepath']
        except:
            self.parent.filepath = os.path.expanduser('~')+'/'
            try:
                os.makedirs(os.path.expanduser('~')+'/.densitycount/')
            except:
                'unable to make directory: ',os.path.expanduser('~')+'/.densitycount/'


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


    def showDialog(self):
        
        self.parent.selIntButton.setEnabled(False)
        par_obj.file_array =[]
        path =None
        for path in QtGui.QFileDialog.getOpenFileNames(self, 'Open file',self.parent.filepath,'Images(*.tif *.tiff);;'):
            par_obj.file_array.append(path)
        if path==None:return
        self.parent.config['filepath'] = str(QtCore.QFileInfo(path).absolutePath())+'/'
        pickle.dump(self.parent.config, open(str(os.path.expanduser('~')+'/.densitycount/config.p'), "w" ))

        self.parent.image_status_text.showMessage('Status: Loading Images. ')
        success, updateText = v2.import_data_fn(par_obj, par_obj.file_array)

        self.parent.image_status_text.showMessage(updateText)
        if success == True:
            self.parent.updateAfterImport()


class Load_win_fn(QtGui.QWidget):
    """The class for loading and processing images"""
    def __init__(self,par_obj,win):
        super(Load_win_fn, self).__init__()

        #Load images button
        load_im_lo = QtGui.QHBoxLayout()
        self.loadImages_button = QtGui.QPushButton("Load Images", self)
        load_im_lo.addWidget(self.loadImages_button)
        load_im_lo.addStretch()
        self.ex = fileDialog(self)
        self.loadImages_button.clicked.connect(self.ex.showDialog)
        
        #about button
        about=QtGui.QPushButton('About',self)
        load_im_lo.addWidget(about)
        about.clicked.connect(lambda: on_about(self))
        #SigmaData input field.

        resize_lo = QtGui.QHBoxLayout()
        sigma_lo = QtGui.QHBoxLayout()
        sampling_lo= QtGui.QHBoxLayout()
        max_features_lo= QtGui.QHBoxLayout()


        self.samplingText = QtGui.QLabel('Sampling:')
        self.samplingText.resize(40,20)
        self.samplingText.hide()
        self.sampling_input = QtGui.QLineEdit(str(par_obj.limit_ratio_size))
        self.sampling_input.resize(10,10)
        self.sampling_input.textChanged[str].connect(self.sampling_change)
        self.sampling_input.hide()
        sampling_lo.addWidget(self.samplingText)
        sampling_lo.addWidget(self.sampling_input)
        self.featuresText = QtGui.QLabel('Max size Random Feature subset')
        self.featuresText.resize(40,20)
        self.featuresText.hide()
        self.features_input = QtGui.QLineEdit(str(par_obj.max_features))
        self.features_input.resize(10,10)
        self.features_input.textChanged[str].connect(self.features_change)
        self.features_input.hide()
        max_features_lo.addWidget(self.featuresText)
        max_features_lo.addWidget(self.features_input)

        self.feature_scaleText = QtGui.QLabel('Input sigma for features:')
        self.feature_scaleText.resize(40,20)
        self.feature_scaleText.hide()
        self.feature_scale_input = QtGui.QLineEdit(str(par_obj.feature_scale))
        self.feature_scale_input.resize(10,10)
        self.feature_scale_input.textChanged[str].connect(self.feature_scale_change)
        self.feature_scale_input.hide()
        sigma_lo.addWidget(self.feature_scaleText)
        sigma_lo.addWidget(self.feature_scale_input)

        self.resize_factor_text = QtGui.QLabel('Resize Factor:')
        self.resize_factor_text.resize(40,20)
        self.resize_factor_input = QtGui.QLineEdit(str(par_obj.resize_factor))
        self.resize_factor_input.resize(10,10)
        self.resize_factor_input.textChanged[str].connect(self.resize_factor_change)
        resize_lo.addWidget(self.resize_factor_text)
        resize_lo.addWidget(self.resize_factor_input)



        #Channel dialog generation.
        Channel_Select_lo = QtGui.QVBoxLayout()
        Channel_button_lo = QtGui.QHBoxLayout()
        self.Text_CHopt = QtGui.QLabel('Please select which channels you want to include in the feature calculation')
        Channel_Select_lo.addWidget(self.Text_CHopt)
        self.Text_CHopt.resize(500,40)
        self.Text_CHopt.hide()
        #Object factory for channel selection.
        ButtonGroup=create_channel_objects(self,par_obj,10,True)
        for x in ButtonGroup:
            x.hide()
            Channel_button_lo.addWidget(x)
        Channel_Select_lo.addLayout(Channel_button_lo)
        Channel_button_lo.addStretch()
        Channel_Select_lo.addStretch()

        #setup preview plot
        self.figure1 = Figure(figsize=(2,2))
        self.canvas1 = FigureCanvas(self.figure1)
        self.plt1 = self.figure1.add_subplot(1,1,1)
        self.resize(100,100)
        self.figure1.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])
        self.canvas1.hide()
        par_obj.rect_h=0
        par_obj.rect_w=0
        canvas_lo =QtGui.QHBoxLayout()
        canvas_lo.addWidget(self.canvas1)
        canvas_lo.addStretch()


        #Image range selection
        Im_Range_lo=QtGui.QVBoxLayout()
        #Image Details
        self.Text_FrmOpt2 = QtGui.QLabel()
        Im_Range_lo.addWidget(self.Text_FrmOpt2)
        self.Text_FrmOpt2.hide()
        self.Text_FrmOpt4 = QtGui.QLabel()
        Im_Range_lo.addWidget(self.Text_FrmOpt4)
        self.Text_FrmOpt4.hide()

        #Image frames dialog.
        Text_FrmOpt1_panel = QtGui.QHBoxLayout()
        self.Text_FrmOpt1 = QtGui.QLabel('Please choose the z-slices you wish to use for training. Use either \',\' to separate individual frames or a \'-\' to indicate a range:')
        self.Text_FrmOpt1.hide()
        Im_Range_lo.addLayout(Text_FrmOpt1_panel)

        #Image frames input.
        linEdit_Frm_panel = QtGui.QHBoxLayout()
        self.linEdit_Frm = QtGui.QLineEdit()
        self.linEdit_Frm.hide()
        linEdit_Frm_panel.addWidget(self.linEdit_Frm)
        linEdit_Frm_panel.addStretch()
        Im_Range_lo.addLayout(linEdit_Frm_panel)

        Text_FrmOpt3_panel = QtGui.QHBoxLayout()
        self.Text_FrmOpt3 = QtGui.QLabel()
        self.Text_FrmOpt3.setText('Please choose the time-points you wish to use for training. Use either \',\' to separate individual frames or a \'-\' to indicate a range:')
        self.Text_FrmOpt3.hide()
        Text_FrmOpt1_panel.addWidget(self.Text_FrmOpt1)
        Text_FrmOpt3_panel.addWidget(self.Text_FrmOpt3)
        Im_Range_lo.addLayout(Text_FrmOpt3_panel)


        linEdit_Frm_panel2 = QtGui.QHBoxLayout()
        self.linEdit_Frm2 = QtGui.QLineEdit()
        self.linEdit_Frm2.hide()
        linEdit_Frm_panel2.addWidget(self.linEdit_Frm2)
        linEdit_Frm_panel2.addStretch()
        Im_Range_lo.addLayout(linEdit_Frm_panel2)


        #Feature calculation type to perform:
        self.Text_Radio = QtGui.QLabel('Feature select which kind of feature detection you would like to use:')
        self.Text_Radio.resize(500,40)
        self.Text_Radio.hide()
        self.radio_group=QtGui.QButtonGroup(self) # Number
        self.r0 = QtGui.QRadioButton("Basic",self)
        self.r1 = QtGui.QRadioButton("Fine",self)
        self.r2 = QtGui.QRadioButton("Fine3",self)
        self.r0.setChecked(True)
        self.radio_group.addButton(self.r0)
        self.radio_group.addButton(self.r1)
        self.radio_group.addButton(self.r2)
        self.r0.hide()
        self.r1.hide()
        self.r2.hide()
        Radio_Layout=QtGui.QVBoxLayout()
        Radio_Layout.addWidget(self.Text_Radio)
        Radio_Layout.addWidget(self.r0)
        Radio_Layout.addWidget(self.r1)
        Radio_Layout.addWidget(self.r2)
        Radio_Layout.addStretch()

        #Load images button
        Confirm_im_lo= QtGui.QHBoxLayout()
        self.confirmImages_btn = QtGui.QPushButton("Confirm Images")
        self.confirmImages_btn.clicked.connect(self.confirmImages_btn_fn)
        self.confirmImages_btn.setEnabled(False)

        #Move to training button.
        self.selIntButton = QtGui.QPushButton("Goto Training")
        self.selIntButton.clicked.connect(win.loadTrainFn)
        self.selIntButton.setEnabled(False)

        Confirm_im_lo.addWidget(self.confirmImages_btn)
        Confirm_im_lo.addWidget(self.selIntButton)
        Confirm_im_lo.addStretch()
        self.image_status_text = QtGui.QStatusBar()
        self.image_status_text.setStyleSheet("QLabel {  color : green }")
        self.image_status_text.showMessage('Status: Highlight training images in folder. ')




        #Set up Vertical layout
        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addLayout(load_im_lo)
        vbox.addLayout(resize_lo)
        vbox.addLayout(sigma_lo)
        vbox.addLayout(sampling_lo)
        vbox.addLayout(max_features_lo)
        vbox.addLayout(Channel_Select_lo)
        vbox.addLayout(canvas_lo)
        vbox.addLayout(Im_Range_lo)
        vbox.addLayout(Radio_Layout)
        vbox.addStretch()
        vbox.addLayout(Confirm_im_lo)
        vbox.addWidget(self.image_status_text)

        #File browsing functions
        #layout = QtGui.QVBoxLayout()


    def resize_factor_change(self,text):
        """Updates on change of image resize parameter"""
        if text != "":
            par_obj.resize_factor = int(text)
    def sampling_change(self,text):
        """Updates on change of sampling"""
        if (text != ""):
            par_obj.limit_ratio_size = float(text)
    def features_change(self,text):
        """Updates on change of feature number"""
        if (text != ""):
            par_obj.max_features = int(text)
    def feature_scale_change(self,text):
        """Updates on change of feature scale"""
        if (text != ""):
            par_obj.feature_scale = float(text)

            for fileno in range(0,par_obj.max_file):
                for tp in range(0,par_obj.max_t):
                    par_obj.data_store['feat_arr'][fileno][tp] = {}
    def updateAfterImport(self):
        """Specific to ui updates"""
        if par_obj.max_z >1:

            #self.linEdit_Frm.setText('1-'+str(par_obj.max_z))
            self.Text_FrmOpt2.setText('There are '+str(par_obj.max_z)+' z-slices in total. The image has dimensions x: '+str(par_obj.ori_width)+' and y: '+str(par_obj.ori_height))
            if par_obj.max_t > 1:
                #self.linEdit_Frm2.setText('1-'+str(par_obj.max_t))
                self.Text_FrmOpt4.setText('There are '+str(par_obj.max_t)+' timepoints in total.')
                self.linEdit_Frm2.show()
                self.Text_FrmOpt4.show()
                self.Text_FrmOpt3.show()
            self.Text_FrmOpt1.show()
            self.Text_FrmOpt2.show()
            self.linEdit_Frm.show()

        self.confirmImages_btn.setEnabled(True)

        self.plt1.cla()
        self.plt1.imshow(par_obj.ex_img/par_obj.ex_img.max())
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])
        self.canvas1.show()
        self.canvas1.draw()


        par_obj.ch_active =[];
        if par_obj.numCH> 0:
            self.Text_CHopt.show()
            for i in range(0,par_obj.numCH):
                self.CH_cbx[i].show()
                par_obj.ch_active.append(i)
        else:
            par_obj.ch_active.append(0)
        #self.CH_cbx1.stateChange()
        self.feature_scale_input.show()
        self.feature_scaleText.show()
        self.sampling_input.show()
        self.samplingText.show()
        self.features_input.show()
        self.featuresText.show()
        self.r0.show()
        self.r1.show()
        self.r2.show()
        self.Text_Radio.show()
    def hyphen_range(self,s):
        """ yield each integer from a complex range string like "1-9,12, 15-20,23"

        >>> list(hyphen_range('1-9,12, 15-20,23'))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 23]

        >>> list(hyphen_range('1-9,12, 15-20,2-3-4'))
        Traceback (most recent call last):
            ...
        ValueError: format error in 2-3-4
        """
        s_list=[]
        if s != '':
            for x in s.split(','):
                elem = x.split('-')
                if len(elem) == 1: # a number
                    s_list.append(int(elem[0])-1)
                elif len(elem) == 2: # a range inclusive
                    start, end = map(int, elem)
                    for i in xrange(start, end+1):
                        s_list.append(i-1)
                else: # more than one hyphen
                    raise ValueError('format error in %s' % x)
        return s_list

    def confirmImages_btn_fn(self):
        if par_obj.max_t> 1:
            tmStr = self.linEdit_Frm2.text()
            if tmStr != '':
                par_obj.tpt_list= self.hyphen_range(tmStr)
            else:
                par_obj.tpt_list=range(par_obj.max_t)
        else:
            par_obj.tpt_list = [0]
            
        if par_obj.max_z> 0:
            fmStr = self.linEdit_Frm.text()
            if fmStr != '': #catch empty limit
                par_obj.user_max_z = max(self.hyphen_range(fmStr))
                par_obj.user_min_z = min(self.hyphen_range(fmStr))
            else: 
                par_obj.user_max_z = []
                par_obj.user_min_z = 0
        else:
            par_obj.user_max_z = []
            par_obj.user_min_z = 0


        self.image_status_text.showMessage('Status: Images loaded. Click \'Goto Training\'')
        self.selIntButton.setEnabled(True)
        v2.processImgs(self,par_obj)
        if self.r0.isChecked():
            par_obj.feature_type = 'basic'
        if self.r1.isChecked():
            par_obj.feature_type = 'fine'

        if self.r2.isChecked():
            par_obj.feature_type = 'fine3'
        print par_obj.feature_type

    def report_progress(self,message):
        self.image_status_text.showMessage('Status: '+message)
        app.processEvents()
class Win_fn(QtGui.QWidget):
    """Class which houses main training functionality"""
    def __init__(self,par_obj):
        super(Win_fn, self).__init__()
        #Sets up the figures for displaying images.
        self.figure1 = Figure(figsize=(8, 8), dpi=100)
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure1.patch.set_facecolor('grey')
        self.cursor = ROI(self,par_obj)

        self.plt1 = self.figure1.add_subplot(1, 1, 1)
        im_RGB = np.zeros((512, 512))
        #Makes sure it spans the whole figure.
        self.figure1.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)

        self.toolbar = NavigationToolbar(self.canvas1,self)
        self.toolbar.actionTriggered.connect(self.on_resize)

        self.plt1.imshow(im_RGB)

        #Removes the tick labels
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])

        #Initialises the second figure.
        self.figure2 = Figure(figsize=(8, 8), dpi=100)
        self.canvas2 = FigureCanvas(self.figure2)
        self.figure2.patch.set_facecolor('grey')
        self.plt2 = self.figure2.add_subplot(1, 1, 1)
        self.plt2_is_clear=False

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
        top_right_panel = QtGui.QGroupBox('Advanced Controls')

        #Grid layouts for the top and left panels.
        self.top_left_grid = QtGui.QGridLayout()
        self.top_right_grid = QtGui.QGridLayout()
        self.top_left_grid.setSpacing(2)
        self.top_right_grid.setSpacing(2)

        #Widgets for the top panel.
        top_panel.addWidget(top_left_panel)
        top_panel.addWidget(top_right_panel)
        top_panel.addStretch()

        #Set the layout of the panels to be the grids.
        top_left_panel.setLayout(self.top_left_grid)
        top_right_panel.setLayout(self.top_right_grid)


        #Sets the current text.
        self.image_num_txt.setText(' Image is: ' + str(par_obj.curr_z +1))
        self.count_txt = QtGui.QLabel()

        #Sets up the button which saves the ROI.
        self.save_ROI_btn = QtGui.QPushButton('Save ROI')
        self.save_ROI_btn.setEnabled(True)

        #Sets up the button which saves the ROI.
        self.save_dots_btn = QtGui.QPushButton('Save Dots')
        self.save_dots_btn.setEnabled(False)

        #Button for training model
        self.train_model_btn = QtGui.QPushButton('Train Model')
        self.train_model_btn.setEnabled(False)

        #Selects and reactivates an existing ROI.
        self.sel_ROI_btn = QtGui.QPushButton('Select ROI')
        self.sel_ROI_btn.setEnabled(True)
        self.select_ROI= False

        #Allows deletion of dots.
        self.remove_dots_btn = QtGui.QPushButton('Remove Dots')
        self.remove_dots_btn.setEnabled(False)
        self.remove_dots = False

        #Load in ROI file.
        self.load_gt_btn = QtGui.QPushButton('Import ROI/Dots')
        self.load_gt_btn.setEnabled(True)
        self.load_gt = False

        #Save in ROI file.
        self.save_gt_btn = QtGui.QPushButton('Output ROI/Dots')
        self.save_gt_btn.setEnabled(True)
        self.save_gt = False
        
        #common navigation elements
        self.Btn_fns = btn_fn(self)     
        navigation_setup(self,par_obj)

        #Populates the grid with the different widgets.
        self.top_left_grid.addLayout(self.panel_buttons, 0, 0,1,4)

        self.top_left_grid.addWidget(self.image_num_txt, 2, 0, 2, 3)
        self.top_left_grid.addWidget(self.save_ROI_btn, 4, 0)
        self.top_left_grid.addWidget(self.save_dots_btn, 4, 1)
        self.top_left_grid.addWidget(self.train_model_btn, 4, 2)
        self.top_left_grid.addWidget(self.sel_ROI_btn, 5, 0)
        self.top_left_grid.addWidget(self.remove_dots_btn, 5, 1)
        self.top_left_grid.addWidget(self.load_gt_btn, 6,0,1,1)
        self.top_left_grid.addWidget(self.save_gt_btn, 6,1,1,1)

        #SigmaData input Label.
        self.sigma_data_text = QtGui.QLabel(self)
        self.sigma_data_text.setText('Input Sigma for Kernel Size:')
        self.top_right_grid.addWidget(self.sigma_data_text, 0, 0)

        #SigmaData input field.
        self.sigma_data_input = QtGui.QLineEdit(str(par_obj.sigma_data))
        self.sigma_data_input.onChanged = self.sigmaOnChange
        self.sigma_data_input.setFixedWidth(40)
        self.sigma_data_input.textChanged[str].connect(self.sigmaOnChange)
        self.top_right_grid.addWidget(self.sigma_data_input, 0, 1)


        #Feature scale input Label.
        #self.sigma_data_text = QtGui.QLabel()
        #self.sigma_data_text.setText('Scale of Feature Descriptor:')
        #self.top_right_grid.addWidget(self.sigma_data_text, 1, 0)

        #Feature scale input field
        #self.feature_scale_input = QtGui.QLineEdit(str(par_obj.feature_scale))
        #self.feature_scale_input.onChanged = self.feature_scale_change
        #self.feature_scale_input.resize(40, 20)
        #self.feature_scale_input.textChanged[str].connect(self.feature_scale_change)
        #self.feature_scale_input.setFixedWidth(40)
        #self.top_right_grid.addWidget(self.feature_scale_input, 1, 1)

        #Feature scale input btn.
        #self.feat_scale_change_btn = QtGui.QPushButton('Recalculate Features')
        #self.feat_scale_change_btn.setEnabled(True)
        #self.top_right_grid.addWidget(self.feat_scale_change_btn, 1, 2)

        #Saves the model
        self.save_model_btn = QtGui.QPushButton('Save Training Model')
        self.save_model_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.save_model_btn, 1, 0)

        #Saves the extremel random decision tree model
        self.save_model_name_txt = QtGui.QLineEdit('Insert Model Name')
        self.top_right_grid.addWidget(self.save_model_name_txt, 1, 1)

        self.output_count_txt = QtGui.QLabel()
        self.top_right_grid.addWidget(self.output_count_txt, 3,1,1,4)

        #Saves the extremely random decision tree model.
        self.save_model_desc_txt = QtGui.QLineEdit('Insert Model Description')
        self.save_model_desc_txt.setFixedWidth(200)
        self.top_right_grid.addWidget(self.save_model_desc_txt, 2, 0, 1, 4)
        self.clear_dots_btn = QtGui.QPushButton('Clear All ROI')
        self.top_right_grid.addWidget(self.clear_dots_btn, 1, 2)

        #Shows the kernel label distributions
        self.kernel_show_btn = QtGui.QPushButton('Showing Kernel')
        self.kernel_show_btn.setMinimumWidth(170)
        self.clear_dots_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.kernel_show_btn, 2, 2)

        self.evaluate_btn = QtGui.QPushButton('Evaluate Forest')
        self.evaluate_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.evaluate_btn, 3, 0)

        self.count_maxima_btn = QtGui.QPushButton('Count Maxima')
        self.count_maxima_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.count_maxima_btn, 4, 0)

        #Button for overlay
        self.overlay_prediction_btn = QtGui.QPushButton('Overlay Prediction')
        self.overlay_prediction_btn.setEnabled(True)
        self.top_right_grid.addWidget(self.overlay_prediction_btn, 0,2)

        #common navigation buttons


        self.count_replot_btn = QtGui.QPushButton('Replot')

        self.count_maxima_plot_on = QtGui.QCheckBox()
        self.count_maxima_plot_on.setChecked = False


        self.count_txt_1 = QtGui.QLineEdit(str(par_obj.min_distance[0]))
        self.count_txt_1.setFixedWidth(20)
        self.count_txt_2 = QtGui.QLineEdit(str(par_obj.min_distance[1]))
        self.count_txt_2.setFixedWidth(20)
        self.count_txt_3 = QtGui.QLineEdit(str(par_obj.min_distance[2]))
        self.count_txt_3.setFixedWidth(20)

        abs_thr_lbl = QtGui.QLabel('Abs Thr:')
        self.abs_thr_txt = QtGui.QLineEdit(str(par_obj.abs_thr))
        self.abs_thr_txt.setFixedWidth(35)
        z_cal_lbl = QtGui.QLabel('Z Cal.:')
        self.z_cal_txt = QtGui.QLineEdit(str(par_obj.z_cal))
        self.z_cal_txt.setFixedWidth(50)


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
        self.min_distance_panel.addWidget(z_cal_lbl)
        self.min_distance_panel.addWidget(self.z_cal_txt)

        self.top_right_grid.addLayout(self.min_distance_panel,4,1,1,2)
        #self.top_right_grid.addWidget(self.count_maxima_plot_on,4,2)



        self.top_right_grid.setRowStretch(4,2)

        #Sets up the image panel splitter.
        image_panel = QtGui.QSplitter(QtCore.Qt.Horizontal)
        image_panel.addWidget(self.canvas1)
        image_panel.addWidget(self.canvas2)

        #Sets up mouse settings on the image
        self.canvas1.mpl_connect('axes_enter_event', self.on_enter)
        self.canvas1.mpl_connect('axes_leave_event', self.on_leave)
        self.bpe =self.canvas1.mpl_connect('button_press_event', self.on_click)
        self.bre =self.canvas1.mpl_connect('button_release_event', self.on_unclick)
        self.ome =self.canvas1.mpl_connect('motion_notify_event', self.on_motion)
        self.okp =self.canvas1.mpl_connect('key_press_event', self.on_key)
        #Splitter which separates the controls at the top and the images below.
        splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        hbox1 = QtGui.QWidget()
        hbox1.setLayout(top_panel)
        splitter.addWidget(hbox1)
        splitter.addWidget(image_panel)
        box.addWidget(splitter)

        #Status bar which is located beneath images.
        self.image_status_text = QtGui.QStatusBar()
        box.addWidget(self.toolbar)
        box.addWidget(self.image_status_text)
        self.image_status_text.showMessage('Status: Please Select a Region and Click \'Save ROI\'. ')

        #Connects the buttons.
        self.save_ROI_btn.clicked.connect(self.save_roi_fn)
        self.save_dots_btn.clicked.connect(self.save_dots_fn)

        self.sel_ROI_btn.clicked.connect(self.sel_ROI_btn_fn)
        self.remove_dots_btn.clicked.connect(self.remove_dots_btn_fn)
        self.train_model_btn.clicked.connect(self.train_model_btn_fn)
        self.count_maxima_btn.clicked.connect(self.count_maxima_btn_fn)
        self.count_replot_btn.clicked.connect(self.replot_fn)
        self.load_gt_btn.clicked.connect(self.load_gt_fn)
        self.save_gt_btn.clicked.connect(self.save_gt_fn)
        self.overlay_prediction_btn.clicked.connect(self.overlay_prediction_btn_fn)
        #self.feat_scale_change_btn.clicked.connect(self.feat_scale_change_btn_fn)
        self.kernel_show_btn.clicked.connect(self.kernel_btn_fn)
        self.clear_dots_btn.clicked.connect(self.clear_dots_fn)
        self.save_model_btn.clicked.connect(self.saveForestFn)
        self.evaluate_btn.clicked.connect(self.evaluate_forest_fn)

        #Initialises the variables for the beginning of the counting.
        par_obj.first_time = True
        par_obj.dots = []
        par_obj.rects = np.zeros((1,5))
        par_obj.var =[]
        par_obj.saved_dots = []
        par_obj.saved_ROI = []
        par_obj.subdivide_ROI = []
        self.m_Cursor = self.makeCursor()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
    def evaluate_forest_fn(self):
        #Don't want to train for all the images so we select them.
        v2.im_pred_inline_fn_new(par_obj, self,range(par_obj.max_z+1),[par_obj.curr_t],[par_obj.curr_file],threaded=True)

        v2.evaluate_forest_new(par_obj,self, False,0,range(par_obj.max_z+1),[par_obj.curr_t],[par_obj.curr_file])
        par_obj.show_pts= 0
        self.kernel_btn_fn()
        print 'evaluating'
        self.image_status_text.showMessage('Evaluation complete')
        
    def save_gt_v2(self):
        
        model_name = self.save_model_name_txt.text()
        
        #funky ordering TZCYX
        for fileno,imfile in enumerate(par_obj.filehandlers):
            
            rects=[par_obj.saved_ROI[x] for x,y in enumerate(par_obj.saved_ROI) if y[6]==fileno]
            dots=[par_obj.saved_dots[x] for x,y in enumerate(par_obj.saved_ROI) if y[6]==fileno]
    
            file_to_save = {'dots':dots,'rect':rects}
        
            fileName = imfile.path+'/'+imfile.name+'_'+model_name+'.quantiROI'
            pickle.dump( file_to_save, open( fileName, "wb" ) )
                        
    def save_gt_fn(self):
        file_to_save = {'dots':par_obj.saved_dots,'rect':par_obj.saved_ROI}
        fileName = QtGui.QFileDialog.getSaveFileName(self, "Save dots and regions", "~/Documents", ".quantiROI");
        print 'the filename address',fileName
        if fileName[-10:]=='.quantiROI':
            fileName=fileName[0:-10]
        pickle.dump( file_to_save, open( fileName+".quantiROI", "wb" ) )
        
    def load_gt_fn(self):
        print 'load the gt'
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Load dots and regions", "~/Documents", "QuantiFly ROI files (*.quantiROI) ;;MTrackJ Data Format (*.mdf)")
        filename, file_ext = os.path.splitext(fileName)
        print 'load the file', fileName
        if file_ext=='.quantiROI':
            with open( fileName, "rb" ) as open_file:
                the_file = pickle.load(open_file)
            par_obj.saved_dots = the_file['dots']
            par_obj.saved_ROI = the_file['rect']
            self.clear_dots_btn.setEnabled(True)

        elif file_ext=='.mdf':
            lines_list = list(open(fileName, 'rb').read().split('\n'))
            par_obj.saved_ROI=[]
            par_obj.saved_dots=[]
            for i in lines_list:
                i=i.split(' ')
                if len(i)>0:
                    if i[0]=='Point':
                        #trim point and convert to int
                        i=[int(float(num)) for num in i[1:-1]]
                        #order ID x y t z c from mdf
                        #order in save_dots: txyz
                        #mdf is 1 indexed
                        rects = (i[4]-1, 0, 0, int(abs(par_obj.ori_width)), abs(par_obj.ori_height),i[3]-1,par_obj.curr_file)
                        #append if dots already in roi, otherwise create roi and append
                        if rects in par_obj.saved_ROI:
                            idx=par_obj.saved_ROI.index(rects)
                            par_obj.saved_dots[idx].append((i[4]-1,i[1]-1,i[2]-1))
                        else:
                            par_obj.saved_dots.append([(i[4]-1,i[1]-1,i[2]-1)])
                            par_obj.saved_ROI.append(rects)
        #check if have loaded points and update
        if par_obj.saved_ROI !=[]:
            v2.refresh_all_density(par_obj)
            self.train_model_btn.setEnabled(True)
        self.goto_img_fn(par_obj.curr_z,par_obj.curr_t)

    def load_gt_mdf(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Load dots", "~/Documents", "MTrackJ Data Format (*.mdf)");
        filename, file_extension = os.path.splitext(fileName)
        print 'load the file', fileName
        lines_list = list(open(fileName, 'rb').read().split('\n'))
        for i in lines_list:
            i=i.split(' ')
            if len(i)>0:
                if i[0]=='Point':
                    #trim point and convert to int
                    i=[int(float(num)) for num in i[1:-1]]
                    #order ID x y t z c from mdf
                    #order in save_dots: txyz
                    #mdf is 1 indexed
                    rects = (i[4]-1, 0, 0, int(abs(par_obj.ori_width)), abs(par_obj.ori_height),i[3]-1,par_obj.curr_file)
                    #append if dots already in roi, otherwise create roi and append
                    if rects in par_obj.saved_ROI:
                        idx=par_obj.saved_ROI.index(rects)
                        par_obj.saved_dots[idx].append((i[4]-1,i[1]-1,i[2]-1))
                    else:
                        par_obj.saved_dots.append([(i[4]-1,i[1]-1,i[2]-1)])
                        par_obj.saved_ROI.append(rects)
    def overlay_prediction_btn_fn(self):
        par_obj.overlay=not par_obj.overlay
        self.goto_img_fn(par_obj.curr_z,par_obj.curr_t)

    def replot_fn(self):
            v2.eval_pred_show_fn(par_obj.curr_z,par_obj,self)

    def count_maxima_btn_fn(self):
        t0=time.time()
        par_obj.max_det=[]
        par_obj.min_distance = [float(self.count_txt_1.text()),float(self.count_txt_2.text()),float(self.count_txt_3.text())]
        par_obj.abs_thr =float(self.abs_thr_txt.text())
        #par_obj.z_cal =float(self.z_cal_txt.text())
        v2.count_maxima(par_obj,par_obj.curr_t,par_obj.curr_file)
        par_obj.show_pts = 1
        self.kernel_btn_fn()

    def report_progress(self,message):
        self.image_status_text.showMessage('Status: '+message)
        app.processEvents()
        
    def keyPressEvent(self, ev):
        """When the . and , keys are pressed"""
        if ev.key() == QtCore.Qt.Key_Period:
            self.Btn_fns.next_file(par_obj)
        elif ev.key() == QtCore.Qt.Key_Comma:
            self.Btn_fns.prev_file(par_obj)
        elif ev.key() == QtCore.Qt.Key_Up:
            self.Btn_fns.next_im(par_obj)
        elif ev.key() == QtCore.Qt.Key_Down:
            self.Btn_fns.prev_im(par_obj)
        elif ev.key() == QtCore.Qt.Key_Left:
            self.Btn_fns.prev_time(par_obj)
        elif ev.key() == QtCore.Qt.Key_Right:
            self.Btn_fns.next_time(par_obj)
    def wheelEvent(self, event):
        """When the mousewheel is rotated"""
        if event.delta() > 0:
            self.Btn_fns.next_im(par_obj)
        if event.delta() < 0:
            self.Btn_fns.prev_im(par_obj)
            
    def loadTrainFn(self):
        #Win_fn()
        channel_wid = QtGui.QWidget()
        channel_lay = QtGui.QHBoxLayout()
        channel_wid.setLayout(channel_lay)

        win.top_left_grid.addWidget(channel_wid,1,0,1,3)

        ButtonGroup=create_channel_objects(self,par_obj,par_obj.numCH)
        for x in ButtonGroup:
            channel_lay.addWidget(x)
            x.show()
            x.setChecked(True)

        channel_lay.addStretch()

        win_tab.setCurrentWidget(win)
        app.processEvents()
        self.checkChange()

    def on_key(self,event):
        if event.key==' ':
            if par_obj.draw_dots==True:
                self.save_dots_fn()
            else:
                self.save_roi_fn()
#        if event.key==' ':
#            if par_obj.draw_dots==True:
#            self.save_dots_fn()
        elif event.key == '.':
            self.Btn_fns.next_file(par_obj)
        elif event.key == ',':
            self.Btn_fns.prev_file(par_obj)
        elif event.key == "up":
            self.Btn_fns.next_im(par_obj)
        elif event.key == "down":
            self.Btn_fns.prev_im(par_obj)
        elif event.key == "left":
            self.Btn_fns.prev_time(par_obj)
        elif event.key == "right":
            self.Btn_fns.next_time(par_obj)
    def on_click(self,event):
        """When the image is clicked"""
        if event.button == 1: #left
            if(par_obj.draw_ROI == True):
                
                par_obj.mouse_down = True
                self.save_roi_fn()
            else:
                par_obj.mouse_down = True

        elif event.button == 3:#right
             #When the draw ROI functionality is enabled:
            par_obj.mouse_down = True

            if(par_obj.draw_ROI == True):
    
                self.x1 = event.xdata
                self.y1 = event.ydata
                par_obj.ori_x = event.xdata
                par_obj.ori_y = event.ydata

    def on_resize(self,event):
        """When the toolbar is activated, resize plot 2"""
        self.plt2.set_ylim(self.plt1.get_ylim())
        self.plt2.set_xlim(self.plt1.get_xlim())
        self.canvas2.draw()

    def on_motion(self,event):
        """When the mouse is being dragged"""
        #When the draw ROI functionality is enabled:
        if(par_obj.draw_ROI == True and par_obj.mouse_down == True and event.button == 3):
            #Finds current cursor position
            if (event.xdata != None and event.ydata != None):
                par_obj.ori_x_2 = event.xdata
                par_obj.ori_y_2 = event.ydata
                try:


                        self.plt1.lines.remove(self.l1[0])
                        self.plt1.lines.remove(self.l2[0])
                        self.plt1.lines.remove(self.l3[0])
                        self.plt1.lines.remove(self.l4[0])


                except:
                    pass
                self.plt1.autoscale(False)
                self.l1 = self.plt1.plot([self.x1, event.xdata], [self.y1, self.y1], '-' ,color='r')
                self.l2 = self.plt1.plot([event.xdata, event.xdata], [self.y1, event.ydata], '-' ,color='r')
                self.l3 = self.plt1.plot([event.xdata, self.x1], [ event.ydata,  event.ydata], '-' ,color='r')
                self.l4 = self.plt1.plot([self.x1, self.x1], [ event.ydata, self.y1], '-' ,color='r')


            #self.plt1.Line2D([event.xdata, event.xdata], [self.y1, event.ydata], transform=self.plt1.transData,  figure=self.plt1,color='r')
            #self.plt1.Line2D([event.xdata, self.x1], [ event.ydata,  event.ydata], transform=self.plt1.transData,  figure=self.plt1,color='r')
            #self.plt1.Line2D([self.x1, self.x1], [ event.ydata, self.y1], transform=self.plt1.transData,  figure=self.plt1,color='r')

            self.canvas1.draw()


    def on_unclick(self, event):
        """When the mouse is released"""

        par_obj.mouse_down = False
        self.on_resize(None) #ensures zoom is updated when scaling in figure1
        #If we are in the roi drawing phase
        if(par_obj.draw_ROI == True and event.button == 3):
            t2 = time.time()
            x = event.xdata
            y = event.ydata
            if (x != None and y != None):
                par_obj.rect_w = x - par_obj.ori_x
                par_obj.rect_h = y - par_obj.ori_y

                #Corrects the corrdinates if out of rectangle.
                if(x < 0): x=0
                if(y < 0): y=0
                if(x > par_obj.width): x=par_obj.width-1
                if(y > par_obj.height): y=par_obj.height-1
            else:
                par_obj.rect_w = par_obj.ori_x_2 - par_obj.ori_x
                par_obj.rect_h = par_obj.ori_y_2 - par_obj.ori_y
            t1 = time.time()
            print t1-t2
            
        #If we are in the dot drawing phase
        elif(par_obj.draw_dots == True and event.button == 1):
            x = int(np.round(event.xdata,0))
            y = int(np.round(event.ydata,0))

            #Are we with an existing box.
            if(x > par_obj.rects[1]-par_obj.roi_tolerance and x < (par_obj.rects[1]+ par_obj.rects[3])+par_obj.roi_tolerance and y > par_obj.rects[2]-par_obj.roi_tolerance and y < (par_obj.rects[2]+ par_obj.rects[4])+par_obj.roi_tolerance):
                appendDot = True
                #Appends dots to array if in an empty pixel.
                if(par_obj.dots.__len__()>0):
                    for i in range(0,par_obj.dots.__len__()):
                        if (x == par_obj.dots[i][1] and y == par_obj.dots[i][2]):
                            appendDot = False

                if(appendDot == True):
                    par_obj.dots.append((par_obj.curr_z,x,y))
                    i = par_obj.dots[-1]
                    self.plt1.autoscale(False)
                    self.plt1.plot([i[1]-5,i[1]+5],[i[2],i[2]],'-',color='r')
                    self.plt1.plot([i[1],i[1]],[i[2]-5,i[2]+5],'-',color='r')
                    self.canvas1.draw()
        elif(par_obj.remove_dots == True and event.button == 1):
            #par_obj.pixMap = QtGui.QPixmap(q2r.rgb2qimage(par_obj.imgs[par_obj.curr_z]))
            x = event.xdata
            y = event.ydata
            self.draw_saved_dots_and_roi()
            print 'curr_z',par_obj.curr_z
            print 'timepoint',par_obj.curr_t
            #Are we with an existing box.
            if(x > par_obj.rects[1]-par_obj.roi_tolerance and x < (par_obj.rects[1]+ par_obj.rects[3])+par_obj.roi_tolerance and y > par_obj.rects[2]-par_obj.roi_tolerance and y < (par_obj.rects[2]+ par_obj.rects[4])+par_obj.roi_tolerance):
                #Appends dots to array if in an empty pixel.
                if(par_obj.dots.__len__()>0):
                    for i in range(0,par_obj.dots.__len__()):
                        if ((abs(x -par_obj.dots[i][1])<3 and abs(y - par_obj.dots[i][2])<3)):
                            par_obj.dots.pop(i)
                            par_obj.saved_dots.append(par_obj.dots)
                            par_obj.saved_ROI.append(par_obj.rects)
                            #Reloads the roi so can edited again. It is now at the end of the array.
                            par_obj.dots = par_obj.saved_dots[-1]
                            par_obj.rects =  par_obj.saved_ROI[-1]
                            par_obj.saved_dots.pop(-1)
                            par_obj.saved_ROI.pop(-1)
                            break

            for i in range(0,self.plt1.lines.__len__()):
                self.plt1.lines.pop(0)
            self.dots_and_square(par_obj.dots,par_obj.rects,'y')
            self.canvas1.draw()

        elif(par_obj.select_ROI== True and event.button == 1):
            x = event.xdata
            y = event.ydata
            for b in range(0,par_obj.ROI_index.__len__()):
                dots = par_obj.saved_dots[par_obj.ROI_index[b]]
                rects = par_obj.saved_ROI[par_obj.ROI_index[b]]
                if(x > rects[1] and x < (rects[1]+ rects[3]) and y > rects[2] and y < (rects[2]+ rects[4])):

                    par_obj.roi_select = b
                    par_obj.dots = par_obj.saved_dots[par_obj.ROI_index[par_obj.roi_select]]
                    par_obj.rects =  par_obj.saved_ROI[par_obj.ROI_index[par_obj.roi_select]]
                    par_obj.saved_dots.pop(par_obj.ROI_index[par_obj.roi_select])
                    par_obj.saved_ROI.pop(par_obj.ROI_index[par_obj.roi_select])

                    for i in range(0,self.plt1.lines.__len__()):
                        self.plt1.lines.pop(0)
                    self.draw_saved_dots_and_roi()
                    self.dots_and_square(dots,rects,'y')
                    self.canvas1.draw()
                    self.sel_ROI_btn.setEnabled(False)
                    self.save_dots_btn.setEnabled(True)
                    self.remove_dots_btn.setEnabled(True)
                    par_obj.select_ROI= False
                    par_obj.draw_ROI = False
                    par_obj.draw_dots = True
        event.xdata=None
        event.ydata=None
    def dots_and_square(self, dots,rects,colour):


        #self.l5 = lines.Line2D([rects[1], rects[1]+rects[3]], [rects[2],rects[2]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.l6 = lines.Line2D([rects[1]+rects[3], rects[1]+rects[3]], [rects[2],rects[2]+rects[4]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.l7 = lines.Line2D([rects[1]+rects[3], rects[1]], [rects[2]+rects[4],rects[2]+rects[4]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.l8 = lines.Line2D([rects[1], rects[1]], [rects[2]+rects[4],rects[2]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.plt1.lines.extend([self.l5,self.l6,self.l7,self.l8])
        self.plt1.autoscale(False)
        self.plt1.plot([rects[1], rects[1]+rects[3]], [rects[2],rects[2]], '-',color=colour)
        self.plt1.plot([rects[1]+rects[3], rects[1]+rects[3]], [rects[2],rects[2]+rects[4]], '-',color=colour)
        self.plt1.plot([rects[1]+rects[3], rects[1]], [rects[2]+rects[4],rects[2]+rects[4]], '-', color=colour)
        #self.plt1.plot([rects[1]+rects[3], rects[1]], [rects[2]+rects[4],rects[2]+rects[4]], '-',  figure=self.plt1,color=colour)
        self.plt1.plot([rects[1], rects[1]], [rects[2]+rects[4],rects[2]], '-',color=colour)


            #Draws dots in list
        for i in iter(dots):
            self.plt1.plot([i[1]-5,i[1]+5],[i[2],i[2]],'-',color=colour)
            self.plt1.plot([i[1],i[1]],[i[2]-5,i[2]+5],'-',color=colour)




        return
    def makeCursor(self):
        m_LPixmap = QtGui.QPixmap(28, 28)
        bck = QtGui.QColor(168, 34, 3)
        bck.setAlpha(0)
        m_LPixmap.fill(bck)
        qp = QtGui.QPainter(m_LPixmap)
        qp.setPen(QtGui.QColor(0, 255, 0,200))
        qp.drawLine(14,0,14,28)
        qp.drawLine(0,14,28,14)
        qp.setOpacity(1.0)
        m_Cursor = QtGui.QCursor(m_LPixmap)
        qp.setOpacity(0.0)
        qp.end()
        return m_Cursor
    def on_enter(self,ev):
        #Changes cursor to the special crosshair on entering image pane.
        QtGui.QApplication.setOverrideCursor(self.m_Cursor)
        self.canvas1.setFocus()
    def on_leave(self,ev):
        QtGui.QApplication.restoreOverrideCursor()
    def save_roi_fn(self):
        #If there is no width or height either no roi is selected or it is too thin.
        success = v2.save_roi_fn(par_obj)
        if success == True:
            print ('Saved ROI')
            win.image_status_text.showMessage('Status: Select instances in region then click \'save Dots\' ')
            par_obj.draw_ROI = False
            par_obj.draw_dots = True
            win.save_ROI_btn.setEnabled(False)
            win.save_dots_btn.setEnabled(True)
            win.remove_dots_btn.setEnabled(True)
            win.sel_ROI_btn.setEnabled(False)
            par_obj.remove_dots = False
    def deleteDotsFn(self,sel_ROI_btn_fn):
        print('Dot deleted')
        par_obj.saved_dots.append(par_obj.dots)
        par_obj.saved_ROI.append(par_obj.rects)
        par_obj.dots = par_obj.saved_dots[par_obj.ROI_index[par_obj.roi_select]]
        par_obj.rects = par_obj.saved_ROI[par_obj.ROI_index[par_obj.roi_select]]
        par_obj.saved_dots.pop(par_obj.ROI_index[par_obj.roi_select])
        par_obj.saved_ROI.pop(par_obj.ROI_index[par_obj.roi_select])
        #Creates the qpainter object


        #Now we update a density image of the current Image.
        self.update_density_fn()
    def save_dots_fn(self):
        print('Saved Dots')
        win.image_status_text.showMessage('Status: Highlight new ROI or train. ')
        win.train_model_btn.setEnabled(True)
        par_obj.saved_dots.append(par_obj.dots)
        par_obj.saved_ROI.append(par_obj.rects)
        #self.draw_saved_dots_and_roi()
        self.save_ROI_btn.setEnabled(True)
        self.save_dots_btn.setEnabled(False)
        self.remove_dots_btn.setEnabled(False)
        self.sel_ROI_btn.setEnabled(True)
        self.clear_dots_btn.setEnabled(True)
        par_obj.draw_ROI = True
        par_obj.draw_dots = False
        par_obj.remove_dots = False
        par_obj.dots_past = par_obj.dots
        par_obj.dots = []
        par_obj.rects = np.zeros((1,5))
        par_obj.ori_x=0
        par_obj.ori_y=0
        par_obj.rect_w=0
        par_obj.rect_h =0


        #self.goto_img_fn(par_obj.curr_z,par_obj.curr_t)
        #Now we update a density image of the current Image.
        self.update_density_fn()


    def update_density_fn(self):
        #Construct empty array for current image.
        tpt=par_obj.curr_t
        zslice=par_obj.curr_z
        fileno=par_obj.curr_file
        v2.update_com_fn(par_obj,tpt,zslice,fileno)

        self.goto_img_fn(par_obj.curr_z,par_obj.curr_t)
        '''
        self.plt2.cla()
        self.plt2.imshow(par_obj.data_store[tpt]['dense_arr'][zslice])
        self.plt2.set_xticklabels([])
        self.plt2.set_yticklabels([])
        '''
        self.canvas2.draw()
    def draw_saved_dots_and_roi(self):
        for i in range(0,par_obj.subdivide_ROI.__len__()):
            if(par_obj.subdivide_ROI[i][0] ==par_obj.curr_z and par_obj.subdivide_ROI[i][5] == par_obj.curr_t and par_obj.subdivide_ROI[i][6] == par_obj.curr_file):
                rects =par_obj.subdivide_ROI[i]
                dots = []
                self.dots_and_square(dots,rects,'w')
        for i in range(0,par_obj.saved_dots.__len__()):
            if(par_obj.saved_ROI[i][0] == par_obj.curr_z and par_obj.saved_ROI[i][5] == par_obj.curr_t and par_obj.saved_ROI[i][6] == par_obj.curr_file):
                dots = par_obj.saved_dots[i]
                rects = par_obj.saved_ROI[i]
                self.dots_and_square(dots,rects,'w')

    def goto_img_fn(self,zslice=None,tpt=None,imno=None):
        #update current image/slice/timepoint if changed
        if zslice!=None:
            par_obj.curr_z=zslice
        if tpt!=None:
            par_obj.curr_t=tpt
        if imno!=None:
            par_obj.curr_file=imno
        #Goto and evaluate image function.
        v2.goto_img_fn_new(par_obj,self)
        #updates Z-cal
        if par_obj.filehandlers[par_obj.curr_file].z_calibration !=0:
            par_obj.z_cal = par_obj.filehandlers[par_obj.curr_file].z_calibration
        self.z_cal_txt.setText(str(par_obj.z_cal)[0:6])
        #v2.return_imRGB_slice_new(par_obj,zslice,tpt)
        #self.draw_saved_dots_and_roi()
        #reset controls and box drawing
        par_obj.dots = []
        par_obj.rects = np.zeros((1,5))
        par_obj.select_ROI= False
        par_obj.draw_ROI = True
        par_obj.draw_dots = False
        par_obj.remove_dots = False
        self.save_ROI_btn.setEnabled(True)
        self.save_dots_btn.setEnabled(False)
        self.remove_dots_btn.setEnabled(False)
        self.sel_ROI_btn.setEnabled(True)
        par_obj.ROI_index=[]
        #app.processEvents()

    def sel_ROI_btn_fn(self):
        par_obj.ROI_index =[]
        if(par_obj.select_ROI== False):
            self.save_ROI_btn.setEnabled(False)
            par_obj.select_ROI= True
            par_obj.draw_ROI = False
            par_obj.draw_dots = False
            par_obj.remove_dots = False
            for i in range(0,par_obj.saved_ROI.__len__()):
                if(par_obj.saved_ROI[i][0] == par_obj.curr_z and par_obj.saved_ROI[i][5] == par_obj.curr_t and par_obj.saved_ROI[i][6] == par_obj.curr_file):
                    par_obj.ROI_index.append(i)
            for b in range(0,par_obj.ROI_index.__len__()):
                dots = par_obj.saved_dots[par_obj.ROI_index[b]]
                rects = par_obj.saved_ROI[par_obj.ROI_index[b]]
                self.dots_and_square(dots,rects,'y')
        else:
            self.save_ROI_btn.setEnabled(True)
            par_obj.select_ROI= False
            par_obj.draw_ROI = True
            self.draw_saved_dots_and_roi()
    def remove_dots_btn_fn(self):
        if(par_obj.remove_dots == False):
            par_obj.remove_dots = True
            par_obj.draw_dots = False
        else:
            par_obj.remove_dots = False
            par_obj.draw_dots = True
    def clear_dots_fn(self):
        par_obj.saved_dots = []
        par_obj.saved_ROI = []
        for fileno in par_obj.data_store['dense_arr']:
            for tpt in par_obj.data_store['dense_arr'][fileno]:
                par_obj.data_store['dense_arr'][fileno][tpt]={}
        #par_obj.data_store['dense_arr'][imno].clear()
        self.goto_img_fn(par_obj.curr_z)
        self.update_density_fn()
        self.train_model_btn.setEnabled(False)
        self.clear_dots_btn.setEnabled(False)
    def train_model_btn_fn(self):
        self.image_status_text.showMessage('Training Ensemble of Decision Trees. ')
        #added to make sure current timepoint has all features precalculated
        v2.im_pred_inline_fn_new(par_obj, self,range(par_obj.max_z+1),[par_obj.curr_t],[par_obj.curr_file],True)

        for i in range(0,par_obj.saved_ROI.__len__()):

            zslice = par_obj.saved_ROI[i][0]
            tpt =par_obj.saved_ROI[i][5]
            imno =par_obj.saved_ROI[i][6]
            print 'calculating features, time point',tpt+1,' image slice ',zslice+1
            v2.im_pred_inline_fn_new(par_obj, self,[zslice],[tpt],[imno],threaded=False)

        par_obj.f_matrix=[]
        par_obj.o_patches=[]
        t0=time.time()
        print
        for i in par_obj.saved_ROI:
            #zslice = par_obj.saved_ROI[i][0]
            #tpt =par_obj.saved_ROI[i][5]
            #imno =par_obj.saved_ROI[i][6]
            v2.update_training_samples_fn_new_only(par_obj,self,i)
        print time.time()-t0
        t0=time.time()
        self.image_status_text.showMessage('Training Model')
        v2.train_forest(par_obj,self,0)
        self.image_status_text.showMessage('Evaluating Images with the Trained Model. ')
        app.processEvents()
        v2.evaluate_forest_new(par_obj,self, False,0,range(par_obj.max_z+1),[par_obj.curr_t],[par_obj.curr_file])
        #v2.make_correction(par_obj, 0)
        self.image_status_text.showMessage('Model Trained. Continue adding samples, or click \'Save Training Model\'. ')
        par_obj.eval_load_im_win_eval = True
        par_obj.show_pts= 0
        self.kernel_btn_fn()
        self.save_model_btn.setEnabled(True)
        self.count_maxima_btn.setEnabled(True)
        self.evaluate_btn.setEnabled(True)
        if par_obj.double_train==True:
            self.double_train_model_btn_fn()
    def double_train_model_btn_fn(self):
        self.image_status_text.showMessage('Training Ensemble of Decision Trees. ')
        #added to make sure current timepoint has all features precalculated
        for i in range(0,par_obj.saved_ROI.__len__()):
            zslice = par_obj.saved_ROI[i][0]
            tpt =par_obj.saved_ROI[i][5]
            imno =par_obj.saved_ROI[i][6]
            v2.evaluate_forest_new(par_obj,self, False,0,[zslice],[tpt],[imno])

        v2.im_pred_inline_fn_new(par_obj, self,range(par_obj.max_z+1),[par_obj.curr_t],[par_obj.curr_file],'auto')

        for i in range(0,par_obj.saved_ROI.__len__()):
            zslice = par_obj.saved_ROI[i][0]
            tpt =par_obj.saved_ROI[i][5]
            imno =par_obj.saved_ROI[i][6]
            print 'calculating features, time point',tpt+1,' image slice ',zslice+1
            v2.im_pred_inline_fn_new(par_obj, self,[zslice],[tpt],[imno],threaded='auto')

        par_obj.f_matrix=[]
        par_obj.o_patches=[]
        t0=time.time()
        print
        for i in par_obj.saved_ROI:
            v2.update_training_samples_fn_new_only(par_obj,self,i,'double_feat_arr')
        print time.time()-t0
        t0=time.time()
        self.image_status_text.showMessage('Training Model')
        v2.train_forest(par_obj,self,1)
        self.image_status_text.showMessage('Evaluating Images with the Trained Model. ')
        app.processEvents()
        v2.evaluate_forest_auto(par_obj,self, False,1,range(par_obj.max_z+1),[par_obj.curr_t],[par_obj.curr_file])
        #v2.make_correction(par_obj, 0)
        self.image_status_text.showMessage('Model Trained. Continue adding samples, or click \'Save Training Model\'. ')
        par_obj.eval_load_im_win_eval = True
        par_obj.show_pts= 0
        self.kernel_btn_fn()
        self.save_model_btn.setEnabled(True)
        self.count_maxima_btn.setEnabled(True)
        self.evaluate_btn.setEnabled(True)
    def sigmaOnChange(self,text,imno=0):
        if (text != ""):
            par_obj.sigma_data = float(text)
            par_obj.gaussian_im_max=[]
            v2.refresh_all_density(par_obj)
            par_obj.min_distance[0]= int(round(par_obj.sigma_data))
            par_obj.min_distance[1]= int(round(par_obj.sigma_data))
            par_obj.min_distance[2]= int(round(par_obj.sigma_data))
            self.count_txt_1.setText(str(par_obj.min_distance[0]))
            self.count_txt_2.setText(str(par_obj.min_distance[1]))
            self.count_txt_3.setText(str(par_obj.min_distance[2]))

            self.update_density_fn()
    def feat_scale_change_btn_fn(self):
        self.feat_scale_change_btn.setEnabled(False)
        print('Training Features')
        v2.processImgs()
        v2.refresh_all_density()
        self.feat_scale_change_btn.setEnabled(True)
        self.image_status_text.showMessage('Model Trained. Continue adding samples, or click \'Save Training Model\'. ')
        self.save_model_btn.setEnabled(True)
        v2.eval_pred_show_fn(par_obj.curr_z,par_obj,self)

    def kernel_btn_fn(self,set=False):
        """Shows the kernels on the image."""
        if set == 'Kernel':
            par_obj.show_pts =0
        elif set=='Probability':
            par_obj.show_pts =1
        elif set=='Counts':
            par_obj.show_pts =2
        elif set==False:
            par_obj.show_pts = (par_obj.show_pts+ 1)%3

        print 'show',par_obj.show_pts

        if par_obj.show_pts == 0:
            self.kernel_show_btn.setText('Showing Kernel')
            self.update_density_fn()
        elif par_obj.show_pts == 1:
            self.kernel_show_btn.setText('Showing Probability')
            v2.goto_img_fn_new(par_obj,self)
        elif par_obj.show_pts == 2:
            self.kernel_show_btn.setText('Showing Counts')
            v2.goto_img_fn_new(par_obj,self)

    def predShowFn(self):
        #Captures the button event.
        v2.eval_pred_show_fn(par_obj.curr_z,par_obj,self)

    def saveForestFn(self):

        path = os.path.expanduser('~')+'/.densitycount/models/'
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        local_time = time.asctime( time.localtime(time.time()) )
        par_obj.modelName = str(self.save_model_name_txt.text())
        par_obj.modelDescription = str(self.save_model_desc_txt.text())

        cleanString = re.sub('\W+', '', par_obj.modelName )

        basename = path+"pv20"
        suffix = str(int(round(time.time(),0)))
        filename = "_".join([basename,suffix,str(cleanString),  ".mdla"])
        save_file = {}

        #Formats image to make a better icon.
        if par_obj.save_im.shape[0] > 300 and  par_obj.save_im.shape[1] > 300:
            save_im = np.zeros((300, 300, 3))
            cent_y = np.floor(par_obj.save_im.shape[0]/2).astype(np.int32)
            cent_x = np.floor(par_obj.save_im.shape[1]/2).astype(np.int32)
            if par_obj.save_im.shape[2]> 2:
                save_im[:,:,0] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,0]
                save_im[:,:,1] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,1]
                save_im[:,:,2] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,2]
            else:
                save_im[:,:,0] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,0]
                save_im[:,:,1] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,0]
                save_im[:,:,2] =  par_obj.save_im[cent_y-150:cent_y+150, cent_x-150:cent_x+150,0]
        else:
            save_im = np.zeros((par_obj.save_im.shape[0], par_obj.save_im.shape[1], 3))
            if par_obj.save_im.shape[2]> 2:
                save_im[:,:,0] = par_obj.save_im[:, :, 0]
                save_im[:,:,1] = par_obj.save_im[:, :, 1]
                save_im[:,:,2] = par_obj.save_im[:, :, 2]
            else:
                save_im[:,:,0] = par_obj.save_im[:, :,0]
                save_im[:,:,1] = par_obj.save_im[:, :,0]
                save_im[:,:,2] = par_obj.save_im[:, :,0]
                
        par_obj.file_ext=''        #added for backwards compatibility when saving

        save_file ={"name":par_obj.modelName,'description':par_obj.modelDescription,"c":par_obj.c,"M":par_obj.M,\
        "sigma_data":par_obj.sigma_data, "model":par_obj.RF, "date":local_time, "feature_type":par_obj.feature_type, \
        "feature_scale":par_obj.feature_scale, "ch_active":par_obj.ch_active, "limit_ratio_size":par_obj.limit_ratio_size, \
        "max_depth":par_obj.max_depth, "min_samples":par_obj.min_samples_split, "min_samples_leaf":par_obj.min_samples_leaf,\
        "max_features":par_obj.max_features, "num_of_tree":par_obj.num_of_tree, "file_ext":par_obj.file_ext, "imFile":save_im,\
        "resize_factor":par_obj.resize_factor, "min_distance":par_obj.min_distance, "abs_thr":par_obj.abs_thr,"rel_thr":par_obj.rel_thr,"max_det": par_obj.max_det};

        pickle.dump(save_file, open(filename, "wb"))
        self.save_model_btn.setEnabled(False)
        self.report_progress('Model Saved.')

    def checkChange(self):

        #v2.eval_goto_img_fn(par_obj, self,par_obj.curr_z,par_obj.curr_t)
        v2.load_and_initiate_plots(par_obj, self)
        self.sigmaOnChange(par_obj.sigma_data) #makes sure Z_calibration is set



# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    #generate layout
    app = QtGui.QApplication([])
    QtGui.QApplication.setQuitOnLastWindowClosed(True)
    # Create and display the splash screen
    splash_pix = QtGui.QPixmap('splash_loading.png')
    splash = QtGui.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()
    #Creates tab widget.
    win_tab = QtGui.QTabWidget()
    #Creates win, an instance of QWidget
    par_obj  = parameterClass()
    win = Win_fn(par_obj)
    loadWin= Load_win_fn(par_obj,win)

    #Adds win tab and places button in par_obj.
    win_tab.addTab(loadWin, "Load Images")
    win_tab.addTab(win, "Train Model")

    #Defines size of the widget.
    win_tab.resize(1000,600)

    time.sleep(0.2)
    splash.finish(win_tab)

    win_tab.showMaximized()
    win_tab.activateWindow()
    #Automates the loading for testing.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
