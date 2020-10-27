#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Main script for running QuantiFly training.
"""CytoCensus Software v0.1

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
import time
import errno
import os
import os.path
import re
import pickle
import sys
from multiprocessing import freeze_support
freeze_support()
from PyQt5 import QtGui, QtCore, QtWidgets  # ,QtWebKit
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

#import numdifftools as ndt
from common.common_navigation import navigation_setup, create_channel_objects, btn_fn, on_about
from parameters.parameter_object import ParameterClass
from ROI.user_ROI import ROI
#from functions import v2_functions
from features import local_features
from fileio import file_handler
from common import common_navigation
from functions.maxima import count_maxima
from functions import v2_functions as v2

class fileDialog(QtWidgets.QMainWindow):
    """The dialog for loading images"""

    def __init__(self, parent):
        super(fileDialog, self).__init__()
        self.parent = parent
        self.initUI()
        self.parent.config = {}

        try:
            with open(os.path.expanduser('~')+'/.densitycount/config.p', "rb") as pfile:
                self.parent.config = pickle.load(pfile)
                self.parent.filepath = self.parent.config['filepath']
        except Exception: #catch all because pickle can throw loads of Exceptions
            self.parent.filepath = os.path.expanduser('~')+'/'
            try:

                os.makedirs(os.path.expanduser('~')+'/.densitycount/')
            except OSError as exc:
                if exc.errno != errno.EACCES:
                    print ('Write permission to '+self.parent.filepath+' denied')
                else:
                    print ('unable to make directory: ', os.path.expanduser(
                    '~') + '/.densitycount/')

    def initUI(self):

        self.text_edit = QtWidgets.QTextEdit()
        self.setCentralWidget(self.text_edit)
        self.statusBar()

        openFile = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Open', self)

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

        par_obj.file_array = []
        path = None
        for path in QtWidgets.QFileDialog.getOpenFileNames(self, 'Open file', self.parent.filepath, 'Images(*.tif *.tiff *.mrc *.dv);;'):
            if path != '' and path !='Images(*.tif *.tiff *.mrc *.dv)':
                if type(path) is not list:
                    par_obj.file_array.append(path)
                else:
                    par_obj.file_array=par_obj.file_array+path

        if par_obj.file_array is []:
            return

        self.parent.config['filepath'] = str(
            QtCore.QFileInfo(path).absolutePath())+'/'
        pickle.dump(self.parent.config, open(
            str(os.path.expanduser('~')+'/.densitycount/config.p'), "wb"))

        self.parent.image_status_text.showMessage('Status: Loading Images. ')

        success, updateText = v2.import_data_fn(par_obj, par_obj.file_array)

        self.parent.image_status_text.showMessage(updateText)
        if success is True:
            self.parent.updateAfterImport()
            #self.parent.resize_factor_text.hide()
            #self.parent.resize_factor_input.hide()
        else:
            self.parent.image_status_text.showMessage(
                'Image import unsuccesful. Use TIF files')
        return


class Load_win_fn(QtWidgets.QWidget):
    """The class for loading and processing images"""

    def __init__(self):  # ,par_obj,win):
        super(Load_win_fn, self).__init__()

        # Load images button
        load_im_lo = QtWidgets.QHBoxLayout()
        self.loadImages_button = QtWidgets.QPushButton("Load Images", self)
        load_im_lo.addWidget(self.loadImages_button)
        load_im_lo.addStretch()
        self.ex = fileDialog(self)
        self.loadImages_button.clicked.connect(self.ex.showDialog)

        # about button
        about = QtWidgets.QPushButton('About', self)
        load_im_lo.addWidget(about)
        about.clicked.connect(lambda: on_about(self))
        # SigmaData input field.

        resize_lo = QtWidgets.QHBoxLayout()

        sampling_lo = QtWidgets.QHBoxLayout()
        max_features_lo = QtWidgets.QHBoxLayout()

        self.samplingText = QtWidgets.QLabel('Sampling :')
        self.samplingText.setToolTip(
            '5 is a good default, try setting this lower for rare and small objects')
        self.samplingText.resize(40, 20)
        self.samplingText.hide()
        self.sampling_input = QtWidgets.QLineEdit(
            str(par_obj.limit_ratio_size))
        self.sampling_input.setToolTip(
            '5 is a good default, try setting this lower for rare and small objects')
        self.sampling_input.resize(10, 10)
        self.sampling_input.textChanged[str].connect(self.sampling_change)
        self.sampling_input.hide()
        sampling_lo.addWidget(self.samplingText)
        sampling_lo.addWidget(self.sampling_input)
        sampling_lo.addStretch()
        #self.featuresText = QtWidgets.QLabel('Max size Random Feature subset')
        #self.featuresText.resize(40, 20)
        # self.featuresText.hide()
        #self.features_input = QtWidgets.QLineEdit(str(par_obj.max_features))
        # self.features_input.resize(10,10)
        # self.features_input.textChanged[str].connect(self.features_change)
        # self.features_input.hide()
        # max_features_lo.addWidget(self.featuresText)
        # max_features_lo.addWidget(self.features_input)

        self.resize_factor_text = QtWidgets.QLabel('Resize Factor:')
        self.resize_factor_text.resize(40, 20)
        self.resize_factor_text.setToolTip(
            'Subsample image for speed of calculation')
        self.resize_factor_input = QtWidgets.QLineEdit(
            str(par_obj.resize_factor))
        self.resize_factor_input.setToolTip(
            'Subsample image for speed of calculation')
        self.resize_factor_input.resize(10, 10)
        self.resize_factor_input.textChanged[str].connect(
            self.resize_factor_change)
        resize_lo.addWidget(self.resize_factor_text)
        resize_lo.addWidget(self.resize_factor_input)
        resize_lo.addStretch()

        # Channel dialog generation.
        Channel_Select_lo = QtWidgets.QVBoxLayout()
        Channel_button_lo = QtWidgets.QHBoxLayout()
        self.Text_CHopt = QtWidgets.QLabel(
            'Please select which channels you want to include in the feature calculation')
        Channel_Select_lo.addWidget(self.Text_CHopt)
        self.Text_CHopt.resize(500, 40)
        self.Text_CHopt.hide()
        # Object factory for channel selection.

        ButtonGroup = create_channel_objects(self, par_obj, 10, True)
        for cbx in ButtonGroup:
            cbx[0].hide()
            Channel_button_lo.addWidget(cbx[0])
        Channel_Select_lo.addLayout(Channel_button_lo)
        Channel_button_lo.addStretch()
        Channel_Select_lo.addStretch()

        # setup preview plot
        self.figure1 = Figure(figsize=(2, 2))
        self.canvas1 = FigureCanvas(self.figure1)
        self.plt1 = self.figure1.add_subplot(1, 1, 1)
        self.resize(100, 100)
        self.figure1.subplots_adjust(
            left=0.001, right=0.999, top=0.999, bottom=0.001)
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])
        self.canvas1.hide()
        par_obj.rect_h = 0
        par_obj.rect_w = 0
        canvas_lo = QtWidgets.QHBoxLayout()
        canvas_lo.addWidget(self.canvas1)
        canvas_lo.addStretch()

        # Image range selection
        Im_Range_lo = QtWidgets.QVBoxLayout()
        # Image Details
        self.Text_FrmOpt2 = QtWidgets.QLabel()
        Im_Range_lo.addWidget(self.Text_FrmOpt2)
        self.Text_FrmOpt2.hide()
        self.Text_FrmOpt4 = QtWidgets.QLabel()
        Im_Range_lo.addWidget(self.Text_FrmOpt4)
        self.Text_FrmOpt4.hide()

        # Image frames dialog.
        Text_FrmOpt1_panel = QtWidgets.QHBoxLayout()
        self.Text_FrmOpt1 = QtWidgets.QLabel(
            'Please choose the z-slices you wish to use for training. Use \'-\' to indicate a range:')
        self.Text_FrmOpt1.hide()
        Im_Range_lo.addLayout(Text_FrmOpt1_panel)

        # Image frames input.
        linEdit_Frm_panel = QtWidgets.QHBoxLayout()
        self.linEdit_Frm = QtWidgets.QLineEdit()
        self.linEdit_Frm.hide()
        linEdit_Frm_panel.addWidget(self.linEdit_Frm)
        linEdit_Frm_panel.addStretch()
        Im_Range_lo.addLayout(linEdit_Frm_panel)

        Text_FrmOpt3_panel = QtWidgets.QHBoxLayout()
        self.Text_FrmOpt3 = QtWidgets.QLabel()
        self.Text_FrmOpt3.setText(
            'Please choose the time-points you wish to use for training. Use either \',\' to separate individual frames or a \'-\' to indicate a range:')
        self.Text_FrmOpt3.hide()
        Text_FrmOpt1_panel.addWidget(self.Text_FrmOpt1)
        Text_FrmOpt3_panel.addWidget(self.Text_FrmOpt3)
        Im_Range_lo.addLayout(Text_FrmOpt3_panel)

        linEdit_Frm_panel2 = QtWidgets.QHBoxLayout()
        self.linEdit_Frm2 = QtWidgets.QLineEdit()
        self.linEdit_Frm2.hide()
        linEdit_Frm_panel2.addWidget(self.linEdit_Frm2)
        linEdit_Frm_panel2.addStretch()
        Im_Range_lo.addLayout(linEdit_Frm_panel2)

        # Feature calculation type to perform:
        self.Text_Radio = QtWidgets.QLabel(
            'Feature select which kind of feature detection you would like to use:')
        self.Text_Radio.resize(500, 40)
        self.Text_Radio.hide()
        self.radio_group = QtWidgets.QButtonGroup(self)  # Number
        self.radio0 = QtWidgets.QRadioButton("Basic", self)
        self.radio0.setToolTip(
            '13 features/channel (Set Feature size) - Fast, less accurate')
        self.radio1 = QtWidgets.QRadioButton("Detailed", self)
        self.radio1.setToolTip(
            '21 features/channel (Set Feature size) - Slower, more accurate')
        self.radio2 = QtWidgets.QRadioButton("Pyramid (Default)", self)
        self.radio2.setToolTip(
            '26 features/channel, calculated more efficiently. No need to set feature size')
        self.radio3 = QtWidgets.QRadioButton("Histogram equalised", self)
        self.radio3.setToolTip(
            'Pyramid features but equalised for image variation. Try if your imaging conditions vary a lot - Slowest')
        self.radio4 = QtWidgets.QRadioButton("Radial", self)
        self.radio4.setToolTip(
            'Radial features - Try if your objects are circular but vary in size')
        self.radio5 = QtWidgets.QRadioButton("2-layer", self)
        self.radio5.setToolTip(
            'Calculate features of the probability map, similar to autocontext. Slower, and risks over-training')
        self.radio2.setChecked(True)
        self.radio_group.addButton(self.radio0)
        self.radio_group.addButton(self.radio1)
        self.radio_group.addButton(self.radio2)
        self.radio_group.addButton(self.radio3)
        self.radio_group.addButton(self.radio4)
        self.radio_group.addButton(self.radio5)

        self.radio0.hide()
        self.radio1.hide()
        self.radio2.hide()
        self.radio3.hide()
        self.radio4.hide()
        self.radio5.hide()
        Radio_Layout = QtWidgets.QVBoxLayout()

        Radio_Layout.addWidget(self.Text_Radio)
        Radio_Layout.addWidget(self.radio0)
        Radio_Layout.addWidget(self.radio1)
        Radio_Layout.addWidget(self.radio2)
        Radio_Layout.addWidget(self.radio3)
        Radio_Layout.addWidget(self.radio4)
        Radio_Layout.addWidget(self.radio5)
        Radio_Layout.addStretch()

        # Feature sigma, if basic or detailed features chosen
        sigma_lo = QtWidgets.QHBoxLayout()
        self.feature_scaleText = QtWidgets.QLabel('Feature size (sigma):')
        self.feature_scaleText.resize(40, 20)
        self.feature_scaleText.hide()
        self.feature_scale_input = QtWidgets.QLineEdit(
            str(par_obj.feature_scale))
        self.feature_scale_input.resize(10, 10)
        self.feature_scale_input.textChanged[str].connect(
            self.feature_scale_change)
        self.feature_scale_input.hide()
        sigma_lo.addWidget(self.feature_scaleText)
        sigma_lo.addWidget(self.feature_scale_input)
        sigma_lo.addStretch()

        # Load images button
        Confirm_im_lo = QtWidgets.QHBoxLayout()
        self.confirmImages_btn = QtWidgets.QPushButton("Confirm Images")
        self.confirmImages_btn.clicked.connect(self.confirmImages_btn_fn)
        self.confirmImages_btn.setEnabled(False)

        # Move to training button.
        self.selIntButton = QtWidgets.QPushButton("Goto Training")
        self.selIntButton.clicked.connect(win.loadTrainFn)
        self.selIntButton.setEnabled(False)

        Confirm_im_lo.addWidget(self.confirmImages_btn)
        Confirm_im_lo.addWidget(self.selIntButton)
        Confirm_im_lo.addStretch()
        self.image_status_text = QtWidgets.QStatusBar()
        self.image_status_text.setStyleSheet("QLabel { color : green }")
        self.image_status_text.showMessage(
            'Status: Highlight training images in folder. ')

        # Set up Vertical layout
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addLayout(load_im_lo)
        vbox.addLayout(resize_lo)
        vbox.addLayout(sampling_lo)
        vbox.addLayout(max_features_lo)
        vbox.addLayout(Channel_Select_lo)
        vbox.addLayout(canvas_lo)
        vbox.addLayout(Im_Range_lo)
        vbox.addLayout(Radio_Layout)
        vbox.addLayout(sigma_lo)
        vbox.addStretch()
        vbox.addLayout(Confirm_im_lo)
        vbox.addWidget(self.image_status_text)

        # File browsing functions
        #layout = QtWidgets.QVBoxLayout()

    def resize_factor_change(self, text):
        """Updates on change of image resize parameter"""
        if text != "":
            par_obj.resize_factor = int(text)

    def sampling_change(self, text):
        """Updates on change of sampling"""
        if text != "":
            par_obj.limit_ratio_size = float(text)

    def features_change(self, text):
        """Updates on change of feature number"""
        if text != "":
            par_obj.max_features = int(text)

    def feature_scale_change(self, text):
        """Updates on change of feature scale"""
        if text != "":
            par_obj.feature_scale = float(text)

            for fileno in range(0, par_obj.max_file):
                for tp in range(0, par_obj.max_t):
                    par_obj.data_store['feat_arr'][fileno][tp] = {}

    def updateAfterImport(self):
        """Specific to ui updates"""
        if par_obj.max_z > 1:

            # self.linEdit_Frm.setText('1-'+str(par_obj.max_z))
            self.Text_FrmOpt2.setText('There are '+str(par_obj.max_z)+' z-slices in total. The image has dimensions x: '
                                      + str(par_obj.ori_width)+' and y: '+str(par_obj.ori_height))
            if par_obj.max_t > 1:
                # self.linEdit_Frm2.setText('1-'+str(par_obj.max_t))
                self.Text_FrmOpt4.setText(
                    'There are '+str(par_obj.max_t)+' timepoints in total.')
                self.linEdit_Frm2.show()
                self.Text_FrmOpt4.show()
                self.Text_FrmOpt3.show()
            self.Text_FrmOpt1.show()
            self.Text_FrmOpt2.show()
            self.linEdit_Frm.show()

        self.confirmImages_btn.setEnabled(True)

        self.plt1.cla()

        if par_obj.ex_img.max() != 0:
            self.plt1.imshow(par_obj.ex_img[:,:,:3]/par_obj.ex_img.max())
        else:
            self.plt1.imshow(par_obj.ex_img[:,:,:3])

        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])
        self.canvas1.show()
        self.canvas1.draw()

        par_obj.ch_active = []
        if par_obj.numCH > 0:
            self.Text_CHopt.show()
            for i in range(0, par_obj.numCH):
                self.CH_cbx[i].show()
                par_obj.ch_active.append(i)
        else:
            par_obj.ch_active.append(0)
        # self.CH_cbx1.stateChange()
        self.feature_scale_input.show()
        self.feature_scaleText.show()
        # self.features_input.show()
        # self.featuresText.show()
        self.radio0.show()
        self.radio1.show()
        self.radio2.show()
        self.radio3.show()
        self.radio4.show()
        self.sampling_input.show()
        self.samplingText.show()

        self.Text_Radio.show()

    def hyphen_range(self, s):
        """ yield each integer from a complex range string like "1-9,12, 15-20,23"

        >>> list(hyphen_range('1-9,12, 15-20,23'))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 23]

        >>> list(hyphen_range('1-9,12, 15-20,2-3-4'))
        Traceback (most recent call last):
            ...
        ValueError: format error in 2-3-4
        """
        s_list = []
        if s != '':
            for x in s.split(','):
                elem = x.split('-')
                if len(elem) == 1:  # a number
                    s_list.append(int(elem[0])-1)
                elif len(elem) == 2:  # a range inclusive
                    start, end = map(int, elem)
                    for i in xrange(start, end+1):
                        s_list.append(i-1)
                else:  # more than one hyphen
                    raise ValueError('format error in %s' % x)
        return s_list

    def confirmImages_btn_fn(self):
        if par_obj.max_t > 1:
            tmStr = self.linEdit_Frm2.text()
            if tmStr != '':
                par_obj.tpt_list = self.hyphen_range(tmStr)
            else:
                par_obj.tpt_list = range(par_obj.max_t)
        else:
            par_obj.tpt_list = [0]

        if par_obj.max_z > 0: #if current file has multiple z
            #this fails if first file has single z
            fmStr = self.linEdit_Frm.text()
            if fmStr != '':  # catch empty limit
                par_obj.user_max_z = max(self.hyphen_range(fmStr))
                par_obj.user_min_z = min(self.hyphen_range(fmStr))
            else: #excessively large default max_z for training
                par_obj.user_max_z = 1000
                par_obj.user_min_z = 0
        else:
            par_obj.user_max_z = 0
            par_obj.user_min_z = 0

        self.image_status_text.showMessage(
            'Status: Images loaded. Click \'Goto Training\'')
        self.selIntButton.setEnabled(True)
        v2.setup_parameters(self, par_obj)

        if self.radio0.isChecked():
            par_obj.feature_type = 'basic'
        if self.radio1.isChecked():
            par_obj.feature_type = 'fine'
        if self.radio2.isChecked():
            par_obj.feature_type = 'pyramid'
        if self.radio3.isChecked():
            par_obj.feature_type = 'histeq'
        if self.radio4.isChecked():
            par_obj.feature_type = 'radial'
        print (par_obj.feature_type)

    def report_progress(self, message):
        self.image_status_text.showMessage('Status: ' + message)
        app.processEvents()


class Win_fn(QtWidgets.QWidget):
    """Class which houses main training functionality"""

    def __init__(self, par_obj):
        super(Win_fn, self).__init__()
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount = 1
        # Sets up the figures for displaying images.
        self.figure1 = Figure(figsize=(8, 8), dpi=100)
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure1.patch.set_facecolor('grey')
        self.cursor = ROI(self, par_obj)

        self.plt1 = self.figure1.add_subplot(1, 1, 1)
        im_RGB = np.zeros((512, 512))
        # Makes sure it spans the whole figure.
        self.figure1.subplots_adjust(
            left=0.001, right=0.999, top=0.999, bottom=0.001)

        self.toolbar = NavigationToolbar(self.canvas1, self)
        self.toolbar.actionTriggered.connect(self.on_resize)
        self.toolbar.actionTriggered.connect(self.on_resize)

        self.plt1.imshow(im_RGB)

        # Removes the tick labels
        self.plt1.set_xticklabels([])
        self.plt1.set_yticklabels([])

        # Initialises the second figure.
        self.figure2 = Figure(figsize=(8, 8), dpi=100)
        self.canvas2 = FigureCanvas(self.figure2)
        self.figure2.patch.set_facecolor('grey')
        self.plt2 = self.figure2.add_subplot(1, 1, 1)
        self.plt2_is_clear = False

        # Makes sure it spans the whole figure.
        self.figure2.subplots_adjust(
            left=0.001, right=0.999, top=0.999, bottom=0.001)
        self.plt2.imshow(im_RGB)
        self.plt2.set_xticklabels([])
        self.plt2.set_yticklabels([])

        # The ui for training
        self.count_txt = QtWidgets.QLabel()
        self.image_num_txt = QtWidgets.QLabel()
        box = QtWidgets.QVBoxLayout()
        self.setLayout(box)

        # Widget containing the top panel.
        top_panel = QtWidgets.QHBoxLayout()

        # Top left and right widget panels
        top_left_panel = QtWidgets.QGroupBox('Navigation and Annotation')
        top_right_panel = QtWidgets.QGroupBox('Detection and Model')

        # Grid layouts for the top and left panels.
        self.top_left_grid = QtWidgets.QGridLayout()
        self.top_right_grid = QtWidgets.QGridLayout()
        self.top_left_grid.setSpacing(2)
        self.top_right_grid.setSpacing(2)

        # Widgets for the top panel.
        top_panel.addWidget(top_left_panel)
        top_panel.addWidget(top_right_panel)
        top_panel.addStretch()

        # Set the layout of the panels to be the grids.
        top_left_panel.setLayout(self.top_left_grid)
        top_right_panel.setLayout(self.top_right_grid)

        # Sets the current text.
        self.image_num_txt.setText(' Image is: ' + str(par_obj.curr_z + 1))
        self.count_txt = QtWidgets.QLabel()

        # Sets up the button which saves the ROI.
        self.save_ROI_btn = QtWidgets.QPushButton('1. Save ROI')

        self.save_ROI_btn.setToolTip(
            'Right-click or Ctrl-click and drag to make ROI. Then Save ROI (Space)')
        self.save_ROI_btn.setEnabled(True)

        # Sets up the button which saves the ROI.
        self.save_dots_btn = QtWidgets.QPushButton('2. Save Dots')
        self.save_dots_btn.setToolTip(
            'Click within ROI to add annotations. Then Save Dots')
        self.save_dots_btn.setEnabled(False)

        # Button for training model
        self.train_model_btn = QtWidgets.QPushButton('3. Train Model')
        self.train_model_btn.setToolTip(
            'When annotations are added, train the model')
        self.train_model_btn.setEnabled(False)

        # Selects and reactivates an existing ROI.
        self.delete_ROI_btn = QtWidgets.QPushButton('Delete ROI')
        self.delete_ROI_btn.setToolTip(
            'Delete selected ROI')
        self.delete_ROI_btn.setEnabled(False)

        # Selects and reactivates an existing ROI.
        self.sel_ROI_btn = QtWidgets.QPushButton('Edit ROI')
        self.sel_ROI_btn.setToolTip(
            'Click on ROI to select- Select ROI highlighted in yellow')
        self.sel_ROI_btn.setEnabled(True)
        self.select_ROI = False

        # Allows deletion of dots.
        self.remove_dots_btn = QtWidgets.QPushButton('Remove Dots')
        self.remove_dots_btn.setToolTip(
            'Select ROI, then click Remove dots, and choose dots to remove')
        self.remove_dots_btn.setEnabled(False)
        self.remove_dots = False

        # Load in ROI file.
        self.load_gt_btn = QtWidgets.QPushButton('Import ROI/Dots')
        self.load_gt_btn.setToolTip('Load previously saved annotation file')
        self.load_gt_btn.setEnabled(True)
        self.load_gt = False

        # Save in ROI file.
        self.save_gt_btn = QtWidgets.QPushButton('Export ROI/Dots')
        self.save_gt_btn.setToolTip(
            'Save ROIs and dots to annotation file (.quantiROI)')
        self.save_gt_btn.setEnabled(True)
        self.save_gt = False

        # common navigation elements
        self.Btn_fns = btn_fn(self)
        navigation_setup(self, par_obj)

        # Populates the grid with the different widgets.
        self.top_left_grid.addLayout(self.panel_buttons, 0, 0, 1, 3)

        self.top_left_grid.addWidget(self.image_num_txt, 2, 0, 2, 3)
        self.top_left_grid.addWidget(self.save_ROI_btn, 4, 0)
        self.top_left_grid.addWidget(self.save_dots_btn, 4, 1)
        self.top_left_grid.addWidget(self.train_model_btn, 4, 2)
        self.top_left_grid.addWidget(self.sel_ROI_btn, 5, 0)

        self.top_left_grid.addWidget(self.remove_dots_btn, 5, 1)
        self.top_left_grid.addWidget(self.load_gt_btn, 6, 2, 1, 1)
        self.top_left_grid.addWidget(self.save_gt_btn, 6, 1, 1, 1)
        self.top_left_grid.addWidget(self.delete_ROI_btn, 6, 0, 1, 1 )
        # SigmaData input Label.
        self.sigma_data_text = QtWidgets.QLabel(self)
        self.sigma_data_text.setText('Object size (pixels):')
        self.sigma_data_text.setToolTip(
            'Set this smaller than the object size.\n Map on right hand side should show similar size objects to those in your image.\n If your object size is >10 you should resize your images')
        self.top_right_grid.addWidget(self.sigma_data_text, 0, 0)

        # SigmaData input field.
        self.sigma_data_input = QtWidgets.QLineEdit(str(par_obj.sigma_data))
        self.sigma_data_input.onChanged = self.sigmaOnChange
        self.sigma_data_input.setFixedWidth(40)
        self.sigma_data_input.textChanged[str].connect(self.sigmaOnChange)
        self.top_right_grid.addWidget(self.sigma_data_input, 0, 1)

        # Feature scale input Label.
        #self.sigma_data_text = QtWidgets.QLabel()
        #self.sigma_data_text.setText('Scale of Feature Descriptor:')
        #self.top_right_grid.addWidget(self.sigma_data_text, 1, 0)

        # Feature scale input field
        #self.feature_scale_input = QtWidgets.QLineEdit(str(par_obj.feature_scale))
        #self.feature_scale_input.onChanged = self.feature_scale_change
        #self.feature_scale_input.resize(40, 20)
        # self.feature_scale_input.textChanged[str].connect(self.feature_scale_change)
        # self.feature_scale_input.setFixedWidth(40)
        #self.top_right_grid.addWidget(self.feature_scale_input, 1, 1)

        # Feature scale input btn.
        #self.feat_scale_change_btn = QtWidgets.QPushButton('Recalculate Features')
        # self.feat_scale_change_btn.setEnabled(True)
        #self.top_right_grid.addWidget(self.feat_scale_change_btn, 1, 2)

        # Saves the model
        self.save_model_btn = QtWidgets.QPushButton('Save Training Model')
        self.save_model_btn.setToolTip(
            'When happy with your training, save the model for batch processing')
        self.save_model_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.save_model_btn, 1, 0)

        # Saves the extremel random decision tree model
        self.save_model_name_txt = QtWidgets.QLineEdit('Insert Model Name')
        self.top_right_grid.addWidget(self.save_model_name_txt, 1, 1)

        self.output_count_txt = QtWidgets.QLabel()
        self.top_right_grid.addWidget(self.output_count_txt, 3, 1, 1, 4)

        # Saves the extremely random decision tree model.
        self.save_model_desc_txt = QtWidgets.QLineEdit(
            'Insert Model Description')
        self.save_model_desc_txt.setFixedWidth(200)
        self.top_right_grid.addWidget(self.save_model_desc_txt, 2, 0, 1, 4)
        self.clear_dots_btn = QtWidgets.QPushButton('Clear All ROI')
        self.top_right_grid.addWidget(self.clear_dots_btn, 1, 2)

        # Shows the kernel label distributions
        self.kernel_show_btn = QtWidgets.QPushButton('Showing Kernel')
        self.kernel_show_btn.setMinimumWidth(170)
        self.clear_dots_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.kernel_show_btn, 2, 2)

        self.evaluate_btn = QtWidgets.QPushButton('Evaluate Forest')
        self.evaluate_btn.setToolTip('Evaluate the model on another timepoint')
        self.evaluate_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.evaluate_btn, 3, 0)

        self.count_maxima_btn = QtWidgets.QPushButton('Find object centres')
        self.count_maxima_btn.setToolTip(
            'Use the probability map to estimate cell centres')
        self.count_maxima_btn.setEnabled(False)
        self.top_right_grid.addWidget(self.count_maxima_btn, 4, 0)

        # Button for overlay
        self.overlay_prediction_btn = QtWidgets.QPushButton(
            'Overlay Prediction')
        self.overlay_prediction_btn.setToolTip(
            'Switch between displaying Prediction, Training, and Cell centres')
        self.overlay_prediction_btn.setEnabled(True)
        self.top_right_grid.addWidget(self.overlay_prediction_btn, 0, 2)


        # common navigation buttons

        self.count_replot_btn = QtWidgets.QPushButton('Replot')

        self.count_maxima_plot_on = QtWidgets.QCheckBox()
        self.count_maxima_plot_on.setChecked = False

        self.count_txt_1 = QtWidgets.QLineEdit(str(par_obj.min_distance[0]))
        self.count_txt_1.setFixedWidth(20)
        self.count_txt_2 = QtWidgets.QLineEdit(str(par_obj.min_distance[1]))
        self.count_txt_2.setFixedWidth(20)
        self.count_txt_3 = QtWidgets.QLineEdit(str(par_obj.min_distance[2]))
        self.count_txt_3.setFixedWidth(20)

        abs_thr_lbl = QtWidgets.QLabel('Abs Thr:')
        self.abs_thr_txt = QtWidgets.QLineEdit(str(par_obj.abs_thr))
        self.abs_thr_txt.setFixedWidth(35)
        z_cal_lbl = QtWidgets.QLabel('Z Cal.:')
        self.z_cal_txt = QtWidgets.QLabel(str(par_obj.z_cal))
        self.z_cal_txt.setFixedWidth(50)

        self.strictness_btn = QtWidgets.QCheckBox()
        self.strictness_btn.setChecked(True)
        strictness_btn_lbl = QtWidgets.QLabel('Strict')
        self.strictness_btn.setToolTip('Set whether to enforce size strictness')


        self.min_distance_panel = QtWidgets.QHBoxLayout()
        self.min_distance_panel.addStretch()
        self.min_distance_panel.addWidget(QtWidgets.QLabel("x:"))
        self.min_distance_panel.addWidget(self.count_txt_1)
        self.min_distance_panel.addWidget(QtWidgets.QLabel("y:"))
        self.min_distance_panel.addWidget(self.count_txt_2)
        self.min_distance_panel.addWidget(QtWidgets.QLabel("z:"))
        self.min_distance_panel.addWidget(self.count_txt_3)
        self.min_distance_panel.addWidget(abs_thr_lbl)
        self.min_distance_panel.addWidget(self.abs_thr_txt)
        self.min_distance_panel.addWidget(z_cal_lbl)
        self.min_distance_panel.addWidget(self.z_cal_txt)

        self.min_distance_panel.addWidget(strictness_btn_lbl)
        self.min_distance_panel.addWidget(self.strictness_btn)

        self.top_right_grid.addLayout(self.min_distance_panel, 4, 1, 1, 2)
        # self.top_right_grid.addWidget(self.count_maxima_plot_on,4,2)

        self.top_right_grid.setRowStretch(4, 2)

        # Sets up the image panel splitter.
        image_panel = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        image_panel.addWidget(self.canvas1)
        image_panel.addWidget(self.canvas2)

        # Sets up mouse settings on the image
        self.canvas1.mpl_connect('axes_enter_event', self.on_enter)
        self.canvas1.mpl_connect('axes_leave_event', self.on_leave)
        self.bpe = self.canvas1.mpl_connect(
            'button_press_event', self.on_click)
        self.bre = self.canvas1.mpl_connect(
            'button_release_event', self.on_unclick)
        self.ome = self.canvas1.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.l1 = None
        self.l2 = None
        self.l3 = None
        self.l4 = None
        self.okp = self.canvas1.mpl_connect('key_press_event', self.on_key)
        # Splitter which separates the controls at the top and the images below.
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        hbox1 = QtWidgets.QWidget()
        hbox1.setLayout(top_panel)
        splitter.addWidget(hbox1)
        splitter.addWidget(image_panel)
        box.addWidget(splitter)

        # Status bar which is located beneath images.
        self.image_status_text = QtWidgets.QStatusBar()
        box.addWidget(self.toolbar)
        box.addWidget(self.image_status_text)
        self.image_status_text.showMessage(
            'Status: Please Select a Region and Click \'Save ROI\'. ')

        # Connects the buttons.
        self.save_ROI_btn.clicked.connect(self.save_roi_fn)
        self.save_dots_btn.clicked.connect(self.save_dots_fn)

        self.sel_ROI_btn.clicked.connect(self.sel_ROI_btn_fn)
        self.delete_ROI_btn.clicked.connect(self.delete_roi_fn)

        self.remove_dots_btn.clicked.connect(self.remove_dots_btn_fn)
        self.train_model_btn.clicked.connect(self.train_model_btn_fn)
        self.count_maxima_btn.clicked.connect(self.count_maxima_btn_fn)
        self.load_gt_btn.clicked.connect(self.load_gt_fn)
        self.save_gt_btn.clicked.connect(self.save_gt_fn)
        self.overlay_prediction_btn.clicked.connect(
            self.overlay_prediction_btn_fn)
        self.strictness_btn.clicked.connect(self.strictness_btn_fn)
        # self.feat_scale_change_btn.clicked.connect(self.feat_scale_change_btn_fn)
        self.kernel_show_btn.clicked.connect(self.kernel_btn_fn)
        self.clear_dots_btn.clicked.connect(self.clear_dots_fn)
        self.save_model_btn.clicked.connect(self.saveForestFn)
        self.evaluate_btn.clicked.connect(self.evaluate_forest_fn)

        # Initialises the variables for the beginning of the counting.
        par_obj.first_time = True
        par_obj.dots = []
        par_obj.rects = None
        par_obj.var = []
        par_obj.saved_dots = []
        par_obj.saved_ROI = []
        #par_obj.subdivide_ROI = []
        self.m_Cursor = self.makeCursor()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def evaluate_forest_fn(self):
        # Don't want to train for all the images so we select them.
        v2.im_pred_inline_fn_new(par_obj, self, range(
            par_obj.max_z+1), [par_obj.curr_t], [par_obj.curr_file], threaded=True)

        v2.evaluate_forest_new(par_obj, self, False, 0, range(
            par_obj.max_z+1), [par_obj.curr_t], [par_obj.curr_file])
        par_obj.show_pts = 0
        self.kernel_btn_fn()
        print ('evaluating')
        self.image_status_text.showMessage('Evaluation complete')

    def save_gt_v2(self):

        model_name = self.save_model_name_txt.text()

        # funky ordering TZCYX
        for fileno, imfile in par_obj.filehandlers.iteritems():

            rects = [par_obj.saved_ROI[x]
                     for x, y in enumerate(par_obj.saved_ROI) if y[6] == fileno]
            dots = [par_obj.saved_dots[x]
                    for x, y in enumerate(par_obj.saved_ROI) if y[6] == fileno]

            file_to_save = {'dots': dots, 'rect': rects}

            fileName = imfile.path+'/'+imfile.name+'_'+model_name+'.quantiROI'
            pickle.dump(file_to_save, open(fileName, "wb"))

    def save_gt_fn(self):
        file_to_save = {'dots': par_obj.saved_dots, 'rect': par_obj.saved_ROI}
        fileName = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save dots and regions", "~/Documents", ".quantiROI")
        print ('the filename address', fileName)
        if fileName[-10:] == '.quantiROI':
            fileName = fileName[0:-10]
        pickle.dump(file_to_save, open(fileName + ".quantiROI", "wb"))

    def load_gt_fn(self):
        print ('load the gt')
        fileName = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load dots and regions", "~/Documents", "QuantiFly ROI files (*.quantiROI) ;;MTrackJ Data Format (*.mdf)")
        file_ext = os.path.splitext(fileName)[1]
        print ('load the file', fileName)
        if file_ext == '.quantiROI':
            with open(fileName, "rb") as open_file:
                the_file = pickle.load(open_file)
            par_obj.saved_dots = the_file['dots']
            par_obj.saved_ROI = the_file['rect']
            self.clear_dots_btn.setEnabled(True)

        elif file_ext == '.mdf':
            lines_list = list(open(fileName, 'rb').read().split('\n'))
            par_obj.saved_ROI = []
            par_obj.saved_dots = []
            for i in lines_list:
                i = i.split(' ')
                if len(i) > 0:
                    if i[0] == 'Point':
                        # trim point and convert to int
                        i = [int(float(num)) for num in i[1:-1]]
                        # order ID x y t z c from mdf
                        # order in save_dots: txyz
                        # mdf is 1 indexed
                        rects = (i[4]-1, 0, 0, int(abs(par_obj.ori_width)),
                                 abs(par_obj.ori_height), i[3]-1, par_obj.curr_file)
                        # append if dots already in roi, otherwise create roi and append
                        if rects in par_obj.saved_ROI:
                            idx = par_obj.saved_ROI.index(rects)
                            par_obj.saved_dots[idx].append(
                                (i[4]-1, i[1]-1, i[2]-1))
                        else:
                            par_obj.saved_dots.append(
                                [(i[4]-1, i[1]-1, i[2]-1)])
                            par_obj.saved_ROI.append(rects)
        # check if have loaded points and update
        if par_obj.saved_ROI != []:
            v2.refresh_all_density(par_obj)
            self.train_model_btn.setEnabled(True)
        self.goto_img_fn(par_obj.curr_z, par_obj.curr_t)

    def load_gt_mdf(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load dots", "~/Documents", "MTrackJ Data Format (*.mdf)")
        print ('load the file', fileName)
        lines_list = list(open(fileName, 'rb').read().split('\n'))
        for i in lines_list:
            i = i.split(' ')
            if len(i) > 0:
                if i[0] == 'Point':
                    # trim point and convert to int
                    i = [int(float(num)) for num in i[1:-1]]
                    # order ID x y t z c from mdf
                    # order in save_dots: txyz
                    # mdf is 1 indexed
                    rects = (i[4]-1, 0, 0, int(abs(par_obj.ori_width)),
                             abs(par_obj.ori_height), i[3]-1, par_obj.curr_file)
                    # append if dots already in roi, otherwise create roi and append
                    if rects in par_obj.saved_ROI:
                        idx = par_obj.saved_ROI.index(rects)
                        par_obj.saved_dots[idx].append(
                            (i[4]-1, i[1]-1, i[2]-1))
                    else:
                        par_obj.saved_dots.append([(i[4]-1, i[1]-1, i[2]-1)])
                        par_obj.saved_ROI.append(rects)

    def overlay_prediction_btn_fn(self):
        par_obj.overlay = not par_obj.overlay
        self.goto_img_fn(par_obj.curr_z, par_obj.curr_t)

    def strictness_btn_fn(self):
        par_obj.count_maxima_laplace = not par_obj.count_maxima_laplace

    def count_maxima_btn_fn(self):
        t0 = time.time()
        par_obj.max_det = []
        par_obj.min_distance = [float(self.count_txt_1.text()), float(
            self.count_txt_2.text()), float(self.count_txt_3.text())]
        par_obj.abs_thr = float(self.abs_thr_txt.text())/100
        #par_obj.z_cal =float(self.z_cal_txt.text())
        count_maxima(par_obj, par_obj.curr_t,
                        par_obj.curr_file, reset_max=True)
        par_obj.show_pts = 1
        self.kernel_btn_fn()

    def report_progress(self, message):
        self.image_status_text.showMessage('Status: ' + message)
        app.processEvents()

    def loadTrainFn(self):
        # Win_fn()
        channel_wid = QtWidgets.QWidget()
        channel_lay = QtWidgets.QHBoxLayout()
        channel_wid.setLayout(channel_lay)

        win.top_left_grid.addWidget(channel_wid, 1, 0, 1, 3)

        # cleanup if reloading image
        if not hasattr(self, 'ChannelGroup'):
            # define channel brightness controls
            self.ChannelGroup = []

        for itemset in self.ChannelGroup:
            for item in itemset:
                item.hide()
                item.deleteLater()

        ChannelGroup = create_channel_objects(self, par_obj, par_obj.numCH)
        for chbx, clabel, contrast, blabel, brightness in ChannelGroup:
            channel_lay.addWidget(chbx)
            channel_lay.addWidget(clabel)
            channel_lay.addWidget(contrast)
            channel_lay.addWidget(blabel)
            channel_lay.addWidget(brightness)
            #contrast.show()
            chbx.show()
            chbx.setChecked(True)
        self.ChannelGroup = ChannelGroup
        channel_lay.addStretch()

        win_tab.setCurrentWidget(win)
        app.processEvents()
        self.checkChange()

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
        if event.angleDelta().y() > 3:
            self.Btn_fns.next_im(par_obj)
        if event.angleDelta().y() < -3:
            self.Btn_fns.prev_im(par_obj)
        if event.angleDelta().x() > 10:
            self.Btn_fns.next_time(par_obj)
        if event.angleDelta().x() < -10:
            self.Btn_fns.prev_time(par_obj)

    def on_key(self, event):
        if event.key == ' ':
            if par_obj.draw_dots is True:
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

    def on_click(self, event):
        """When the image is clicked"""
        if event.button == 1:  # left
            if par_obj.draw_ROI is True:

                par_obj.mouse_down = True
                self.save_roi_fn()
            else:
                par_obj.mouse_down = True

        elif event.button == 3:  # right
             # When the draw ROI functionality is enabled:
            par_obj.mouse_down = True

            if par_obj.draw_ROI is True:

                self.x1 = event.xdata
                self.y1 = event.ydata
                par_obj.ori_x = event.xdata
                par_obj.ori_y = event.ydata

    def on_resize(self, event):
        """When the toolbar is activated, resize plot 2"""
        self.plt2.set_ylim(self.plt1.get_ylim())
        self.plt2.set_xlim(self.plt1.get_xlim())
        self.canvas2.draw()

    def on_motion(self, event):
        """When the mouse is being dragged"""
        # When the draw ROI functionality is enabled:
        if(par_obj.draw_ROI == True and par_obj.mouse_down == True and event.button == 3):
            # Finds current cursor position
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
                self.l1 = self.plt1.plot([self.x1, event.xdata], [
                                         self.y1, self.y1], '-', color='r')
                self.l2 = self.plt1.plot([event.xdata, event.xdata], [
                                         self.y1, event.ydata], '-', color='r')
                self.l3 = self.plt1.plot([event.xdata, self.x1], [
                                         event.ydata, event.ydata], '-', color='r')
                self.l4 = self.plt1.plot([self.x1, self.x1], [
                                         event.ydata, self.y1], '-', color='r')

            #self.plt1.Line2D([event.xdata, event.xdata], [self.y1, event.ydata], transform=self.plt1.transData,  figure=self.plt1,color='r')
            #self.plt1.Line2D([event.xdata, self.x1], [ event.ydata,  event.ydata], transform=self.plt1.transData,  figure=self.plt1,color='r')
            #self.plt1.Line2D([self.x1, self.x1], [ event.ydata, self.y1], transform=self.plt1.transData,  figure=self.plt1,color='r')

            self.canvas1.draw()

    def on_unclick(self, event):
        """When the mouse is released"""

        par_obj.mouse_down = False
        self.on_resize(None)  # ensures zoom is updated when scaling in figure1
        # If we are in the roi drawing phase
        if(par_obj.draw_ROI == True and event.button == 3):
            t2 = time.time()
            x = event.xdata
            y = event.ydata
            if (x != None and y != None):
                par_obj.rect_w = x - par_obj.ori_x
                par_obj.rect_h = y - par_obj.ori_y

                # Corrects the corrdinates if out of rectangle.
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if x > par_obj.width:
                    x = par_obj.width-1
                if y > par_obj.height:
                    y = par_obj.height-1
            else:
                par_obj.rect_w = par_obj.ori_x_2 - par_obj.ori_x
                par_obj.rect_h = par_obj.ori_y_2 - par_obj.ori_y
            t1 = time.time()
            print (t1-t2)

        # If we are in the dot drawing phase
        elif(par_obj.draw_dots == True and event.button == 1):
            #catch not initialised
            if event.xdata == None or event.ydata == None:
                return
            x = int(np.round(event.xdata, 0))
            y = int(np.round(event.ydata, 0))

            # Are we with an existing box.
            if(x > par_obj.rects[1]-par_obj.roi_tolerance and x < (par_obj.rects[1] + par_obj.rects[3])+par_obj.roi_tolerance and y > par_obj.rects[2]-par_obj.roi_tolerance and y < (par_obj.rects[2] + par_obj.rects[4])+par_obj.roi_tolerance):
                appendDot = True
                # Appends dots to array if in an empty pixel.
                if par_obj.dots.__len__() > 0:
                    for i in range(0, par_obj.dots.__len__()):
                        if (x == par_obj.dots[i][1] and y == par_obj.dots[i][2]):
                            appendDot = False

                if appendDot == True:
                    par_obj.dots.append((par_obj.curr_z, x, y))
                    i = par_obj.dots[-1]
                    self.plt1.autoscale(False)
                    self.plt1.plot([i[1]-5, i[1]+5],
                                   [i[2], i[2]], '-', color='m')
                    self.plt1.plot(
                        [i[1], i[1]], [i[2]-5, i[2]+5], '-', color='m')
                    self.canvas1.draw()
        elif(par_obj.remove_dots == True and event.button == 1):
            #par_obj.pixMap = QtWidgets.QPixmap(q2r.rgb2qimage(par_obj.imgs[par_obj.curr_z]))
            x = event.xdata
            y = event.ydata
            self.draw_saved_dots_and_roi()
            print ('curr_z', par_obj.curr_z)
            print ('timepoint', par_obj.curr_t)
            # Are we with an existing box.
            if(x > par_obj.rects[1]-par_obj.roi_tolerance and x < (par_obj.rects[1] + par_obj.rects[3])+par_obj.roi_tolerance and y > par_obj.rects[2]-par_obj.roi_tolerance and y < (par_obj.rects[2] + par_obj.rects[4])+par_obj.roi_tolerance):
                # Appends dots to array if in an empty pixel.
                if par_obj.dots.__len__() > 0:
                    for i in range(0, par_obj.dots.__len__()):
                        if ((abs(x - par_obj.dots[i][1]) < 3 and abs(y - par_obj.dots[i][2]) < 3)):
                            par_obj.dots.pop(i)
                            par_obj.saved_dots.append(par_obj.dots)
                            par_obj.saved_ROI.append(par_obj.rects)
                            # Reloads the roi so can edited again. It is now at the end of the array.
                            par_obj.dots = par_obj.saved_dots[-1]
                            par_obj.rects = par_obj.saved_ROI[-1]
                            par_obj.saved_dots.pop(-1)
                            par_obj.saved_ROI.pop(-1)
                            break

            for i in range(0, self.plt1.lines.__len__()):
                self.plt1.lines.pop(0)
            self.dots_and_square(par_obj.dots, par_obj.rects, 'y')
            self.canvas1.draw()

        elif(par_obj.select_ROI == True and event.button == 1):
            x = event.xdata
            y = event.ydata
            for b in range(0, par_obj.ROI_index.__len__()):
                dots = par_obj.saved_dots[par_obj.ROI_index[b]]
                rects = par_obj.saved_ROI[par_obj.ROI_index[b]]
                if(x > rects[1] and x < (rects[1] + rects[3]) and y > rects[2] and y < (rects[2] + rects[4])):

                    par_obj.roi_select = b
                    par_obj.dots = par_obj.saved_dots[par_obj.ROI_index[par_obj.roi_select]]
                    par_obj.rects = par_obj.saved_ROI[par_obj.ROI_index[par_obj.roi_select]]
                    par_obj.saved_dots.pop(
                        par_obj.ROI_index[par_obj.roi_select])
                    par_obj.saved_ROI.pop(
                        par_obj.ROI_index[par_obj.roi_select])

                    for i in range(0, self.plt1.lines.__len__()):
                        self.plt1.lines.pop(0)
                    self.draw_saved_dots_and_roi()
                    self.dots_and_square(dots, rects, 'y')
                    self.canvas1.draw()
                    self.sel_ROI_btn.setEnabled(False)
                    self.save_dots_btn.setEnabled(True)
                    self.remove_dots_btn.setEnabled(True)
                    par_obj.select_ROI = False
                    par_obj.draw_ROI = False
                    par_obj.draw_dots = True
        event.xdata = None
        event.ydata = None

    def dots_and_square(self, dots, rects, colour):

        #self.l5 = lines.Line2D([rects[1], rects[1]+rects[3]], [rects[2],rects[2]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.l6 = lines.Line2D([rects[1]+rects[3], rects[1]+rects[3]], [rects[2],rects[2]+rects[4]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.l7 = lines.Line2D([rects[1]+rects[3], rects[1]], [rects[2]+rects[4],rects[2]+rects[4]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        #self.l8 = lines.Line2D([rects[1], rects[1]], [rects[2]+rects[4],rects[2]], transform=self.plt1.transData,  figure=self.plt1,color=colour)
        # self.plt1.lines.extend([self.l5,self.l6,self.l7,self.l8])
        self.plt1.autoscale(False)
        self.plt1.plot([rects[1], rects[1]+rects[3]],
                       [rects[2], rects[2]], '-', color=colour)
        self.plt1.plot([rects[1]+rects[3], rects[1]+rects[3]],
                       [rects[2], rects[2]+rects[4]], '-', color=colour)
        self.plt1.plot([rects[1]+rects[3], rects[1]], [rects[2] +
                        rects[4], rects[2]+rects[4]], '-', color=colour)
        #self.plt1.plot([rects[1]+rects[3], rects[1]], [rects[2]+rects[4],rects[2]+rects[4]], '-',  figure=self.plt1,color=colour)
        self.plt1.plot([rects[1], rects[1]], [
                       rects[2]+rects[4], rects[2]], '-', color=colour)

        # Draws dots in list
        for i in iter(dots):
            self.plt1.plot([i[1]-5, i[1]+5], [i[2], i[2]], '-', color=colour)
            self.plt1.plot([i[1], i[1]], [i[2]-5, i[2]+5], '-', color=colour)

        return

    def makeCursor(self):
        m_LPixmap = QtGui.QPixmap(28, 28)
        bck = QtGui.QColor(168, 34, 3)
        bck.setAlpha(0)
        m_LPixmap.fill(bck)
        qp = QtGui.QPainter(m_LPixmap)
        qp.setPen(QtGui.QColor(0, 255, 0, 200))
        qp.drawLine(14, 0, 14, 28)
        qp.drawLine(0, 14, 28, 14)
        qp.setOpacity(1.0)
        m_Cursor = QtGui.QCursor(m_LPixmap)
        qp.setOpacity(0.0)
        qp.end()
        return m_Cursor

    def on_enter(self, ev):
        # Changes cursor to the special crosshair on entering image pane.
        QtWidgets.QApplication.setOverrideCursor(self.m_Cursor)
        self.canvas1.setFocus()

    def on_leave(self, ev):
        QtWidgets.QApplication.restoreOverrideCursor()

    def save_roi_fn(self):
        # If there is no width or height either no roi is selected or it is too thin.
        success = v2.save_roi_fn(par_obj)
        if success == True:
            print ('Saved ROI')
            win.image_status_text.showMessage(
                'Status: Select instances in region then click \'save Dots\' ')
            par_obj.draw_ROI = False
            par_obj.draw_dots = True
            win.save_ROI_btn.setEnabled(False)
            win.save_dots_btn.setEnabled(True)
            win.remove_dots_btn.setEnabled(True)
            win.sel_ROI_btn.setEnabled(False)
            win.delete_ROI_btn.setEnabled(True)
            par_obj.remove_dots = False

    def deleteDotsFn(self, sel_ROI_btn_fn):
        print('Dot deleted')
        par_obj.saved_dots.append(par_obj.dots)
        par_obj.saved_ROI.append(par_obj.rects)
        par_obj.dots = par_obj.saved_dots[par_obj.ROI_index[par_obj.roi_select]]
        par_obj.rects = par_obj.saved_ROI[par_obj.ROI_index[par_obj.roi_select]]
        par_obj.saved_dots.pop(par_obj.ROI_index[par_obj.roi_select])
        par_obj.saved_ROI.pop(par_obj.ROI_index[par_obj.roi_select])
        # Creates the qpainter object

        # Now we update a density image of the current Image.
        self.update_density_fn()

    def save_dots_fn(self):
        print('Saved Dots')
        win.image_status_text.showMessage(
            'Status: Highlight new ROI or train. ')
        win.train_model_btn.setEnabled(True)
        par_obj.saved_dots.append(par_obj.dots)
        par_obj.saved_ROI.append(par_obj.rects)
        # self.draw_saved_dots_and_roi()
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
        par_obj.rects = None
        par_obj.ori_x = 0
        par_obj.ori_y = 0
        par_obj.rect_w = 0
        par_obj.rect_h = 0

        # Now we update a density image of the current Image.
        self.update_density_fn()
    def delete_roi_fn(self,ev=None,update_display=True):
        win.image_status_text.showMessage(
            'Status: Highlight new ROI or train. ')
        par_obj.rects = None
        par_obj.ori_x = 0
        par_obj.ori_y = 0
        par_obj.rect_w = 0
        par_obj.rect_h = 0
        # self.draw_saved_dots_and_roi()
        self.save_ROI_btn.setEnabled(True)
        self.save_dots_btn.setEnabled(False)
        self.remove_dots_btn.setEnabled(False)
        self.sel_ROI_btn.setEnabled(True)
        self.clear_dots_btn.setEnabled(True)
        win.delete_ROI_btn.setEnabled(False)
        par_obj.draw_ROI = True
        par_obj.draw_dots = False
        par_obj.remove_dots = False
        par_obj.dots_past = par_obj.dots
        par_obj.dots = []
        self.plt1.lines=[]

        # Now we update a density image of the current Image.
        if update_display==True:
            tpt = par_obj.curr_t
            zslice = par_obj.curr_z
            fileno = par_obj.curr_file
            v2.update_com_fn(par_obj, tpt, zslice, fileno)
            self.draw_saved_dots_and_roi()
            self.canvas1.draw()
            '''
            #self.draw_saved_dots_and_roi()
            self.goto_img_fn(keep_roi=True)
            self.canvas1.draw()
            self.canvas2.draw()'''

    def update_density_fn(self):
        # Construct empty array for current image.
        tpt = par_obj.curr_t
        zslice = par_obj.curr_z
        fileno = par_obj.curr_file
        v2.update_com_fn(par_obj, tpt, zslice, fileno)

        self.goto_img_fn(par_obj.curr_z, par_obj.curr_t)

        self.canvas2.draw()

    def draw_saved_dots_and_roi(self,color='w'):

        for i in range(0, par_obj.saved_dots.__len__()):
            if(par_obj.saved_ROI[i][0] == par_obj.curr_z and par_obj.saved_ROI[i][5] == par_obj.curr_t and par_obj.saved_ROI[i][6] == par_obj.curr_file):
                dots = par_obj.saved_dots[i]
                rects = par_obj.saved_ROI[i]
                self.dots_and_square(dots, rects, color)

    def goto_img_fn(self, zslice=None, tpt=None, imno=None, keep_roi=False):
        # update current image/slice/timepoint if changed
        if zslice != None:
            par_obj.curr_z = zslice
        if tpt != None:
            par_obj.curr_t = tpt
        if imno != None:
            par_obj.curr_file = imno

        # reset drawing tools unless just changing channel
        if keep_roi == False:
            self.delete_roi_fn(update_display=False)
            '''
            # reset controls and box drawing
            par_obj.dots = []
            par_obj.rects = None
            par_obj.select_ROI = False
            par_obj.draw_ROI = True
            par_obj.draw_dots = False
            par_obj.remove_dots = False
            self.save_ROI_btn.setEnabled(True)
            self.save_dots_btn.setEnabled(False)
            self.remove_dots_btn.setEnabled(False)
            self.sel_ROI_btn.setEnabled(True)
            par_obj.ROI_index = []
            '''

        # Goto and evaluate image function.
        v2.goto_img_fn_new(par_obj, self, keep_roi=keep_roi)
        # updates Z-cal
        if par_obj.filehandlers[par_obj.curr_file].z_calibration != 0:
            par_obj.z_cal = par_obj.filehandlers[par_obj.curr_file].z_calibration
        self.z_cal_txt.setText(str(par_obj.z_cal)[0:6])





    def sel_ROI_btn_fn(self):
        par_obj.ROI_index = []
        if par_obj.select_ROI == False:
            self.save_ROI_btn.setEnabled(False)
            par_obj.select_ROI = True
            par_obj.draw_ROI = False
            par_obj.draw_dots = False
            par_obj.remove_dots = False
            for i in range(0, par_obj.saved_ROI.__len__()):
                if(par_obj.saved_ROI[i][0] == par_obj.curr_z and par_obj.saved_ROI[i][5] == par_obj.curr_t and par_obj.saved_ROI[i][6] == par_obj.curr_file):
                    par_obj.ROI_index.append(i)
            for b in range(0, par_obj.ROI_index.__len__()):
                dots = par_obj.saved_dots[par_obj.ROI_index[b]]
                rects = par_obj.saved_ROI[par_obj.ROI_index[b]]
                self.dots_and_square(dots, rects, 'y')
            win.delete_ROI_btn.setEnabled(True)
        else:
            self.save_ROI_btn.setEnabled(True)
            par_obj.select_ROI = False
            par_obj.draw_ROI = True
            self.draw_saved_dots_and_roi()


    def remove_dots_btn_fn(self):
        if par_obj.remove_dots == False:
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
                par_obj.data_store['dense_arr'][fileno][tpt] = {}
        # par_obj.data_store['dense_arr'][imno].clear()
        self.goto_img_fn(par_obj.curr_z)
        self.update_density_fn()
        self.train_model_btn.setEnabled(False)
        self.clear_dots_btn.setEnabled(False)

    def train_model_btn_fn(self):
        self.image_status_text.showMessage(
            'Training Ensemble of Decision Trees. ')
        # added to make sure current timepoint has all features precalculated
        v2.im_pred_inline_fn_new(par_obj, self, range(
            par_obj.max_z+1), [par_obj.curr_t], [par_obj.curr_file], True)

        for i in range(0, par_obj.saved_ROI.__len__()):

            zslice = par_obj.saved_ROI[i][0]
            tpt = par_obj.saved_ROI[i][5]
            imno = par_obj.saved_ROI[i][6]
            print ('calculating features, time point', tpt+1, ' image slice ', zslice+1)
            v2.im_pred_inline_fn_new(par_obj, self, [zslice], [
                                     tpt], [imno], threaded=False)

        par_obj.f_matrix = []
        par_obj.o_patches = []
        t0 = time.time()
        print
        for i in par_obj.saved_ROI:
            #zslice = par_obj.saved_ROI[i][0]
            #tpt =par_obj.saved_ROI[i][5]
            #imno =par_obj.saved_ROI[i][6]
            v2.update_training_samples_fn_new_only(par_obj, self, i)
        print (time.time()-t0)
        t0 = time.time()
        self.image_status_text.showMessage('Training Model')
        v2.train_forest(par_obj, self, 0)
        self.image_status_text.showMessage(
            'Evaluating Images with the Trained Model. ')
        app.processEvents()
        v2.evaluate_forest_new(par_obj, self, False, 0, range(
            par_obj.max_z+1), [par_obj.curr_t], [par_obj.curr_file])
        #v2.make_correction(par_obj, 0)
        self.image_status_text.showMessage(
            'Model Trained. Continue adding samples, or click \'Save Training Model\'. ')
        par_obj.eval_load_im_win_eval = True
        par_obj.show_pts = 0
        self.kernel_btn_fn()
        self.save_model_btn.setEnabled(True)
        self.count_maxima_btn.setEnabled(True)
        self.evaluate_btn.setEnabled(True)
        if par_obj.double_train == True:
            self.double_train_model_btn_fn()

    def double_train_model_btn_fn(self):
        self.image_status_text.showMessage(
            'Training Ensemble of Decision Trees. ')
        # added to make sure current timepoint has all features precalculated
        for i in range(0, par_obj.saved_ROI.__len__()):
            zslice = par_obj.saved_ROI[i][0]
            tpt = par_obj.saved_ROI[i][5]
            imno = par_obj.saved_ROI[i][6]
            v2.evaluate_forest_new(par_obj, self, False, 0, [
                                   zslice], [tpt], [imno])

        v2.im_pred_inline_fn_new(par_obj, self, range(
            par_obj.max_z+1), [par_obj.curr_t], [par_obj.curr_file], 'auto')

        for i in range(0, par_obj.saved_ROI.__len__()):
            zslice = par_obj.saved_ROI[i][0]
            tpt = par_obj.saved_ROI[i][5]
            imno = par_obj.saved_ROI[i][6]
            print ('calculating features, time point', tpt+1, ' image slice ', zslice+1)
            v2.im_pred_inline_fn_new(par_obj, self, [zslice], [
                                     tpt], [imno], threaded='auto')

        par_obj.f_matrix = []
        par_obj.o_patches = []
        t0 = time.time()

        for i in par_obj.saved_ROI:
            v2.update_training_samples_fn_new_only(
                par_obj, self, i, 'double_feat_arr')
        print (time.time()-t0)
        t0 = time.time()
        self.image_status_text.showMessage('Training Model')
        v2.train_forest(par_obj, self, 1)
        self.image_status_text.showMessage(
            'Evaluating Images with the Trained Model. ')
        app.processEvents()
        v2.evaluate_forest_auto(par_obj, self, False, 1, range(
            par_obj.max_z+1), [par_obj.curr_t], [par_obj.curr_file])
        #v2.make_correction(par_obj, 0)
        self.image_status_text.showMessage(
            'Model Trained. Continue adding samples, or click \'Save Training Model\'. ')
        par_obj.eval_load_im_win_eval = True
        par_obj.show_pts = 0
        self.kernel_btn_fn()
        self.save_model_btn.setEnabled(True)
        self.count_maxima_btn.setEnabled(True)
        self.evaluate_btn.setEnabled(True)

    def sigmaOnChange(self, text):
        if text != "":
            par_obj.sigma_data = float(text)
            par_obj.gaussian_im_max = []
            v2.refresh_all_density(par_obj)
            par_obj.min_distance[0] = int(round(par_obj.sigma_data))
            par_obj.min_distance[1] = int(round(par_obj.sigma_data))
            par_obj.min_distance[2] = int(round(par_obj.sigma_data))
            self.count_txt_1.setText(str(par_obj.min_distance[0]))
            self.count_txt_2.setText(str(par_obj.min_distance[1]))
            self.count_txt_3.setText(str(par_obj.min_distance[2]))

            self.update_density_fn()
            if par_obj.sigma_data>10:
                self.image_status_text.showMessage("Feature Calculation with large sigma (>10 pix) is inefficient, consider resizing your images before import")
                self.image_status_text.setStyleSheet("QStatusBar{color: red; font-weight: bold}")
                #time.sleep(.01)

                self.timer = QtCore.QTimer()
                self.timer.isSingleShot = True
                self.timer.timeout.connect(lambda: self.image_status_text.setStyleSheet("QStatusBar{color: black, font-weight: normal}"))
                self.timer.start(2000)

                #self.image_status_text.setStyleSheet("QStatusBar{color: black, font: normal}")
    def kernel_btn_fn(self, setting=False):
        """Shows the kernels on the image."""
        if setting == 'Kernel':
            par_obj.show_pts = 0
        elif setting == 'Probability':
            par_obj.show_pts = 1
        elif setting == 'Counts':
            par_obj.show_pts = 2
        elif setting == False:
            par_obj.show_pts = (par_obj.show_pts + 1) % 3

        print ('show', par_obj.show_pts)

        if par_obj.show_pts == 0:
            self.kernel_show_btn.setText('Showing Kernel')
            self.update_density_fn()
        elif par_obj.show_pts == 1:
            self.kernel_show_btn.setText('Showing Probability')
            v2.goto_img_fn_new(par_obj, self)
        elif par_obj.show_pts == 2:
            self.kernel_show_btn.setText('Showing Counts')
            v2.goto_img_fn_new(par_obj, self)

    def saveForestFn(self):

        path = os.path.expanduser('~')+'/.densitycount/models/'
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        local_time = time.asctime(time.localtime(time.time()))
        par_obj.modelName = str(self.save_model_name_txt.text())
        par_obj.modelDescription = str(self.save_model_desc_txt.text())

        cleanString = re.sub(r'\W+', '', par_obj.modelName)

        basename = path + "pv20"
        suffix = str(int(round(time.time(), 0)))
        filename = "_".join([basename, suffix, str(cleanString), ".mdla"])
        save_file = {}

        # Formats image to make a better icon.
        if par_obj.save_im.shape[0] > 300 and par_obj.save_im.shape[1] > 300:
            save_im = np.zeros((300, 300, 3))
            cent_y = np.floor(par_obj.save_im.shape[0]/2).astype(np.int32)
            cent_x = np.floor(par_obj.save_im.shape[1]/2).astype(np.int32)
            if par_obj.save_im.shape[2] > 2:
                save_im[:, :, 0] = par_obj.save_im[cent_y -
                                                   150:cent_y+150, cent_x-150:cent_x+150, 0]
                save_im[:, :, 1] = par_obj.save_im[cent_y -
                                                   150:cent_y+150, cent_x-150:cent_x+150, 1]
                save_im[:, :, 2] = par_obj.save_im[cent_y -
                                                   150:cent_y+150, cent_x-150:cent_x+150, 2]
            else:
                save_im[:, :, 0] = par_obj.save_im[cent_y -
                                                   150:cent_y+150, cent_x-150:cent_x+150, 0]
                save_im[:, :, 1] = par_obj.save_im[cent_y -
                                                   150:cent_y+150, cent_x-150:cent_x+150, 0]
                save_im[:, :, 2] = par_obj.save_im[cent_y -
                                                   150:cent_y+150, cent_x-150:cent_x+150, 0]
        else:
            save_im = np.zeros(
                (par_obj.save_im.shape[0], par_obj.save_im.shape[1], 3))
            if par_obj.save_im.shape[2] > 2:
                save_im[:, :, 0] = par_obj.save_im[:, :, 0]
                save_im[:, :, 1] = par_obj.save_im[:, :, 1]
                save_im[:, :, 2] = par_obj.save_im[:, :, 2]
            else:
                save_im[:, :, 0] = par_obj.save_im[:, :, 0]
                save_im[:, :, 1] = par_obj.save_im[:, :, 0]
                save_im[:, :, 2] = par_obj.save_im[:, :, 0]

        par_obj.file_ext = ''  # added for backwards compatibility when saving

        save_file = {"name": par_obj.modelName, 'description': par_obj.modelDescription, "c": par_obj.c, "M": par_obj.M,
                     "sigma_data": par_obj.sigma_data, "model": par_obj.RF, "date": local_time, "feature_type": par_obj.feature_type,
                     "feature_scale": par_obj.feature_scale, "ch_active": par_obj.ch_active, "limit_ratio_size": par_obj.limit_ratio_size,
                     "max_depth": par_obj.max_depth, "min_samples": par_obj.min_samples_split, "min_samples_leaf": par_obj.min_samples_leaf,
                     "max_features": par_obj.max_features, "num_of_tree": par_obj.num_of_tree, "file_ext": par_obj.file_ext, "imFile": save_im,
                     "resize_factor": par_obj.resize_factor, "min_distance": par_obj.min_distance, "abs_thr": par_obj.abs_thr,
                     "rel_thr": par_obj.rel_thr, "max_det": par_obj.max_det, "count_maxima_laplace": par_obj.count_maxima_laplace}

        pickle.dump(save_file, open(filename, "wb"))
        self.save_model_btn.setEnabled(False)
        self.report_progress('Model Saved.')

    def checkChange(self):

        #v2.eval_goto_img_fn(par_obj, self,par_obj.curr_z,par_obj.curr_t)
        v2.load_and_initiate_plots(par_obj, self)
        # makes sure Z_calibration is set
        self.sigmaOnChange(par_obj.sigma_data)


# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    #freeze_support()
    # generate layout
    app = QtWidgets.QApplication([])
    QtWidgets.QApplication.setQuitOnLastWindowClosed(True)
    # Create and display the splash screen
    splash_pix = QtGui.QPixmap('splash_loading.png')
    splash = QtWidgets.QSplashScreen(
        splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    #app.processEvents()
    #timer = QtCore.QTimer()
    #timer.timeout.connect(lambda: time.sleep(0.001))
    #timer.start(100)
    # Creates tab widget.
    win_tab = QtWidgets.QTabWidget()
    # Creates win, an instance of QWidget
    par_obj = ParameterClass()
    win = Win_fn(par_obj)
    loadWin = Load_win_fn()  # par_obj,win)

    # Adds win tab and places button in par_obj.
    win_tab.addTab(loadWin, "Load Images")
    win_tab.addTab(win, "Train Model")

    # Defines size of the widget.
    win_tab.resize(1000, 600)

    time.sleep(0.2)
    splash.finish(win_tab)
    win_tab.showMaximized()
    win_tab.activateWindow()

    sys.exit(app.exec_())
    '''
    # Automates the loading for testing.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
    '''
