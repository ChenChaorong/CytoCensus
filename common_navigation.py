# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 18:03:50 2016

@author: martin

Common colour checkboxes and 4D navigation functions
 Currently must import directly, need to tidy this up later
"""
from PyQt4 import QtGui
from gnu import return_license
import numpy as np

def navigation_setup(self,par_obj):
    
    #Sets up the button which changes to the prev image
    self.prev_im_btn = QtGui.QPushButton('Z (<)')
    self.prev_im_btn.setEnabled(True)
    
    #Sets up the button which changes to the next Image.
    self.next_im_btn = QtGui.QPushButton('Z (>)')
    self.next_im_btn.setEnabled(True)
    
    #Sets up the button which changes to the prev image
    self.first_im_btn = QtGui.QPushButton('Z (<<)')
    self.first_im_btn.setEnabled(True)
    
    #Sets up the button which changes to the next Image.
    self.last_im_btn = QtGui.QPushButton('Z (>>)')
    self.last_im_btn.setEnabled(True)
    
     #Sets up the button which changes to the prev image
    self.prev_time_btn = QtGui.QPushButton('Time (<)')
    self.prev_time_btn.setEnabled(True)
    
    #Sets up the button which changes to the next Image.
    self.next_time_btn = QtGui.QPushButton('Time (>)')
    self.next_time_btn.setEnabled(True)
    
    #Sets up the button which changes to the next File.
    self.prev_file_btn = QtGui.QPushButton('File (<)')
    self.prev_file_btn.setEnabled(True)
    
    #Sets up the button which changes to the next File.
    self.next_file_btn = QtGui.QPushButton('File (>)')
    self.next_file_btn.setEnabled(True)
    
    self.output_count_txt = QtGui.QLabel()
    
    self.panel_buttons = QtGui.QHBoxLayout()
    self.panel_buttons.addWidget(self.prev_im_btn)
    self.panel_buttons.addWidget(self.next_im_btn)
    self.panel_buttons.addWidget(self.first_im_btn)
    self.panel_buttons.addWidget(self.last_im_btn)
    self.panel_buttons.addWidget(self.prev_time_btn)
    self.panel_buttons.addWidget(self.next_time_btn)
    self.panel_buttons.addWidget(self.prev_file_btn)
    self.panel_buttons.addWidget(self.next_file_btn)

    self.prev_im_btn.clicked.connect(lambda: self.Btn_fns.prev_im(par_obj))
    self.next_im_btn.clicked.connect(lambda: self.Btn_fns.next_im(par_obj))
    self.first_im_btn.clicked.connect(lambda: self.Btn_fns.first_im(par_obj))
    self.last_im_btn.clicked.connect(lambda: self.Btn_fns.last_im(par_obj))
    
    self.prev_time_btn.clicked.connect(lambda: self.Btn_fns.prev_time(par_obj))
    self.next_time_btn.clicked.connect(lambda: self.Btn_fns.next_time(par_obj))
    
    self.prev_file_btn.clicked.connect(lambda: self.Btn_fns.prev_file(par_obj))
    self.next_file_btn.clicked.connect(lambda: self.Btn_fns.next_file(par_obj))
    
class btn_fn:
    def __init__(self,win):
        self.win=win
        self.goto_img_fn=win.goto_img_fn
    def prev_im(self,par_obj):
        if par_obj.curr_z > 0:
            zslice = par_obj.curr_z-1
            self.goto_img_fn(zslice=zslice)
            
    def next_im(self,par_obj):
        if par_obj.curr_z < par_obj.max_z:
            zslice = par_obj.curr_z+1
            self.goto_img_fn(zslice=zslice)
            
    def last_im(self,par_obj):
        zslice = par_obj.max_z
        self.goto_img_fn(zslice=zslice)
                    
    def first_im(self,par_obj):
        zslice = 0
        self.goto_img_fn(zslice=zslice)
                    
    def prev_time(self,par_obj):
        for idx,tpt in enumerate(par_obj.tpt_list):
            if par_obj.curr_t == tpt:
                if idx>0:
                    tpt = par_obj.tpt_list[idx-1]
                    self.goto_img_fn(tpt=tpt)
                    break
    
    def next_time(self,par_obj):
        if par_obj.max_t>par_obj.curr_t:
            for idx,tpt in enumerate(par_obj.tpt_list):
                if par_obj.curr_t == tpt:
                    if idx+1<len(par_obj.tpt_list):
                        tpt = par_obj.tpt_list[idx+1]
                        self.goto_img_fn(tpt=tpt)
                        break
                    
    def prev_file(self,par_obj):
        if par_obj.curr_file > 0:
            par_obj.max_z = min(par_obj.filehandlers[par_obj.curr_file-1].max_z,par_obj.user_max_z)
            par_obj.max_t = min(par_obj.filehandlers[par_obj.curr_file-1].max_t,par_obj.tpt_list[-1])
            if par_obj.curr_z>par_obj.max_z: par_obj.curr_z = par_obj.max_z
            if par_obj.curr_t>par_obj.max_t: par_obj.curr_t = par_obj.max_t
            self.goto_img_fn(imno=par_obj.curr_file-1)
            #limit z by user and file extent
    
    
    def next_file(self,par_obj):
        for fileno in range(par_obj.max_file):
            if fileno == par_obj.curr_file:
                if fileno < par_obj.max_file-1:
                    par_obj.max_z = par_obj.filehandlers[par_obj.curr_file+1].max_z
                    par_obj.max_t = par_obj.filehandlers[par_obj.curr_file+1].max_t
                    if par_obj.curr_z>par_obj.max_z: par_obj.curr_z = par_obj.max_z
                    if par_obj.curr_t>par_obj.max_t: par_obj.curr_t = par_obj.max_t
                    self.goto_img_fn(imno=par_obj.curr_file+1)
    
                    break;

def on_about(self):
    self.about_win = QtGui.QWidget()
    self.about_win.setWindowTitle('About QBrain Software v2.0')

    license = return_license()
    #with open (sys.path[0]+'/GNU GENERAL PUBLIC LICENSE.txt', "r") as myfile:
    #    data=myfile.read().replace('\n', ' ')
    #    license.append(data)

    # And give it a layout
    layout = QtGui.QVBoxLayout()

    self.view = QtWebKit.QWebView()
    self.view.setHtml('''
      <html>


        <body>
          <form>
            <h1 >About</h1>

            <p>Software written by Dominic Waithe and Martin Hailstone (c) 2015-2016</p>
            '''+str(license)+'''


          </form>
        </body>
      </html>
    ''')
    layout.addWidget(self.view)
    self.about_win.setLayout(layout)
    self.about_win.show()
    self.about_win.raise_()
class checkBoxCH(QtGui.QCheckBox):
    def __init__(self,par_obj,Win,ID,feature_select,text=None):
        QtGui.QCheckBox.__init__(self,text)
        self.stateChanged.connect(self.stateChange)
        self.feature_select = feature_select
        self.ID=ID
        self.Win=Win
        self.par_obj=par_obj
    def stateChange(self):
        par_obj=self.par_obj
        Win=self.Win
        # set channels to use for feature calculation
        if self.feature_select==True:
            if self.isChecked()==True:
                if self.ID not in par_obj.ch_active:
                    par_obj.ch_active.append(self.ID)
                    par_obj.ch_active.sort()
            else:
                if self.ID in par_obj.ch_active:
                    del par_obj.ch_active[par_obj.ch_active.index(self.ID)]
                    
            if par_obj.ex_img is not None:
                newImg = np.zeros((par_obj.ex_img.shape[0], par_obj.ex_img.shape[1], 3))
                if par_obj.ch_active.__len__() > 1:
                    for b in par_obj.ch_active:
                        if b==2:break
                        newImg[:, :, b] = par_obj.ex_img[:, :, b]
    
                elif par_obj.ch_active.__len__() ==1:
                    newImg = par_obj.ex_img[:, :, par_obj.ch_active[0]]
                Win.plt1.images[0].set_data(newImg/par_obj.tiffarraymax)
                Win.canvas1.draw()

        else: #set visible channels
            #if self.ID not in par_obj.ch_display:
            if self.isChecked():
                if self.ID not in par_obj.ch_display:
                    par_obj.ch_display.append(self.ID)
                    par_obj.ch_display.sort()
            else:#removes channel when unticked
                if self.ID in par_obj.ch_display:
                    del par_obj.ch_display[par_obj.ch_display.index(self.ID)]
            Win.goto_img_fn(par_obj.curr_z,par_obj.curr_t)
def create_channel_objects(self,par_obj,num,feature_select=False,parent=None):
    #Object factory for channel selection.
    parent=[]
    self.CH_cbx = []
    for i in range(0,num):
        cbx=checkBoxCH(par_obj,self,i,feature_select,'CH '+str(i+1)+':')
        parent.append(cbx)
        self.CH_cbx.append(cbx)
        self.CH_cbx[i].setChecked(True)
        #self.CH_cbx[i].hide()
        #self.CH_cbx[i].update()
    return parent