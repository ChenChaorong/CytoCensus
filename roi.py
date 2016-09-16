from PyQt4 import QtGui, QtCore,QtWebKit
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import copy
import sys
sys.path.append('final_software')
import v2_functions as v2
import numpy as np

  




class Win_fn(QtGui.QWidget):
    """Class which houses main training functionality"""
    def __init__(self,par_obj):
        super(Win_fn, self).__init__()
        self.var = 1
        self.show_image()
        self.show()
        self.par_obj = par_obj


        
    def show_image(self):

       
        
        self.figure1 = Figure(figsize=(8, 8), dpi=100)
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure1.patch.set_facecolor('grey')
        
        self.plt1 = self.figure1.add_subplot(1, 1, 1)
        self.plt1.set_title("Title")
        #self.plt1.set_xlim([0,100])
        #self.plt1.set_ylim([0,100])
        
        self.cursor = v2.ROI(self,par_obj)
        
        
        im_RGB = np.zeros((512, 512))
        #Makes sure it spans the whole figure.
        #self.figure1.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)
        
        self.plt1.imshow(par_obj.ex_img)
        self.plt1.set_ylim(0,par_obj.ex_img.shape[0])
        self.plt1.set_xlim(0,par_obj.ex_img.shape[1])
        self.canvas1.show()
        
        self.top_panel = QtGui.QHBoxLayout()
        self.compl_btn = QtGui.QPushButton('complete')
        self.compl_btn.clicked.connect(self.cursor.complete_roi)
        self.next_btn = QtGui.QPushButton('next')
        self.next_btn.clicked.connect(self.next_im_btn_fn)
        self.prev_btn = QtGui.QPushButton('prev')
        self.prev_btn.clicked.connect(self.prev_im_btn_fn)
        self.interpolate_btn = QtGui.QPushButton('interpolate')

        self.interpolate_btn.clicked.connect(self.cursor.interpolate_ROI)
        self.top_panel.addWidget(self.compl_btn)
        self.top_panel.addWidget(self.prev_btn)
        self.top_panel.addWidget(self.next_btn)
        self.top_panel.addWidget(self.interpolate_btn)
        self.top_panel.addStretch()
        
        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addLayout(self.top_panel)
        vbox.addWidget(self.canvas1)



        
        
        

        self.canvas1.mpl_connect('motion_notify_event', self.cursor.motion_notify_callback)
        self.canvas1.mpl_connect('button_press_event', self.cursor.button_press_callback)
        self.canvas1.mpl_connect('button_release_event', self.cursor.button_release_callback)
        #self.canvas1.show()
    def prev_im_btn_fn(self):
        im_num = par_obj.curr_img - 1
        if im_num >-1:
            par_obj.prev_img = par_obj.curr_img
            par_obj.curr_img = im_num
            self.goto_img_fn(im_num,self.par_obj, self)
            
    def next_im_btn_fn(self):
        im_num = par_obj.curr_img + 1
        if im_num <par_obj.test_im_end:
            par_obj.prev_img = par_obj.curr_img
            par_obj.curr_img = im_num
            self.goto_img_fn(im_num, self.par_obj, self)
    def goto_img_fn(self,im_num, par_obj, int_obj):
        """Loads up and converts image to correct format"""

        #Finds the current frame and file.
        count = -1
        for b in par_obj.left_2_calc:
            frames =par_obj.frames_2_load[b]
            for i in frames:
                count = count+1
                
                if par_obj.curr_img == count:
                    break;
            else:
                continue 
            break 
        

        
        imRGB = par_obj.oib_file[0,count,:,:]
        
    #            if event.button == 3:
        self.plt1.cla()
        self.plt1.imshow(imRGB)
        self.plt1.set_ylim(0,imRGB.shape[0])
        self.plt1.set_xlim(0,imRGB.shape[1])
        self.cursor.ppt_x = []
        self.cursor.ppt_y = []
        self.cursor.line = [None]
        self.cursor.draw_ROI()
        self.canvas1.draw()
        
        self.cursor.complete = False
        self.cursor.flag = False
        for bt in self.par_obj.roi_stk_x:
            if bt == self.par_obj.curr_img:
                self.cursor.complete = True
                self.cursor.ppt_x = copy.deepcopy(self.par_obj.roi_stk_x[bt])
                self.cursor.ppt_y = copy.deepcopy(self.par_obj.roi_stk_y[bt])
                
                break;
        
            

        
class parameterClass:
    def __init__(self):
        self.file_array = ['/Users/dwaithe/Documents/collaborators/YangLu/working_examples/Genotype_wt_syp_Staiining_Ase_Dpn/20150206_DAPI_Dpn568_Ase647_60x_4xZoom_20Stacks_WT04.oib']
        self.left_2_calc = [0]
        self.curr_img = 0
        self.roi_stk_x = {}
        self.roi_stk_y = {}
        self.roi_stkint_x ={}
        self.roi_stkint_y ={}
        self.npts = 500
        
    

if __name__ == "__main__":
    app = QtGui.QApplication([])
    app.processEvents()
    par_obj = parameterClass()
    success, updateText = v2.import_data_fn(par_obj, par_obj.file_array)
    
    #par_obj.frames_2_load = [range(0,par_obj.dv_file.maxFrames)]
    par_obj.frames_2_load = [range(0,par_obj.oib_file.shape[1])]


    win = Win_fn(par_obj)
    win_tab = QtGui.QTabWidget()
    win_tab.addTab(win, "Show plot")
    win_tab.resize(1000,600)

    print success
    win_tab.show()
    
      
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_() 