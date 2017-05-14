# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:06:33 2016

@author: martin

Common parameterObject from Evaluate and Release
 Includes defaults
 May still be missing one or two parameters
"""
import os
import errno
import shelve
import UserDict
class parameterClass:
    def __init__(self):
        #debugging
        self.FORCE_nothreading=False # confusingly, set to 0 for debugging, fix later
        #window related- should these really be in this object??
        self.evalLoadImWin_loaded = False
        self.evalLoadModelWin_loaded = False
        self.evalLoadImWin_loaded = False
        self.evalDispImWin_evaluated = False
        self.eval_load_im_win_eval = False
        self.show_pts= 0 #switch for display of prediction- #TODO better as toggle button state
        self.overlay=False #overlay prediction -better as tickbox?
        
        self.mouse_down = False #ROI drawing
        
        #hard maxima to prevent opening overly huge images
        self.x_limit = 5024
        self.y_limit = 5024
        
        #crop numbers: historical? remove in future?
        self.to_crop = False
        self.crop_x1 = 0
        self.crop_x2 = 0
        self.crop_y1 = 0
        self.crop_y2 = 0
        #historical-remove in future
        self.left2calc = 0
        self.fresh_features = True
        self.p_size = 1
        self.prev_img=[]
        self.oldImg=[]
        self.newImg=[]
        self.auto = True        #Auto mode. #what is this??
        self.kernel_toggle = False
        self.c = 0
        self.M = 1
        

        #store high level file data and metadata
        self.file_name={}
        self.file_array =[]
        self.tiffarray=[] #memmap object list
        self.z_calibration={}
        self.z_calibration[0]=1
        self.order={} #ordering of tiff objects
        #default file extents
        self.max_file=0
        self.total_time_pt = 1
        self.max_zslices =0

        self.curr_file=0
        self.curr_z = 0
        self.time_pt = 0 #TODO make these names consistent
        #use these?
        self.t={}
        self.z={}
        self.f={}
        
        self.height = 0
        self.width = 0
        self.resize_factor = 1
        
        #hessian defaults
        self.min_distance = [2,2,2]
        self.min_distance_old=[]
        self.abs_thr = 0.05
        self.rel_thr = 0
        self.max_det=[]
        #ROI point
        self.npts = 100

        #store for processed data
        self.data_store ={}
        
        #save path #TODO check different operating systems
        self.forPath = os.path.expanduser('~')+'/.densitycount/models/'

        try:
            os.makedirs(self.forPath)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        self.csvPath = os.path.expanduser('~')+'/'
        
        '''Parameters specific to training'''
        #Parameters of sampling
        self.limit_sample = True
        self.limit_ratio = True #whether to use ratio of roi pixels
        self.limit_ratio_size =21/4 #Gives 3000 patches for 255*255 image.
        self.limit_size = 3000 #patches per image or ROI. # overridden by limit ratio
        self.roi_tolerance = 10
        self.double_train = False
        #Random Forest parameters
        self.RF ={} #possibility to apply multiple models efficiently. Not implemented
        self.max_depth=10
        self.min_samples_split=20
        self.min_samples_leaf=10
        self.max_features = 14#7
        self.num_of_tree = 75#50#30
        self.feature_type=[]
        self.num_of_feat=[0,0]
        # default parameters
        self.sigma_data = 2.0
        self.feature_scale = 1.2
        #TODO check how necessary these are now that gaussians are scaled
        self.maxPred=0
        self.minPred=100
        self.gaussian_im_max=[]
        # file specific, but the same for all models
        self.numCH =0
        self.ch_active=[]
        self.ch_display=[]
        #subset for training
        self.frames_2_load =[]
        self.time_pt_list=[]
        #preview image
        self.ex_img=None
        #ROI parameters
        self.draw_ROI = True #TODO one of these two should be true-consider making this one parameter
        self.select_ROI= False
        
        self.remove_dots = False
        self.draw_dots =False
#        #initiate datastructures
#    for dataname in ['dense_arr','feat_arr','pred_arr','sum_pred','maxi_arr','pts','roi_stk_x','roi_stk_y','roi_stkint_x','roi_stkint_y']:
#        par_obj.data_store[dataname]={}
#        for fileno in range(par_obj.max_file):
#            par_obj.data_store[dataname][fileno]={}
#            par_obj.data_store[dataname][fileno][0]={}
    def initiate_data_store(self,datasets=None):
        if datasets==None:
            datasets=['dense_arr','feat_arr','double_feat_arr','pred_arr','sum_pred','maxi_arr','pts','roi_stk_x','roi_stk_y','roi_stkint_x','roi_stkint_y']
        #initiate datastructures
        
        #self.data_store=shelve.open('datastore.temp','n',writeback=True)
        #self.data_store.clear()
        current_sets=self.data_store.keys()
        for dataname in datasets:
            if dataname in current_sets:
                del self.data_store[dataname]
            self.data_store[dataname]={}
            
            for fileno in range(self.max_file):
                self.data_store[dataname][fileno]={}
                for time_pt in self.time_pt_list:
                    self.data_store[dataname][fileno][time_pt]={}
                    
    class serialdict(shelve.Shelf):
        def __init__(self, *args):
            shelve.Shelf.__init__(self, args)
        def __getitem__(self, key):
            val = shelve.Shelf.__getitem__(self, str(key))
            return val
        def __setitem__(self, key, val):
            shelve.Shelf.__setitem__(self, str(key), val)
            
    def reset_parameters(self): #TODO a better way to integrate this with init
        self.frames_2_load ={}
        self.left_2_calc =[]
        self.saved_ROI =[]
        self.saved_dots=[]