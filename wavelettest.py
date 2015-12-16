# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 02:11:32 2015

@author: martin
"""

import numpy as np
import pywt
from PIL import Image
import pdb
def w2d(img, mode='sym2', level=1):
    imArray = Image.open('/Users/martin/Documents/output014.tif')
    #Datatype conversions
    #convert to grayscale
    #convert to float
    imArray =  np.float32(imArray)   *800
    imArray2 = imArray
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray2, mode, level=level)
    print pywt.wavelist()
    #Process Coefficient
    coeffs_H=list(coeffs)  
    '''for it in range(1,level+1):
        (a,b,c)=coeffs_H[it]
        Image.fromarray((np.float32(a))).show()
        Image.fromarray((np.float32(b))).show()
        Image.fromarray((np.float32(c))).show()'''
    coeffs_H[0] *= 0;  
    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H[0:level+1], mode);
    print imArray_H.max()
    #imArray_H *= 255;
    imArray_H =  np.float32(imArray_H)
    #Display result
    print imArray_H.max()

    im=Image.fromarray(imArray_H)
    im.show()
    pdb.set_trace()
if __name__ == '__main__':
    w2d("test1.png",'sym2',2)