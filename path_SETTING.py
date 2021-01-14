# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:04:07 2020

@author: 86136
"""

import os
import sys

def path_dir(path):

    print('\n============ Function in progress ==============')
    print(sys._getframe().f_code.co_name)
    print('================================================\n')
    
    try:
        os.chdir(path[0])
        return path[0]
    except:
        os.chdir(path[1])    
        return path[1]
