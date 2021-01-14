#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:47:02 2020

@author: tang
"""
print('loading... please wait')
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore','(?s).*MATPLOTLIBDATA.*',category = UserWarning)
import pandas as pd
import numpy as np
import os
from scipy.stats import t
from scipy.stats import f
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import operator
import scipy.stats as stats
import sys
import time
import random
#import math
import csv
import xlrd
import xlwt
from xlutils.copy import copy
global xls_file

from scipy.stats import chi2

from path_SETTING import path_dir    
from Distributions import distribution
#from matrix_array import value_matrix

def write_field_xls(path, sheet_name, value):
    # path：工作簿的路径，sheet_name：第一个sheet的名称，value二维数组，表示插入excel的数据
    # 第一次建立工作簿时候调用
    if type(value) == list:
        value = np.array(value)
    if len(value.shape) == 1:
        out_value = np.zeros([value.shape[0],1])
        out_value[:,0] = value
        value = out_value
    if len(value.shape) == 3:
        out_value = value[:,:,0]
        value = out_value        
    index = len(value)  # 获取需要写入数据的行数
    # workbook = xlwt.Workbook()  # 新建一个工作簿
    workbook = xls_file
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i,j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    print("xls格式表格写入数据成功！")


def write_sheet_xls(path, sheet_name, value):
    # 新建sheet的时候进行调用
    if type(value) == list:
        value = np.array(value)
    if len(value.shape) == 1:
        out_value = np.zeros([value.shape[0],1])
        out_value[:,0] = value
        value = out_value  
    if len(value.shape) == 3:
        out_value = value[:,:,0]
        value = out_value            
    index = len(value)  # 获取需要写入数据的行数
    # workbook = xlwt.Workbook()  # 新建一个工作簿
    rb = xlrd.open_workbook(path, formatting_info=True)
    workbook = copy(rb)
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i,j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    print("xls格式表格写入数据成功！")
    

def raw_input(word,types):
    sig = 1
    while sig:
        sig = 0
        try:
            if types == 'int':
                ss = int(input(word))
            else:
                ss = input(word)
        except:
            print('error input')
            sig += 1               
    return ss




def Time():
     
    print('\n=================================================================')
    localtime = time.asctime(time.localtime(time.time()))
    print('local time:  ' + localtime)
    print('\n=================================================================')    
    
    return localtime


def file_read(ff_name,address):
    
    paths = path_dir(address)       
    if ff_name[-3:] == 'dat':
        
        s_path = paths + '/' + ff_name
        data = pd.read_table(s_path,header=None,sep='\s+')
        #skiprows[a,b,c,d] abcd行不读取//// sep=‘\s+’识别切割的字符（空格，或多个空格），默认为 “，”。   
        #lens=len(data)
        out = data.values
    
    if ff_name[-3:] == 'csv':           
        with open(ff_name, 'r') as f:
            reader = csv.reader(f)
            print(type(reader))
            sign = 0
            for row in reader:
                if sign == 0:
                    out = np.zeros(shape = len(row))
                    for i in range(len(row)):
                        out[i] = row[i]
                else:
                    test = np.zeros(shape = len(row))
                    for i in range(len(row)):
                        test[i] = row[i]
                    out = np.row_stack((out,test))        
                sign += 1         
    return out


def random_array(scale,Range,mean_norm):
    
    if mean_norm == 'mean':        
        matrix_random = np.zeros(scale)
        for i in range(matrix_random.shape[0]):
            for j in range(matrix_random.shape[1]):
                matrix_random[i,j] = np.random.random()
                matrix_random[i,j] = Range[0] + (Range[1] - Range[0]) * matrix_random[i,j]
    elif mean_norm == 'norm':
        if type(Range) == list:
            matrix_random = np.random.randn(scale[0],scale[1])
            maxim = np.max(abs(matrix_random))
            for i in range(matrix_random.shape[0]):
                for j in range(matrix_random.shape[1]):
                    matrix_random[i,j] = 0.5*Range[1] + 0.5*Range[0] + (Range[1] - Range[0])/(maxim*2) * matrix_random[i,j]
        else:
            matrix_random = np.random.randn(scale[0],scale[1])      
    return  matrix_random


def remove_abnormal_file(ff):
    
    out_ff = []
    for i in range(len(ff)):
        if '.' in ff[i] and ff[i][0] != '.':
            out_ff.append(ff[i])
            
    return out_ff


class DAT_read():
    def __init__(self,path):
        self.path = path
        
    def file_list(self):
        paths = path_dir(self.path)
        ff = os.listdir(paths)
        ff.sort()
        ff = remove_abnormal_file(ff)   
        print(ff)
        return ff
        
    def read_file(self,ff_name):
        paths = path_dir(self.path)        
        s_path = paths + '/' + ff_name
        data = pd.read_table(s_path,header=None,sep='\s+')
        #skiprows[a,b,c,d] abcd行不读取//// sep=‘\s+’识别切割的字符（空格，或多个空格），默认为 “，”。   
        #lens=len(data)
        a_data = data.values
        
        return a_data
    

def R_diag(C):
    
    if C.shape == () or C.shape == 1:
        return C
    else:
        return np.diag(C)
    
    
def R_matrix(C,nrow,byrow):
    
    if len(C.shape) == 1 and C.any() != 0:
        length = len(C)
        mat = np.zeros(shape = (nrow,int(length/nrow)))
        if byrow != False and byrow != 0:
            num = 0
            for i in range(nrow):
                for j in range(int(length/nrow)):
                    mat[i,j] = C[num]
                    num += 1
        else:
            for i in range(int(length/nrow)):
                mat[:,i] = C[i*nrow:(i + 1)*nrow]   
        if nrow == 1 and int(length/nrow) == 1:
            mat = mat[0]
    elif len(C.shape) == 2 and C.shape[0] == nrow: 
        mat = C   
    else:  
        print('R_matrix error')
        mat = 0
        
    return mat


def creat_color(return_color):

    print('\n============ Function in progress ==============')
    print(sys._getframe().f_code.co_name)
    print('================================================\n')
    
    color_group = np.array({'': '#DF3D8D', 'blackpink': '#C476F7', 'lightpink': '#D3B2ED', 'green': '#31D964', 'coffee': '#C37B69', 'lightblue': '#82DFE6', 'darkorange': '#D87D2C', 'lightblue2': '#47A1CA', 'coffee2': '#B69885', 'lake': '#96D7D8', 'darkpink': '#DC38C4', 'glass': '#42E194'},
      dtype=object)    
    color_group = color_group.item() # numpy.ndarray对象转换为dict
    print(type(color_group))

    if return_color:
        return color_group
    else:
        colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0,14)]
            
        x_data = ['2012']
        y_data = [58000]
        # 绘图
        plt.bar(x=x_data, height=y_data, label='color example', color="#"+color)
        plt.show()
    
        sign = input('included in colorgroup?(y/n):\n')
        if sign != 'n':
            color_name = input('select color_name? :\n')
            for key in color_group.keys():
                if color_name == key:
                    color_name = input('already exist, select color_name again? :\n')  
            color_group[color_name] = "#"+color
            print('\n')        
            return color_group       
        else:
            print('\n')             
            return "#"+color


def color_random():
    
    tt = creat_color(1)
    return tt[list(tt.keys())[random.randrange(0,len(list(tt.keys())))]]


class plt_text_setting():

    
    def __init__(self):
        
        self.out_setting = {} 
        self.box_setting = {}
        self.family_setting = {}
        
    def text_setting(self,color,fontsize,rotation,alpha,backgroundcolor,pre_setting):

        print('\n============ Function in progress ==============')
        print(sys._getframe().f_code.co_name)
        print('================================================\n')  
                
        self.out_setting['color'] = color        
        self.out_setting['size'] = fontsize
        self.out_setting['rotation'] = rotation
        self.out_setting['alpha'] = alpha    
        self.out_setting['backgroundcolor'] = backgroundcolor        
        
        if type(pre_setting) != dict:
            fontweight = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
            choose_fontweight = input('choose fontweight No.:[‘light’, ‘normal’, ‘medium’, ‘semibold’, ‘bold’, ‘heavy’, ‘black’]: \n')
            self.out_setting['weight'] = fontweight[int(choose_fontweight)-1]
            
            fontstyle = [ 'normal','italic','oblique' ]
            choose_fontstyle = input('choose fontstyle No.:[‘normal’ | ‘italic’ | ‘oblique’]: \n')
            self.out_setting['style'] = fontstyle[int(choose_fontstyle)-1]
            
            verticalalignment = ['center' , 'top' , 'bottom' ,'baseline']
            choose_verticalalignment = input('choose verticalalignment No.:[‘center’ , ‘top’ , ‘bottom’ ,‘baseline’]: \n')
            self.out_setting['verticalalignment'] = verticalalignment[int(choose_verticalalignment)-1]   
            
            horizontalalignment = ['left' , 'center' , 'right']
            choose_horizontalalignment = input('choose horizontalalignment No.:[‘left,center,right]: \n')
            self.out_setting['horizontalalignment'] = horizontalalignment[int(choose_horizontalalignment)-1]  
        else:
            self.out_setting['weight'] = pre_setting['weight']
            self.out_setting['style'] = pre_setting['style']  
            self.out_setting['verticalalignment'] = pre_setting['verticalalignment']
            self.out_setting['horizontalalignment'] = pre_setting['horizontalalignment']           
        
        print(self.out_setting)
        print('\n')        
        return self.out_setting
    
    def text_bbox_setting(self,facecolor,edgecolor,edgewidth,alpha):
    
        print('\n============ Function in progress ==============')
        print(sys._getframe().f_code.co_name)
        print('================================================\n')
        
        self.box_setting['facecolor'] = facecolor
        self.box_setting['edgecolor'] = edgecolor
        self.box_setting['edgewidth'] = edgewidth    
        self.box_setting['bbox_alpha'] = alpha        

        print(self.box_setting)
        
        return self.box_setting
    
    def text_family(self):

        print('\n============ Function in progress ==============')
        print(sys._getframe().f_code.co_name)
        print('================================================\n')
        
        family = ['fantasy', 'serif', 'SimHei', 'Times New Roman', 'Arial']
        choose_family = input('choose fontweight No.:[''fantasy | serif | SimHei | Times New Roman | Arial'']: \n')
        self.family_setting['family'] = family[int(choose_family)-1]
        
        print(self.family_setting)
        print('\n')        
        
        return self.family_setting        



def line_depict(d2_data,data1,savefig):
    
    fig, ax = plt.subplots(nrows=len(d2_data), ncols=len(d2_data[0]),figsize = (len(d2_data[0])*d2_data[0][0]['figsize'][0],len(d2_data)*d2_data[0][0]['figsize'][1]),squeeze=False)
    
    color_group = []  
        
    for i in range(len(d2_data)):
        for j in range(len(d2_data[0])):
            data = d2_data[i][j]
            
            if 'color' not in data['line'].keys() and i == 0 and j == 0:
                for i_color in range(data['y'].shape[1]):
                    color = color_random()
                    color_group.append(color)
            else:
                for i_color in range(len(data['line']['color'])):
                    color_group.append(data['line']['color'][i_color])          
            #print(data)
            #print(data1)
            for i_plot in range(data['y'].shape[1]): 
                if 'legend' in data1.keys():
                    ax[i,j].plot(data['x'],data['y'][:,i_plot],color=color_group[i_plot],linestyle=data['line']['linestyle'][i_plot],linewidth = data['line']['linewidth'][i_plot],label = data1['legend']['label'][i_plot])
                else:
                    ax[i,j].plot(data['x'],data['y'][:,i_plot],color=color_group[i_plot],linestyle=data['line']['linestyle'][i_plot],linewidth = data['line']['linewidth'][i_plot])                    
            ax[i,j].set_title(data['title'],fontdict= data1['font_title'], pad= data1['title_height'])
            ax[i,j].set(xlim=data['xlim'], ylim=data['ylim'])

            ax[i,j].set_xlabel(data['x_title'],fontdict = data1['font_xytitle'])
            ax[i,j].set_ylabel(data['y_title'],fontdict = data1['font_xytitle'])
            if 'x_label' in data.keys():
                ax[i,j].set_xticklabels(data['x_label'],fontdict = data1['font_ticks'])
            if 'y_label' in data.keys():
                ax[i,j].set_yticklabels(data['y_label'],fontdict = data1['font_ticks'])
            if 'text_x' in data.keys():
                for i_text in range(len(data['text_x'])):
                    ax[i,j].text(data['text_x'][i_text],data['text_y'][i_text], data['text'][i_text], fontdict=data1['text_family'], wrap=True)
            if 'y_ticks' in data1.keys():
                #ax[i,j].invert_yaxis()
                ax[i,j].set_yticks(data1['y_ticks'])
            if 'x_ticks' in data1.keys():
                ax[i,j].set_xticks(data1['x_ticks'])
            if 'yticks' in data1.keys() and data1['yticks']:
                #ax[i,j].invert_yaxis()
                ax[i,j].set_yscale('symlog')
            if 'xticks' in data1.keys() and data1['xticks']:
                ax[i,j].set_xscale('symlog')
            if 'axhline'in data1.keys():
                ax[i,j].axhline(y= data1['axhline']['value'] , color=data1['axhline']['color'] , linestyle=data1['axhline']['linestyle'],linewidth = data1['axhline']['linewidth'])
            if 'axvline'in data1.keys():
                ax[i,j].axvline(x= data1['axvline']['value'] , color=data1['axvline']['color'] , linestyle=data1['axvline']['linestyle'],linewidth = data1['axvline']['linewidth'])
            if 'bar' in data.keys():
                ax[i,j].bar(data['bar']['xticks'],data['bar']['value'],width = data['bar']['width'],color  = data['bar']['color'],edgecolor = data['bar']['edgecolor'],alpha  = data['bar']['alpha'],linewidth = data['bar']['linewidth'])
            ax[i,j].tick_params(labelsize = data1['tick_params']['fontsize'])
            ax[i,j] = plt.gca()
            if 'legend' in data1.keys():
                #patches = [mpatches.Patch(color=color_group[i], label="{:s}".format(data1['label_legend'][i]) ) for i in range(len(color_group))]     #用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
                #ax[i,j].legend(handles=patches,loc=data1['loc_legend'], fontsize=data1['fontsize_legend'], frameon=False, fancybox=True, framealpha=data1['alpha_legend'], ncol=data1['ncol_legend'],shadow = True)
                ax[i,j].legend(loc=data1['legend']['loc'], fontsize=data1['legend']['fontsize'], frameon=False, fancybox=True, framealpha=data1['legend']['alpha'], ncol=data1['legend']['ncol'],shadow = True)

    if type(savefig) == dict:
        path_dir(savefig['path'])
        plt.savefig(savefig['name'] + '.' + savefig['format'], dpi = savefig['dpi'])                
    plt.show()
    

    
def depict_para_npy(save_load):
    
    print('line_depict_para_npy')
    if type(save_load) != str:
        save_or_load = int(input('save or delete? (1/2) : '))
    else:
        save_or_load = 0
    para_group = {'test': {2: 0, 'test': 'test'},
 'exit': 1,
 'Fig_7_map_inf1': [[{'figsize': [10, 7],
    'title': '',
    'x_title': '',
    'y_title': '',
    'xlim': [0, 3000],
    'ylim': [0, 0.035],
    'x': np.array([0.000e+00, 1.000e+00, 2.000e+00, ..., 2.997e+03, 2.998e+03,
           2.999e+03]),
    'y': np.array([[ 0.00000000e+000,  1.12359551e-010,  5.91715976e-011],
           [ 3.06974502e-111, -2.13971601e-009, -2.36025358e-009],
           [ 3.63232877e-098, -4.31549979e-009, -4.73661021e-009],
           ...,
           [ 4.40523651e-025,  1.68014410e-012,  5.40812449e-004],
           [ 4.25157444e-025,  1.68014410e-012,  5.40812449e-004],
           [ 4.10325250e-025,  1.68014410e-012,  5.40812449e-004]]),
    'line': {'linestyle': ['-', '-', '-'],
     'linewidth': ['3', '3', '3'],
     'color': ['blue', 'red', 'green']},
    'x_label': ['0', '50', '100', '150', '200', '250', '300']}]],
 'Fig_7_map_inf2': {'font_title': {'family': 'Times New Roman',
   'color': 'black',
   'weight': 'medium',
   'size': 26},
  'title_height': 20,
  'font_xytitle': {'family': 'Times New Roman',
   'color': 'black',
   'weight': 'semibold',
   'size': 22},
  'text_family': {'family': 'Times New Roman',
   'color': 'black',
   'weight': 'bold',
   'size': 22},
  'yticks': 0,
  'xticks': 0,
  'legend': {'loc': 'upper right',
   'fontsize': 20,
   'alpha': 0.5,
   'ncol': 1,
   'label': ['a', 'b', 'c']},
  'font_ticks': {'color': 'black', 'weight': 'medium'},
  'tick_params': {'fontsize': 15}},
 'Allen_2003_p2': {'scatter': {'x': np.array([[-6.903658  ],
          [-3.865246  ],
          [ 0.2264297 ],
          [-0.09945696],
          [ 6.145548  ],
          [ 4.634795  ],
          [13.12407   ],
          [20.56177   ]]), 'y': np.array([[-2.67024],
          [-6.29148],
          [-7.46804],
          [ 2.77877],
          [ 9.84079],
          [ 3.63747],
          [11.45514],
          [13.55746]]), 'color': 'black', 'marker': 'o', 'alpha': 0.8, 'linewidth': 1},
  'polyfit': {'color': 'black', 'linewidth': 1},    
  'x_title': 'Model-simulated values of signal components',
    'y_title': 'Obeserved values of signal components',
  'font_xytitle': {'family': 'Times New Roman',
   'color': 'black',
   'weight': 'semibold',
   'size': 22}},
 'Allen_1999_p5_inf1': [[{'figsize': [10, 7],
    'title': 'S/N optimised, mass weight',
    'x_title': 'No of EOF pattern',
    'y_title': 'cumulative model/observed residual variance ratio',
    'xlim': [2, 8],
    'ylim': [0, 10],
    'x': np.array([2., 3., 4., 5., 6., 7., 8.]),
    'y': np.array([[  3.28437788, 254.31444455,   0.26031777, 253.89208811,
              0.25823548],
           [  3.33409211,  19.49572575,   0.3338082 ,  19.49241468,
              0.33050789],
           [  2.28501199,   8.5264499 ,   0.38389053,   8.5355798 ,
              0.37957836],
           [  3.04629138,   5.62807153,   0.4215972 ,   5.6400532 ,
              0.41639932],
           [  3.67258275,   4.36499675,   0.45165088,   4.37835577,
              0.44565523],
           [  2.35515412,   3.66886557,   0.47650863,   3.68317365,
              0.46978131],
           [  2.24798261,   3.22975075,   0.49761357,   3.24482973,
              0.49020672]]),
    'line': {'linestyle': ['-', '--', '--', '--', '--'],
     'linewidth': ['3', '2.5', '2.5', '2.5', '2.5'],
     'color': ['black', 'grey', 'grey', 'grey', 'grey']},
    'text_x': [3, 3, 6],
    'text_y': [0.5, 2, 5.5],
    'text': ['too low', 'adequate', 'too high']}]],
 'Allen_1999_p5_inf2': {'font_title': {'family': 'Times New Roman',
   'color': 'black',
   'weight': 'medium',
   'size': 26},
  'title_height': 20,
  'font_xytitle': {'family': 'Times New Roman',
   'color': 'black',
   'weight': 'semibold',
   'size': 22},
  'text_family': {'family': 'Times New Roman',
   'color': 'black',
   'weight': 'bold',
   'size': 22},
  'yticks': 1,
  'xticks': 0,
  'axhline': {'value': 1,
   'linestyle': '-.',
   'linewidth': 2.5,
   'color': 'grey'},
  'tick_params': {'fontsize': 18}},
 'Ribes_2013_7p_inf1': [[{'figsize': [10, 7],
    'title': '',
    'x_title': '',
    'y_title': '',
    'xlim': [0, 3000],
    'ylim': [0, 0.035],
    'x': np.array([0.000e+00, 1.000e+00, 2.000e+00, ..., 2.997e+03, 2.998e+03,
           2.999e+03]),
    'y': np.array([[ 0.00000000e+000,  1.12359551e-010,  5.91715976e-011],
           [ 3.06974502e-111, -2.13971601e-009, -2.36025358e-009],
           [ 3.63232877e-098, -4.31549979e-009, -4.73661021e-009],
           ...,
           [ 4.40523651e-025,  1.68014410e-012,  5.40812449e-004],
           [ 4.25157444e-025,  1.68014410e-012,  5.40812449e-004],
           [ 4.10325250e-025,  1.68014410e-012,  5.40812449e-004]]),
    'line': {'linestyle': ['-', '-', '-'],
     'linewidth': ['3', '3', '3'],
     'color': ['blue', 'red', 'green']},
    'x_label': ['0', '50', '100', '150', '200', '250', '300']}]],
 'Ribes_2013_7p_inf2': {'font_title': {'family': 'Times New Roman',
   'color': 'black',
   'weight': 'medium',
   'size': 26},
  'title_height': 20,
  'font_xytitle': {'family': 'Times New Roman',
   'color': 'black',
   'weight': 'semibold',
   'size': 22},
  'text_family': {'family': 'Times New Roman',
   'color': 'black',
   'weight': 'bold',
   'size': 22},
  'yticks': 0,
  'xticks': 0,
  'legend': {'loc': 'upper right',
   'fontsize': 20,
   'alpha': 0.5,
   'ncol': 1,
   'label': ['a', 'b', 'c']},
  'font_ticks': {'color': 'black', 'weight': 'medium'},
  'tick_params': {'fontsize': 15}}}
    if save_or_load == 1:
        para_name = str(input('para key name? : '))
        para_group[para_name] = save_load
        print('\n keys of para_group up to now: ')
        print(para_group.keys())
        np.save('para_group.npy',para_group)    
        return para_group.keys()
    elif save_or_load == 2:
        para_name = str(input('para key name? : '))
        e1 = para_group.pop(para_name)
        print('below have been deleted: ')
        np.save('para_group.npy',para_group) 
        return e1
    elif type(save_load) == str:
        print('load para ...')
        print('\n keys of para_group up to now: ')
        print(para_group.keys())        
        return para_group[save_load]
    else:
        print('end:')
        return 0
    
    
def scatter_depict(data,savefig):
    
    fig = plt.figure(figsize = (7,7),facecolor='white')
    
    seris = np.linspace(np.min(data['scatter']['x']),np.max(data['scatter']['x']),100)
    #xy_min = [min(data['scatter']['x']),min(data['scatter']['y'])]
    #xy_max = [max(data['scatter']['x']),max(data['scatter']['y'])]      
    plt.scatter(data['scatter']['x'],data['scatter']['y'],color = data['scatter']['color'],\
                marker = data['scatter']['marker'],alpha = data['scatter']['alpha']\
                ,linewidth = data['scatter']['linewidth'])
    if 'recons_scatter' in data.keys():
        plt.scatter(data['recons_scatter']['x'],data['recons_scatter']['y'],color = data['recons_scatter']['color'],\
                    marker = data['recons_scatter']['marker'],alpha = data['recons_scatter']['alpha']\
                    ,linewidth = data['recons_scatter']['linewidth']) 
        #xy_min = [min(min(data['scatter']['x']),min(data['recons_scatter']['x'])),min(min(data['scatter']['y']),min(data['recons_scatter']['y']))]
        #xy_max = [max(max(data['scatter']['x']),max(data['recons_scatter']['x'])),max(max(data['scatter']['y']),max(data['recons_scatter']['y']))]
    if 'polyfit' in data.keys():
        try:
            pfit = np.polyfit(data['scatter']['x'],data['scatter']['y'],1)
        except:
            pfit = np.polyfit(T(data['scatter']['x'])[0,:],T(data['scatter']['y'])[0,:],1)
        y_fun = np.poly1d(pfit)  
        plt.plot(seris,y_fun(seris),color = data['polyfit']['color'] ,linewidth = data['polyfit']['linewidth'],linestyle = data['polyfit']['linestyle']) 
    if 'regression' in data.keys():
        reg_y = np.zeros(shape = len(seris))
        for i_num in range(len(data['regression']['beta'])):
            for i in range(len(seris)):
                reg_y[i] = (seris[i] - data['regression']['b'][i_num]) * data['regression']['beta'][i_num] + data['regression']['u'][i_num]
            plt.plot(seris,reg_y,color = data['regression']['color'][i_num] ,linewidth = data['regression']['linewidth'][i_num],linestyle = data['regression']['linestyle'][i_num]) 
    if 'arrow' in data.keys():
        """
        剪头起始位置（A[0],A[1]）和终点位置（B[0],B[1]）
        length_includes_head = True:表示增加的长度包含箭头部分
        head_width:箭头的宽度
        head_length:箭头的长度
        fc:filling color(箭头填充的颜色)
        ec:edge color(边框颜色)
        ax.arrow(data['arrow']['x'][0],data['arrow']['x'][1],data['arrow']['y'][0]-data['arrow']['x'][0],data['arrow']['y'][1]-data['arrow']['x'][1],\
                 length_includes_head = True,head_width = 0.25,head_length = 0.5,fc = 'b',ec = 'b')
        """
        plt.annotate(data['arrow']['text'],xy=(data['arrow']['y'][0],data['arrow']['y'][1]),xytext=(data['arrow']['x'][0],data['arrow']['x'][1]),arrowprops=dict(connectionstyle="arc3",color='black',width = 1))
            
    plt.xlabel(data['x_title'],fontdict = data['font_xytitle'])
    plt.ylabel(data['y_title'],fontdict = data['font_xytitle'])
    plt.title(data['title'])
    plt.axis('equal')
    fig.set_facecolor('white')
    if type(savefig) == dict:
        path_dir(savefig['path'])
        plt.savefig(savefig['name'] + '.' + savefig['format'],facecolor = fig.get_facecolor(), dpi = savefig['dpi'])
    plt.show()
    

def figname_affix(variable,varname):
    
    if not variable:
        return ''
    
    else:
        return varname
    
    
    
def stdev(data,*ori):
    
    try:
        ori = ori[0]
    except: 
        ori = 0
    if len(data.shape) > 1:
        if ori == 'r': 
            l=len(data)
            m=sum(data)/l
            d=0
            for i in data: d+=(i-m)**2
            return (d*(1/(l-1)))**0.5    
        
        elif ori == 'c':       
            l=data.shape[1]
            m=sum(T(data))/l
            d=0
            for i in T(data): d+=(i-m)**2
            return (d*(1/(l-1)))**0.5      
        
        else:           
            data_1d =[i for j in data for i in j]
            l=len(data_1d)
            m=sum(data_1d)/l
            d=0
            for i in data_1d: d+=(i-m)**2
            return (d*(1/(l-1)))**0.5              
        
    elif len(data.shape) == 1:
        
        l=len(data)
        m=sum(data)/l
        d=0
        for i in data: d+=(i-m)**2
        return (d*(1/(l-1)))**0.5 

    else:
        
        print('error in data')      
        return 0    
    
def Zero_centered(data,ori):
    
    if ori == 'c':
        for i in range(data.shape[1]):
            mean_data = np.mean(data[:,i])
            for j in range(data.shape[0]):
                data[j,i] = data[j,i] - mean_data
    elif ori == 'r':
        for i in range(data.shape[0]):
            mean_data = np.mean(data[i,:])
            for j in range(data.shape[1]):
                data[i,j] = data[i,j] - mean_data        
    
    return data

def T(M):
    # 初始化转置后的矩阵
    result = []
    # 获取转置前的行和列
    try:
        row,col = M.shape
          # 先对列进行循环
        for i in range(col):
            # 外层循环的容器
            item = [] 
            # 在列循环的内部进行行的循环
            for index in range(row):
                item.append(M[index][i])
            result.append(item)
    except:
        row = 1
        col = M.shape[0]
          
        for i in range(col):
            # 外层循环的容器
            item = [] 
            # 在列循环的内部进行行的循环
            for index in range(row):
                item.append(M[i])
            result.append(item)
            
    return np.asarray(result) # list -> array

def multi_dot(*a):
    
    num = len(a)
    temp = a[0]
    output = 0
    for i in range(num - 1):
        output = np.dot(temp,a[i+1])
        temp = output
        
    return output

def svd(M):
    
    svd_m = np.linalg.svd(M)
    m,n = M.shape
    out = {}
    out['u'] = svd_m[0][:,:n]
    out['d'] = svd_m[1]
    out['v'] = svd_m[2][:m,:].T
    
    return out

def Sym_eigen(M):
    
    V,P = np.linalg.eigh(M)
    idx = V.argsort()[::-1]
    V = V[idx]
    P = P[:,idx]
    
    return V,P

def confidence_cross_ellipse(confi_region,s,depict_scale,data,xlabel,ylabel,title):

    x = confi_region[:,0]
    y = confi_region[:,1]
    
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)      
    x_mean = np.mean(x)  
    y_mean = np.mean(y)  

    cov = np.cov(x, y) #计算协方差矩阵
    lambda_, v = Sym_eigen(cov) # 计算矩阵特征向量 np.linalg.eig 存在偏差
    lambda_ = np.sqrt(lambda_)
    fig = plt.figure(figsize = (depict_scale))
    ax = fig.add_subplot(111)
    #ax = plt.subplot(111,aspect='equal')
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
    width=lambda_[0]*np.sqrt(s) *2, height=lambda_[1]*np.sqrt(s)*2,angle=np.rad2deg(np.arccos(v[0, 0])),facecolor='black',alpha=0.3)
    ax.add_artist(ell)
    ax.scatter(x, y)
    plt.axis('scaled')
    plt.axis('equal')
    ax.set_xlim(2*x_min - x_mean,2*x_max - x_mean)
    ax.set_ylim(2*y_min - y_mean,2*y_max - y_mean)  
        
    c_x = data[0,1]
    c_y = data[1,1]
     
    xerr_length = ((data[0,2] - data[0,1]) + (data[0,1] - data[0,0]))/2
    yerr_length = ((data[1,2] - data[1,1]) + (data[1,1] - data[1,0]))/2
    
    ax.errorbar(c_x, c_y, fmt="o", yerr=yerr_length, xerr=xerr_length, ecolor='grey', elinewidth=2, capsize=4, capthick=None,color = 'black')
    
    plt.title(title,fontsize = 28)
    plt.xlabel(xlabel,fontsize=20)  
    plt.ylabel(ylabel,fontsize=20)   
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)    
    plt.show()  


def point_to_ellipse(ell_group,beta,s,depict_scale,xlabel,ylabel,title,xlim,ylim,text_parameters,ori,savefig):

    fig = plt.figure(figsize = (depict_scale),facecolor='snow')
    ax = fig.add_subplot(111)
    c_x = beta[0,1]
    c_y = beta[1,1]
    ax.scatter(c_x, c_y,c = 'black')

    for i_key in list(ell_group.keys()):
        
        confi_region = ell_group[i_key]
        x2 = confi_region[:,0]
        y2 = confi_region[:,1]
        cov = np.cov(x2, y2) #计算协方差矩阵
        lambda_, v = Sym_eigen(cov) # 计算矩阵特征向量 np.linalg.eig 存在偏差
        lambda_ = np.sqrt(lambda_)
        if i_key == 'beta2_max' or i_key == 'beta2_max_95' or i_key == 'tls_confidence':
            if type(ori) == dict: # angle corrected
                if int(np.rad2deg(np.arccos(v[0, 0]))) not in ori[i_key]:
                    angle = 180 - np.rad2deg(np.arccos(v[0, 0]))
                else:
                    angle = np.rad2deg(np.arccos(v[0, 0]))
            else:
                angle = np.rad2deg(np.arccos(v[0, 0]))
            ell = Ellipse(xy=(np.mean(x2), np.mean(y2)),
            width=lambda_[0]*np.sqrt(s) *2, height=lambda_[1]*np.sqrt(s)*2,angle=angle,facecolor='none', zorder=10, edgecolor = color_random(),linewidth = 2,label = i_key,alpha = 0.8)
        else:
            if type(ori) == dict:
                if int(np.rad2deg(np.arccos(v[0, 0]))) not in ori[i_key]:
                    angle = 180 - np.rad2deg(np.arccos(v[0, 0]))
                else:
                    angle = np.rad2deg(np.arccos(v[0, 0]))
            else:
                angle = np.rad2deg(np.arccos(v[0, 0]))           
            ell = Ellipse(xy=(np.mean(x2), np.mean(y2)),
            width=lambda_[0]*np.sqrt(s) *2, height=lambda_[1]*np.sqrt(s)*2,angle=angle,facecolor='none', zorder=10, edgecolor = color_random(),linewidth = 2,label = i_key,alpha = 0.8,linestyle = '--')

        print('original angle')        
        print(np.rad2deg(np.arccos(v[0, 0])))        
        print('corrected angle')        
        print(angle)
        ax.add_patch(ell)
    #plt.axis('scaled')
    #plt.axis('equal')

    if beta.shape == (2, 3):
        xerr_length = ((beta[0,2] - beta[0,1]) + (beta[0,1] - beta[0,0]))/2
        yerr_length = ((beta[1,2] - beta[1,1]) + (beta[1,1] - beta[1,0]))/2
        ax.errorbar(c_x, c_y, fmt="o", yerr=yerr_length, xerr=xerr_length, ecolor='black', elinewidth=2, capsize=6, capthick=None,color = 'black',alpha = 0.8)

    note_x_distance = (xlim[-1] - xlim[0])/100
    note_y_distance = (ylim[-1] - ylim[0])/100    
    plt.text(c_x + note_x_distance , c_y + note_y_distance, 'Best fit', size = text_parameters['size'],\
         family = 'Times New Roman', color = text_parameters['color'], style = text_parameters['style'], weight = text_parameters['weight'])
 
    if 0 > xlim[0] and 0 < xlim[1]:
        plt.axvline(0,color = 'black',linestyle = '-',linewidth = 1,alpha = 0.7) 
        if 0 > ylim[0] and 0 < ylim[1]:
            plt.axhline(0,color = 'black',linestyle = '-',linewidth = 1,alpha = 0.7)
            ax.scatter(0, 0,c = 'black')
            plt.text(note_x_distance , note_y_distance, '(0,0)', size = text_parameters['size'],\
                 family = 'Times New Roman', color = text_parameters['color'], style = text_parameters['style'], weight = text_parameters['weight'])
        
    if note_x_distance > note_y_distance:
        ax.set_xlim(xlim)
        legned_ncol = 2
    else:
        ax.set_ylim(ylim)
        legned_ncol = 1        
    plt.axis('equal')
    plt.title(title,fontsize = 28)
    plt.xlabel(xlabel,fontsize=20)  
    plt.ylabel(ylabel,fontsize=20)   
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20) 
    plt.legend(prop={'size':15},frameon = True,ncol=legned_ncol,loc = 'upper right',shadow = True)
    if type(savefig) == dict:
        path_dir(savefig['path'])
        plt.savefig(savefig['name'] + '.' + savefig['format'], dpi = savefig['dpi'])
    for i_key in list(ell_group.keys()):
        ax.scatter(x2, y2,alpha = 0.1)
    plt.show()  


def variety_Z(sim_Z,color,P,alpha): #图画规格，图画数量

    plt.figure(figsize = [10,7])
    for i in range(sim_Z['X'].shape[-1]):
        plt.plot(sim_Z['X'][:,i],color = color_random(),label = 'model'+str(i+1))
    plt.plot(sim_Z['Y'],color = 'blue',label = 'obs')

    data = sim_Z['noise1']
    X = range(0,data.shape[-1])
    out_data = np.zeros(shape=(2,data.shape[-1]))
    for i in range(data.shape[-1]):
        out_data[0,i] = np.percentile(data[:,i], P)
        out_data[1,i] = np.percentile(data[:,i], 100-P)
    plt.fill_between(X,out_data[0,:],out_data[1,:],color=color[0],alpha=alpha,label = 'noise1')

    data = sim_Z['noise2']
    out_data = np.zeros(shape=(2,data.shape[-1]))
    for i in range(data.shape[-1]):
        out_data[0,i] = np.percentile(data[:,i], P)
        out_data[1,i] = np.percentile(data[:,i], 100-P)
    plt.fill_between(X,out_data[0,:],out_data[1,:],color=color[1],alpha=alpha,label = 'noise2')
    plt.title('variety of Z')
    plt.legend(prop={'size':20},frameon = True,ncol=3,loc = 'upper right',shadow = True)
    plt.show()
    
    
def sqrt_matrix(aa):
    
    if aa.shape[0] != aa.shape[1]:
        print('sqrt.matrix: input matrix is not square')
    a_eig = Sym_eigen(aa)
    a_sqrt = np.dot(np.dot(a_eig[1],R_diag(np.sqrt(a_eig[0]))),np.linalg.inv(a_eig[1]))
        
    return a_sqrt
        
def in_sqrt_matrix(aa):
        
    if aa.shape[0] != aa.shape[1]:
        print('in.sqrt.matrix: input matrix is not square')        
    a_eig = Sym_eigen(aa)        
    a_sqrt = np.dot(np.dot(a_eig[1],R_diag(1/np.sqrt(a_eig[0]))),np.linalg.inv(a_eig[1]))
            
    return a_sqrt


# given space-temporal structure and p1,p2 as base period start and end point
# location, return EOF(,nt-1) for reduce dimension operator
def redop(nt,p1,p2):
    if p1 < 1 or p1 > nt:
        print('input p1 error in reduce')
    if p2 < 1 or p2 > nt:
        print('input p2 error in reduce') 
    M = np.diag(np.ones(shape = nt))
    M[:,p1-1:p2] = M[:,p1-1:p2] - 1/(p2 - p1 + 1)
    eM = Sym_eigen(M)
    if abs(eM[0][nt - 1]) > 0.00005:
        print('ev error in reduce')
    if any(abs(eM[0][:nt - 1]) < 0.00005):
        print('ev error2 in reduce')    
    u = eM[1][:,:nt - 1]
    
    return u

# usage:
# for example, vector of obs length as 50, listed as 10yrs x 5points (data 1~5 for year 1,
#     6~10 for year 2, so on so for, this kind of sequence should apply to all X and noise
#     pieces), then nt=10, ns=5
#     if the data was anomanies wrt year 6~9, namely baseperiod as 6~9, then p1=6, p2=9
# u<-redop(nt=8,p1=1,p2=6)

# given reduce dimension operator u and data vector x and space-temporal structure nt,ns
# return reduced vector (nt-1)*ns
def redvec(u,xx,nt,ns):
    """
    xx = Y
    u = redop_vectors[:,:-1]
    nt = 8
    ns = 1
    """
    timefirst = True

    if len(xx) != nt*ns:
        print('input dim error in redvec')
    if not operator.eq(list(u.shape),[nt,nt - 1]):
        print('input u dim error in redvec')
    x1 = R_matrix(xx,nt,timefirst)
    ox = np.dot(T(u),x1)
    
    return T(ox)

# redECOF is a function to reduce everything (Y,X,noise1,noise2) from a ECOF object and return
# a new ECOF object contains reduced elements (Y,X,noise1,noise2)
def redEOF(Z,u,nt,ns):
    
    newY = T(redvec(u,Z['Y'],nt,ns))
    newX = T(redvec(u,Z['X'],nt,ns))
    newnoise1 = np.zeros(shape = (Z['noise1'].shape[0],len(u) - 1))   
    for i in range(Z['noise1'].shape[0]):
        newnoise1[i,:] = T(redvec(u,Z['noise1'][i],nt,ns))[:,0]
    newnoise2 = np.zeros(shape = (Z['noise2'].shape[0],len(u) - 1))   
    for i in range(Z['noise2'].shape[0]):
        newnoise2[i,:] = T(redvec(u,Z['noise2'][i],nt,ns))[:,0]       
    
    Zr = {}
    Zr['X'] = newX
    Zr['Y'] = newY  
    Zr['noise1'] = newnoise1   
    Zr['noise2'] = newnoise2  
    
    return Zr

def Creg(Cn):
    
    # regularization for noise structure δ֪
    # input Cn is an n x p matrix, n is number of pieces of noise(each line in file2,3); 
    # nx is point number for each piece
    nx = Cn.shape[1]
    n = Cn.shape[0]
    CE = np.dot(T(Cn),Cn/n)
    Ip = R_diag(np.ones(shape = nx))
    m = np.mean(R_diag(CE))
    XP = CE - R_diag(np.ones(shape = nx)*m)
    d2 = np.mean(R_diag(np.dot(XP,T(XP))))
    bt = np.zeros(shape = n)
    for i in range(n):
        Mi = np.dot(T(Cn[i,:]),T(T(Cn[i,:])))
        bt[i] = np.mean(R_diag(np.dot((Mi - CE),T(Mi - CE))))
    bb2 = np.sum(bt)/n**2
    b2 = min(bb2,d2)
    a2 = d2 - b2
    Creg = b2*m/d2*Ip + a2/d2*CE
    
    return Creg

#1.ols(), the Ordinary Least Squares method (Allen_1999). 
#2.tls.A03 (), the Total Least Squares method (Allen_2003). Note that the confidence intervals for the scaling factors are obtained using the method provided in the ROF package by Dr. A. Ribes (Ribes_2012)  
class tls_ols():
    def __init__(self,Z,savefig):
        self.Y = Z['Y']
        self.X = Z['X'] 
        self.noise1 = Z['noise1']    
        self.noise2 = Z['noise2']
        self.nsim_CI = 1000
        self.savefig = savefig        
# sample command lines:
# x<-readin('obs_sig.txt','noise1.txt','noise2.txt')
# o1<-ols(x@Y,x@X,x@noise1,x@noise2,df2=NULL)
    def ols(self,ols_para):
        
        nsig = ols_para['nsig']
        df2 = 0
        plev = ols_para['plev']
        REG = ols_para['REG']
        tilde = ols_para['tilde']
        All_EOF = ols_para['All_EOF']
        
        n = self.Y.shape[0]
        try:
            nx = self.X.shape[1]
        except:
            nx = 1
        nn1 = self.noise1.shape[0]
        nn2 = self.noise2.shape[0]  
        nmin = np.min(np.array([n,nn1,nn2]))
        
        if len(nsig) != nx:
            return ('nsig dont match nx')
                
        if not df2:
            df2 = nn2
            
        if df2 - nmin +1 < 0:
            return ('df2 too small to carry out residual check')

        if REG:
            C1 = Creg(self.noise1)
        else:
            C1 = np.dot(T(self.noise1),(self.noise1/nn1)) #  C1 = Cn  # Cov(cancha) = C    Ribes_2013 f5        【 3 】
        #  Allen_1999 f3
        ev1 = Sym_eigen(C1)[0]
        eof1 = Sym_eigen(C1)[1] # EOFs of the control / eigenvectors of Cn^
        p = np.zeros(shape =  eof1.shape)# P 
            
        for i in range(p.shape[1]):
            p[:,i] = eof1[:,i]/max(np.sqrt(ev1[i]),0.00005) # pre-whitening coordinate transformation 
                # Allen_1999 f3
                # 􏳉 highest-variance EOFs of the control weighted by their inverse singular values (square root of the corresponding eigenvalue of C` D).
                
        # calculate second covariance matrix (cn2) from n2
        C2 = np.dot(T(self.noise2),(self.noise2/nn2))
        PX = np.dot(T(eof1),self.X)
        PY = np.dot(T(eof1),self.Y)  
        noise2t = np.dot(self.noise2,eof1)
    
        ev_2 = Sym_eigen(C2)[0]
        eof_2 = Sym_eigen(C2)[1] # EOFs of the control / eigenvectors of Cn^
        p2 = np.zeros(shape =  eof_2.shape)# P 
                
        for i in range(p2.shape[1]):
            p2[:,i] = eof_2[:,i]/max(np.sqrt(ev_2[i]),0.00005) # pre-whitening coordinate transformation 
                                                                    # Allen_1999 f3
        if All_EOF:
            start_EOF = nx+1
        else:
            start_EOF = nmin
            
        for ith in range(start_EOF,nmin+1):
            
            fi = multi_dot(np.linalg.inv(multi_dot(T(self.X),p[:,:ith],T(p[:,:ith]),self.X)),T(self.X),p[:,:ith],T(p[:,:ith]))
            Cn_inv = np.dot(p[:,:ith],T(p[:,:ith]))
            
            tbeta = np.dot(fi,self.Y) #value
            vbeta = np.dot(np.dot(fi,C2),T(fi)) #  Allen_1999 f13
            p_vbeta = np.linalg.inv(multi_dot(T(self.X),p[:,:ith],T(p[:,:ith]),self.X))
            if REG:
                betalow = tbeta - T(t.ppf(plev,df2) * np.sqrt(R_diag(vbeta))) # Allen_1999 f11
                betaup = tbeta + T(t.ppf(plev,df2) * np.sqrt(R_diag(vbeta)))
            else:
                betalow = tbeta - T(t.ppf(plev,df2) * np.sqrt(R_diag(vbeta)) * np.sqrt((nsig +1)/nsig)) # Allen_1999 f11
                betaup = tbeta + T(t.ppf(plev,df2) * np.sqrt(R_diag(vbeta)) * np.sqrt((nsig +1)/nsig))                

            r2_beta = np.dot( T(np.dot(np.dot(T(p[:,:ith]),self.X),tbeta) - np.dot(T(p[:,:ith]),self.Y)),(np.dot(np.dot(T(p[:,:ith]),self.X),tbeta) - np.dot(T(p[:,:ith]),self.Y)))
            
            # Allen_1999 f15
            beta2 = []
            for i in range(self.noise2.shape[0]):
                sqrt_vbeta = R_diag(np.sqrt(R_diag(vbeta)))
                beta2.append(multi_dot(self.noise2[i,:],T(fi),np.linalg.inv(sqrt_vbeta),fi,T(self.noise2[i,:]))) # Allen_1999 f15
            beta2_max = np.max(beta2)
            beta2_max_95 = np.percentile(beta2,95)#95%分位数 

            cn2t = np.dot(T(noise2t[:,:ith]),noise2t[:,:ith])/nn2
            ev2 = Sym_eigen(cn2t)[0] 
            eof2 = Sym_eigen(cn2t)[1]  
            ev2[ev2 < 0.00005] = 0.00005 
            #print('vbeta '+ str(ith) + str(vbeta))
        # residual base on ith truncation            
            u = PY - np.dot(PX, tbeta)
            û = self.Y - np.dot(self.X, tbeta) # residual base on ith truncation

            # u -> residual term 
            if REG:
                stat = multi_dot(T(û),np.dot(p2[:,:ith],T(p2[:,:ith])),û)   # Ribes_2013 f14                        
            else:
                tmp = np.zeros(shape = ith)
                for j in range(ith):
                    tmp[j] = np.sum(T(u[:ith])*eof2[:ith,j])
                stat = np.sum(tmp**2/ev2[:ith])                                                
            dfn = ith -nx
            dfd = nn2 - ith + 1 
            f1 = dfn * df2 * f.ppf(.05,dfn,dfd)/(df2 - ith +1)
            f2 = dfn * df2 * f.ppf(.95,dfn,dfd)/(df2 - ith +1)           
            # Ribes_13 14 Both formulas are consistent, however, in the case where n2 􏰚 n. AT99 focused on the EOF projection approach, so the size n of Y was actually the EOF truncation k (i.e. n = k), which was rather small. Therefore, this assumption was more reasonable.
            if ith == start_EOF:
                output = {} 
                output['EOF'] = []
                output['beta_low'] = np.zeros(shape = (nmin + 1 - start_EOF,len(betalow)))
                output['beta_hat'] = np.zeros(shape = (nmin + 1 - start_EOF,len(tbeta)))
                output['beta_up'] = np.zeros(shape = (nmin + 1 - start_EOF,len(betaup)))
                output['RCstat'] = []
                output['RClow'] = []
                output['RCup'] = []  
                output['Cn_inv'] = []                
                output['vbeta'] = []
                output['p_vbeta'] = []                
                output['fi'] = []
                output['u'] = [] 
                output['û'] = []  
                output['beta2_max'] = []      
                output['beta2_max_95'] = []
                output['r2_beta'] = []                
                output['EOF'].append(ith)
                output['beta_low'][ith - start_EOF,:] =  T(betalow)
                output['beta_hat'][ith - start_EOF,:] =  T(tbeta)                    
                output['beta_up'][ith - start_EOF,:] =  T(betaup)                    
                output['RCstat'].append(stat)                       
                output['RClow'].append(f1)                       
                output['RCup'].append(f2)      
                output['Cn_inv'].append(Cn_inv)
                output['vbeta'].append(vbeta) 
                output['p_vbeta'].append(p_vbeta)               
                output['fi'].append(fi) 
                output['u'].append(u)           
                output['û'].append(û)        
                output['beta2_max'].append(beta2_max) 
                output['beta2_max_95'].append(beta2_max_95)    
                output['r2_beta'].append(r2_beta) 
                
            else:
                output['EOF'].append(ith)
                output['beta_low'][ith - start_EOF,:] =  T(betalow) 
                output['beta_hat'][ith - start_EOF,:] =  T(tbeta)                      
                output['beta_up'][ith - start_EOF,:] =  T(betaup)                     
                output['RCstat'].append(stat)                       
                output['RClow'].append(f1)                       
                output['RCup'].append(f2)
                output['Cn_inv'].append(Cn_inv)
                output['vbeta'].append(vbeta) 
                output['p_vbeta'].append(p_vbeta) 
                output['fi'].append(fi) 
                output['u'].append(u) 
                output['û'].append(û)          
                output['beta2_max'].append(beta2_max) 
                output['beta2_max_95'].append(beta2_max_95) 
                output['r2_beta'].append(r2_beta) 
                if tilde:
                    output['X_tilde'] = self.X
                    output['Y_tilde'] = self.X * T(tbeta)
                    
        return output
    
# Mandatory input: Y -- n x 1 matrix
#        X -- n x nx matrix, nx is number of signals
#        noise1 -- nnoise1 x n matrix, each row of this matrix is one piece of noise
#        noise2 -- nnoise2 x n matrix, each row of this matrix is one piece of noise
#        nsig -- vector, length of nx, ensemble size for each signal 
# Optional input:
#        nsim.CI -- simulation size for confidence interval, default as 1000
#        df2  --  degree of freedom for noise2, if no input will treat noise2 pieces as 
#                 independent, namely, df2=nnoise2
#        REG  -- regularization flag, apply regularization on noise1 or not, default as FALSE
#        plev -- confidence level used in residual check, defaul as 0.9
# output as matrix (n-nx) rows (corresponding to EOF number), each row contains EOF#, (beta_low,
#        beta_hat, beta_up) for every signal
# sample command lines:
# x<-readin('obs_sig.txt','noise1.txt','noise2.txt')
# o1<-tls.A03(x@Y,x@X,x@noise1,x@noise2,nsig=5,df2=220,REG=TRUE)

    def tls(self,tls_para):        
 
        nsig = tls_para['nsig']
        df2 = 0
        plev = tls_para['plev']
        REG = tls_para['REG']
        tilde = tls_para['tilde']
        All_EOF = tls_para['All_EOF']
            
        n = self.Y.shape[0]
        try:
            nx = self.X.shape[1]
        except:
            nx = 1
        nn1 = self.noise1.shape[0]
        nn2 = self.noise2.shape[0]  
        
        if len(nsig) != nx:
            return ('nsig dont match nx')

        nmin = np.min(np.array([n,nn1,nn2]))
        
        if df2 == 0:
            df2 = nn2

        if tilde:
            C1 = Creg(self.noise1)   
            Cf12 = in_sqrt_matrix(C1)
            Cfp12 = sqrt_matrix(C1)
            Xf = np.dot(T(self.X),Cf12)
            Yf = np.dot(T(self.Y),Cf12)
            Xf = Xf * T(np.sqrt(nsig)) * np.ones(n)
            M = np.row_stack((Xf,Yf))
            u_u = svd(M)['u']
            u_v = svd(M)['v']
            u_d = svd(M)['d'] #result different from scilab

            d_tilde = u_d
            d_tilde[nx] = 0
            d_tilde = np.diag(d_tilde)
            Z_tilde = multi_dot(u_u,d_tilde,T(u_v))
            X_tilde_white = Z_tilde[:nx,:]/T(np.sqrt(nsig)) * np.ones(n)
            Y_tilde_white = Z_tilde[nx:,:]
            X_tilde = np.dot(Cfp12,T(X_tilde_white))
            Y_tilde = np.dot(Cfp12,T(Y_tilde_white))
                    
        if REG:
            C1 = Creg(self.noise1)
        else:
            C1 = np.dot(T(self.noise1),(self.noise1/nn1))
        
        xx = np.zeros(shape = self.X.shape)
        try:
            for i in range(xx.shape[1]):
                xx[:,i] = self.X[:,i]
        except:
            for i in range(len(xx)):
                xx[i] = self.X[i]
        
        for i in range(nx):
            if nx == 1:
                xx[:] = xx[:]*np.sqrt(nsig[0])
            else:
                xx[:,i] = self.X[:,i]*np.sqrt(nsig[i]) # xx is adjusted X with nsig
                
        P = Sym_eigen(C1)[1]
        v1 = Sym_eigen(C1)[0]
        
        for i in range(int(np.min(np.array([n,nn1])))):
            P[:,i] = P[:,i]/np.sqrt(v1[i])
            
        P = T(P) # P p5re - writing operator
        
        Z = np.column_stack((np.dot(P,xx),np.dot(P,self.Y)))

        if All_EOF:
            start_EOF = nx+1
        else:
            start_EOF = n
            
        for ith in range(start_EOF,n+1):
            zz = Z[:ith,:]
            u_v = svd(zz)['v']
            u_d = svd(zz)['d']
            u_u = svd(zz)['u']  
            #corr_ZZ = multi_dot(-u_u,np.diag([0,u_d[1]]),T(u_v)) # tls_overview 2.2
            #corr_ZZ[:,0] = corr_ZZ[:,0]*np.sqrt(nsig)
            #appro_ZZ = multi_dot(u_u,np.diag([u_d[0],0]),T(u_v)) # tls_overview 2.2            
            #appro_ZZ[:,1] = appro_ZZ[:,1]*np.sqrt(nsig)    
            betahat = -(u_v[:nx,nx]/u_v[nx,nx]*np.sqrt(nsig))
            d = u_d**2
            nd = len(u_d)
            ui = u_u[:,nd - 1]
            n2t = np.dot(self.noise2,T(P[:ith,:]))
            r1_stat = d[nd-1]/(np.dot(np.dot(ui,T(n2t)),np.dot(n2t,T(ui)))/nn2)
            dhat = np.zeros(shape = nx+1)
            for i in range(nx+1):
                vi = u_u[:,i]
                dhat[i] = d[i]/(np.dot(np.dot(vi,T(n2t)),np.dot(n2t,T(vi)))/nn2)
            delta_dhat = dhat - np.min(dhat)
# sampling on critical ellipse to construct correspoding beta_simus,
# then get max/min value as upper/lower bound for scaling factor beta_hat
            Crit = f.ppf(plev,1,df2)
            if nx > 1:
                unit_tmp = R_matrix(np.random.randn(self.nsim_CI*nx),self.nsim_CI,nx) # == R_rnorm
                unit = np.zeros(shape = unit_tmp.shape)
                for i in range(unit_tmp.shape[0]):
                    unit[i,:] = unit_tmp[i,:]/np.sqrt(np.sum(unit_tmp[i,:]**2))                
            else:
                unit = T(np.array([1,-1]))
            ai = unit * np.sqrt(Crit)
            Nu = np.zeros(shape = (len(ai),1))
            bi = np.column_stack((ai,Nu))
            for i in range(nx):
                bi[:,i] = ai[:,i]/np.sqrt(delta_dhat[i])
            for i in range(len(bi)):
                bi[i,nx] = np.sqrt(1 - np.sum(bi[i,:nx]**2))
            betaup = np.zeros(shape = nx)
            betalow = np.zeros(shape = nx)
            vc_pts = np.dot(bi,T(u_v))
            for i in range(nx):
                vc_pts[:,i] = vc_pts[:,i]*np.sqrt(nsig[i])
            for j in range(nx):
                if not j:
                    beta_scatter = []
                beta_tmp = -vc_pts[:,j]/vc_pts[:,(nx)]
                betaup[j] = np.max(beta_tmp)
                betalow[j] = np.min(beta_tmp)
                beta_scatter.append(beta_tmp)
            
            if ith == start_EOF:
                output = {} 
                output['EOF'] = []
                output['beta_low'] = np.zeros(shape = (nmin + 1 - start_EOF,len(betalow)))
                output['beta_hat'] = np.zeros(shape = (nmin + 1 - start_EOF,len(betahat)))
                output['beta_up'] = np.zeros(shape = (nmin + 1 - start_EOF,len(betaup)))
                output['u_v'] = []
                output['RClow'] = []
                output['RCup'] = []         
                output['RCstat'] = []
                output['ZZ'] = []
                output['EOF'].append(ith)
                output['beta_scatter'] = []
                output['beta_low'][ith - start_EOF,:] =  betalow
                output['beta_hat'][ith - start_EOF,:] =  betahat                    
                output['beta_up'][ith - start_EOF,:] =  betaup                  
                output['u_v'].append(u_v)                       
                output['RClow'].append((ith-nx)*f.ppf(.05,ith-nx,df2))                       
                output['RCup'].append((ith-nx)*f.ppf(.95,ith-nx,df2))      
                output['RCstat'].append(r1_stat) 
                output['ZZ'].append(Z) 
                output['beta_scatter'].append(beta_scatter)
                
                if tilde:
                    output['X_tilde'] = X_tilde
                    output['Y_tilde'] = Y_tilde
                
            else:
                output['EOF'].append(ith)
                output['beta_low'][ith - start_EOF,:] =  betalow 
                output['beta_hat'][ith - start_EOF,:] =  betahat                      
                output['beta_up'][ith - start_EOF,:] =  betaup                    
                output['u_v'].append(u_v)                       
                output['RClow'].append((ith-nx)*f.ppf(.05,ith-nx,df2))                       
                output['RCup'].append((ith-nx)*f.ppf(.95,ith-nx,df2))
                output['RCstat'].append(r1_stat)  
                output['ZZ'].append(Z) 
                output['beta_scatter'].append(beta_scatter)
        #output['X_tls'] = (Y + T(corr_ZZ[:,1]))/(X + T(corr_ZZ[:,0]))  
                             
        return output    
               
    def scaling_factors_for_beta(self,SF_para):
        
        output = SF_para['output']
        row_col = [1,1]
        color_list = SF_para['color_list']
        ticks_int = output['EOF']
        xticks = [str(i) for i in ticks_int]
        point_value = np.zeros(shape = (1,len(output['beta_hat'])))
        point_value[0,:] = output['beta_hat'][:,0]
        dy = np.zeros(shape = (1,len(output['beta_hat'])))
        dy[0,:] = output['beta_up'][:,0] - output['beta_hat'][:,0]
    
        xx = np.zeros(shape=(point_value.shape[0],point_value.shape[1]))
        for i in range(point_value.shape[0]):
            for j in range(point_value.shape[1]):
                xx[i,j] = i*2+j*10
                
        plt.figure(figsize=(10, 7))
        
        for m in range(1):
        
            map_location = int(str(row_col[0]) + str(row_col[1]) + str(m+1))
            
            plt.subplot(map_location) #确定第一个图的位置
            
            plt.style.use('seaborn-whitegrid')
             
            #fmt是一种控制线条和点的外观的代码格式,语法与plt.plot的缩写代码一样.
            plt.axhline(0,color = 'black',linestyle = '-',linewidth = 2.5)
            plt.axhline(1,color = 'grey',linestyle = '--',linewidth = 2)
            plt.ylabel('scaling factors',fontsize=24) 
            plt.xlabel('Number of EOF patterns retained in the truncation',fontsize=24)
            if 'ZZ' in list(output.keys()):            
                plt.title('TLS Estimates of scaling factors for beta',fontsize = 28) 
            else:
                plt.title('OLS Estimates of scaling factors for beta',fontsize = 28)                 
            
            for i in range(point_value.shape[0]):
                plt.errorbar(xx[i,:],point_value[i,:] ,yerr=dy[i,:],fmt='o',ecolor=color_list[i],color='black',elinewidth=4,capsize=6)
    
            #plt.axvline(17,color = 'black',linestyle = '--',linewidth = 2.5)
            #plt.axvline(27,color = 'grey',linestyle = '--',linewidth = 2.5)
            
            plt.xticks(xx[0,:],xticks,fontsize = 24)
            plt.yticks(fontsize = 20)
            plt.ylim(-2,2)
            #plt.legend(prop={'size':24},frameon = True,ncol=3,loc = 'lower left',shadow = True)
            plt.grid(None)        #删除网格线
            plt.tight_layout()
        #plt.suptitle('Title',fontproperties='SimHei',fontsize = 28)
        if type(self.savefig) == dict:
            path_dir(self.savefig['path'])
            plt.savefig('scaling_factors_for_beta_' + Time().replace(":", "_") + '.' + self.savefig['format'], dpi = self.savefig['dpi']) 
            
    def residual_consistency_test(self,RC_para):

        output = RC_para['output']
        row_col = [1,1]
        label_name = RC_para['label_name']        
        point_value = np.zeros(shape = (1,len(output['RCstat'])))
        down_line = np.zeros(shape = (1,len(output['RCstat'])))
        up_line = np.zeros(shape = (1,len(output['RCstat'])))
        
        for i in range(len(output['RCstat'])):
            point_value[0,i] = 1/float(output['RCstat'][i])
            down_line[0,i] = 1/float(output['RCup'][i])
            up_line[0,i] = 1/float(output['RClow'][i])    
    
        plt.figure(figsize=(10, 7))
        
        boundary = [output['EOF'][0],output['EOF'][-1]]
        
        for i in range(point_value.shape[0]):
        
            map_location = int(str(row_col[0]) + str(row_col[1]) + str(i+1))
                
            plt.subplot(map_location) #确定第一个图的位置
                
            plt.style.use('bmh')
                
            xx = np.linspace(boundary[0],boundary[1],len(output['EOF']))
           
            #fmt是一种控制线条和点的外观的代码格式,语法与plt.plot的缩写代码一样.
            #plt.ylim(1e-3, 8e0)
            plt.semilogy()
            """
            添加对数坐标/semilogx()、semilogy()和loglog(),它们分别绘制X轴为对数坐标、Y轴为对数坐标以及两个轴都为对数坐标时的图表。
            """
            plt.plot(xx,up_line[i],color = 'b',label = label_name[0],linestyle = '--')
            plt.plot(xx,down_line[i],color = 'g',label = label_name[1],linestyle = '--')
            plt.scatter(xx,point_value[i], color = 'black',marker ='o',s=50)
            if 'ZZ' in list(output.keys()):
                plt.title('TLS Residual consistency check',fontproperties='SimHei',fontsize = 28)
            else:
                plt.title('OLS Residual consistency check',fontproperties='SimHei',fontsize = 28)               
            plt.ylabel('Cumulative ratio model/observed variance',fontsize=24) 
            plt.xlabel('Number of EOF patterns retained in the truncation',fontsize=24)         
            plt.yticks(fontsize = 20)
            plt.xticks(fontsize = 20)
            plt.legend(prop={'size':18}, shadow=True) 
            
        plt.suptitle('',fontsize = 28)
        if type(self.savefig) == dict:
            path_dir(self.savefig['path'])
            plt.savefig('residual_consistency_test_' + Time().replace(":", "_") +'.' + self.savefig['format'], dpi = self.savefig['dpi']) 
 
    
    def Signal_to_noise_ratio(self,SN_para):
        
        output = SN_para['output']
        nsig = SN_para['nsig']
        ylim = SN_para['ylim']        
        r_2 = np.zeros(shape = len(output['u']))
        Allen_1999_p5 = np.zeros(shape = (len(output['u']),5))
        
        for i in range(len(output['u'])):
            r_2[i] = (np.dot(np.dot(T(output['û'][i]),output['Cn_inv'][i]),output['û'][i]))   # Ribes_2013 f8 Allen_1999 f18 r^2
            Allen_1999_p5[i,0] = (i + len(nsig))/r_2[i]
            Allen_1999_p5[i,1] = (i + len(nsig))/chi2.ppf(0.05,i + len(nsig))
            Allen_1999_p5[i,2] = (i + len(nsig))/chi2.ppf(0.95,i + len(nsig))  
            Allen_1999_p5[i,3] = 1/f.ppf(0.05,i + len(nsig),self.noise1.shape[0])
            Allen_1999_p5[i,4] = 1/f.ppf(0.95,i + len(nsig),self.noise1.shape[0])  
        
        Allen_1999_p5_inf1 = depict_para_npy('Allen_1999_p5_inf1')
        Allen_1999_p5_inf2 = depict_para_npy('Allen_1999_p5_inf2')
        Allen_1999_p5_inf1[0][0]['title'] = 'OLS S/N optimised, mass weight'
        Allen_1999_p5_inf1[0][0]['y'] = Allen_1999_p5
        Allen_1999_p5_inf1[0][0]['x'] = np.arange(len(nsig) + 1,self.noise1.shape[1] + 1)
        Allen_1999_p5_inf1[0][0]['xlim'] = [len(nsig) + 1,self.noise1.shape[1]]
        Allen_1999_p5_inf1[0][0]['ylim'] = ylim  
        Allen_1999_p5_inf1[0][0]['text_x'] = [len(nsig) + 2,len(nsig) + 3,len(nsig) + 4]
        Allen_1999_p5_inf1[0][0]['text_y'] = [0.2,ylim[1]*0.15,ylim[1]*0.6]
        if type(self.savefig) == dict:        
            self.savefig['name'] = 'Signal_to_noise_ratio_' + Time().replace(":", "_")
        
        line_depict(Allen_1999_p5_inf1,Allen_1999_p5_inf2,self.savefig)
        
        print('KF distribution result')    
        print(Allen_1999_p5[:,1:3])
        print('f distribution result')    
        print(Allen_1999_p5[:,3:])    

    def regression(self,output):
        
        output = output['output']
        Allen_2003_p2 = depict_para_npy('Allen_2003_p2')
        if 'ZZ' in list(output.keys()):
            Allen_2003_p2['title'] = 'TLS with noise in model-simulated signal'
        else:
            Allen_2003_p2['title'] = 'OLS with noise in model-simulated signal'            
        Allen_2003_p2.pop('polyfit')
        Allen_2003_p2['scatter']['x'] = self.X
        Allen_2003_p2['scatter']['color'] = 'green'
        Allen_2003_p2['scatter']['y'] = self.Y
        if 'polyfit' in list(Allen_2003_p2.keys()):
            Allen_2003_p2['polyfit']['linestyle'] =  '-'
        Allen_2003_p2['regression'] = {}
        Allen_2003_p2['regression']['linestyle'] =  ['--','--','--']
        Allen_2003_p2['regression']['beta'] =  [output['beta_hat'][-1],output['beta_low'][-1],output['beta_up'][-1]]
        Allen_2003_p2['regression']['u'] =  [0,0,0]
        Allen_2003_p2['regression']['b'] =  [0,0,0]
        Allen_2003_p2['regression']['linewidth'] =  [2,1,1]
        Allen_2003_p2['regression']['color'] =  ['black','blue','blue']
        if 'X_tilde' in output.keys():
            Allen_2003_p2['recons_scatter'] = {}
            Allen_2003_p2['recons_scatter']['x'] = output['X_tilde']  
            Allen_2003_p2['recons_scatter']['y'] = output['Y_tilde']  
            Allen_2003_p2['recons_scatter']['color'] = 'black'
            Allen_2003_p2['recons_scatter']['marker'] = 'D' 
            Allen_2003_p2['recons_scatter']['alpha'] = 0.8    
            Allen_2003_p2['recons_scatter']['linewidth'] = 1  
            annote_x = np.min(self.X)
            for i in range(len(self.X)):
                if self.X[i,0] == annote_x:
                    annote_y = self.Y[i,0]
            arrow_x = np.min(output['X_tilde'])
            for i in range(len(output['X_tilde'])):
                if output['X_tilde'][i,0] == arrow_x:
                    arrow_y = output['Y_tilde'][i,0] 
            Allen_2003_p2['arrow'] = {}
            Allen_2003_p2['arrow']['x'] = [annote_x,annote_y]
            Allen_2003_p2['arrow']['y'] = [arrow_x,arrow_y]
            Allen_2003_p2['arrow']['text'] = ''
        if type(self.savefig) == dict:
            self.savefig['name'] = 'regression_analysis_' + Time().replace(":", "_") 
            
        scatter_depict(Allen_2003_p2,self.savefig)


    def confidence_ellipse(self,CE_para):
    
        OLS_output2_creg = CE_para['output']
        depict_group = CE_para['depict_group']
        xylim = CE_para['xylim']
        s = CE_para['s']
        K = CE_para['K']
        
        n1 = self.Y.shape[0]
        n2 = self.noise1.shape[0]
        m = self.X.shape[1]
        ell_group = {} 
        ori = {}
        EOF = OLS_output2_creg['EOF']
            
        bar_1 = [0,0,0]
        bar_2 = [0,0,0]
        bar_1[0] = OLS_output2_creg['beta_low'][K][0]
        bar_2[0] = OLS_output2_creg['beta_low'][K][1]
        bar_1[1] = OLS_output2_creg['beta_hat'][K][0]
        bar_2[1] = OLS_output2_creg['beta_hat'][K][1]
        bar_1[2] = OLS_output2_creg['beta_up'][K][0]
        bar_2[2] = OLS_output2_creg['beta_up'][K][1]
        
        if 'cross_bar' in depict_group:
            cross_beta = np.array([bar_1,bar_2])     
        else:
            cross_beta = np.array([bar_1[:2],bar_2[:2]])
        
        if xylim == 0:
            xlim = [OLS_output2_creg['beta_hat'][K][0]-(bar_1[2] - bar_1[0]), OLS_output2_creg['beta_hat'][K][0]+(bar_1[2] - bar_1[0])]
            ylim = [OLS_output2_creg['beta_hat'][K][1]-(bar_2[2] - bar_2[0]), OLS_output2_creg['beta_hat'][K][1]+(bar_2[2] - bar_2[0])]  
        else:
            xlim = xylim[0]
            ylim = xylim[1]          
        x1 = np.linspace(OLS_output2_creg['beta_hat'][K][0]-2, OLS_output2_creg['beta_hat'][K][0]+2, 400) 
        y1 = np.linspace(OLS_output2_creg['beta_hat'][K][1]-2, OLS_output2_creg['beta_hat'][K][1]+2, 400)
        
        if 'beta2_max' in depict_group or 'beta2_max_95' in depict_group or 'F_confidence_95' in depict_group or 'F_confidence_90' in depict_group \
            or 'chi2_confidence_95' in depict_group or 'chi2_confidence_90' in depict_group or 'r2_beta_r2_min_95' in depict_group or \
            'r2_beta_r2_min_90' in depict_group:
        
            tbeta = T(OLS_output2_creg['beta_hat'][K])
            vbeta = OLS_output2_creg['vbeta'][K]
            p_vbeta = OLS_output2_creg['p_vbeta'][K] 
            if 'beta2_max' in depict_group:
                beta2_max = OLS_output2_creg['beta2_max'][K]
            if 'beta2_max_95' in depict_group: 
                beta2_max_95 = OLS_output2_creg['beta2_max_95'][K]
            if 'r2_beta_r2_min_95' in depict_group or 'r2_beta_r2_min_90' in depict_group:
                r2_beta = OLS_output2_creg['r2_beta'][K]
                
            ref = {}
            if 'beta2_max' in depict_group:
                ref['beta2_max'] = beta2_max # Allen_1999 f15
            if 'beta2_max_95' in depict_group:
                ref['beta2_max_95'] = beta2_max_95
            if 'F_confidence_95' in depict_group:
                ref['F_confidence_95'] = m * f.ppf(0.95,m,n2) # Allen_2003 f13
            if 'F_confidence_90' in depict_group:
                ref['F_confidence_90'] = m * f.ppf(0.90,m,n2)        
            if 'chi2_confidence_95' in depict_group:
                ref['chi2_confidence_95'] = chi2.ppf(0.95,m) # Allen_2003 f10
            if 'chi2_confidence_90' in depict_group:
                ref['chi2_confidence_90'] = chi2.ppf(0.90,m) 
            if 'r2_beta_r2_min_95' in depict_group:
                ref['r2_beta_r2_min_95'] = r2_beta - chi2.ppf(0.95,n1 - m) # Allen_2003 f9
            if 'r2_beta_r2_min_90' in depict_group:
                ref['r2_beta_r2_min_90'] = r2_beta - chi2.ppf(0.90,n1 - m) 
        
            print(ref)        
            
            for i_ref in list(ref.keys()):
                ell_group[i_ref] = []
                for i in x1:
                    for j in y1:
                        beta = T(np.array([i,j]))
                        beta2_max_d_LHS = multi_dot(T(tbeta-beta),np.linalg.inv(p_vbeta),tbeta-beta)
                        beta2_max_LHS = multi_dot(T(tbeta-beta),np.linalg.inv(vbeta),tbeta-beta)
                        if 'chi2_confidence_95' in depict_group or 'chi2_confidence_90' in depict_group or 'r2_beta_r2_min_95' in depict_group or 'r2_beta_r2_min_90' in depict_group:
                            ff = beta2_max_d_LHS 
                        if 'beta2_max' in depict_group or 'beta2_max_95' in depict_group or 'F_confidence_95' in depict_group or 'F_confidence_90' in depict_group:                    
                            ff = beta2_max_LHS 
                        if abs(ff - ref[i_ref]) < 0.2:            
                            ell_group[i_ref].append([i,j])
                ell_group[i_ref] = np.array(ell_group[i_ref]) 
                
        if 'tls_confidence' in depict_group:
            ell_group['tls_confidence'] = T(np.array(OLS_output2_creg['beta_scatter'][K])) # Allen_1999 f15    
            
        for i_ref in list(ell_group.keys()):
            x_max = np.max(ell_group[i_ref][:,0])
            x_min = np.min(ell_group[i_ref][:,0])
            for i in range(ell_group[i_ref].shape[0]):
                if ell_group[i_ref][i,0] == x_max:
                    right_point = ell_group[i_ref][i,1]
                if ell_group[i_ref][i,0] == x_min:
                    left_point = ell_group[i_ref][i,1]       
            if right_point > left_point: # angle < 90 for ellipse
                angle_range = range(90)
            else:
                angle_range = range(90,180)
            ori[i_ref] = angle_range        
            
        text_pre_setting = {}
        text_pre_setting['weight'] = 'medium'
        text_pre_setting['style'] = 'normal'
        text_pre_setting['verticalalignment'] = 'top'
        text_pre_setting['horizontalalignment'] = 'left'          
        text_parameters = plt_text_setting().text_setting('black',20,0,0.8,creat_color('coffee'),text_pre_setting)
        if type(self.savefig) == dict:
            self.savefig['name'] = 'confidence_ellipse_' + Time().replace(":", "_")        
        point_to_ellipse(ell_group,cross_beta,s,[10,7],'','','m = ' + str(m) + '  n1 = n2 = ' + str(n2) + ' EOF = ' + str(EOF[K]),xlim,ylim,text_parameters,ori,self.savefig)
    
    def mc_simulation(self,MC_para):
        
        nsig = MC_para['nsig']
        N = MC_para['N']
        method1 = MC_para['method1']
        method2 = MC_para['method2']
        REG = MC_para['REG']        
            
        n1 = self.noise1.shape[0]
        n2 = self.noise2.shape[0]
        if REG:
            Sigma = Creg(self.noise1)
        else:
            noisea = np.vstack((self.noise1,self.noise2))
            Sigma = np.dot(T(noisea),noisea/(n1 + n2))
    
        # X and C(Sigma) are required
        # Check that Sigma is a square matrix
        n = Sigma.shape[0]
        if Sigma.shape[1] != n or Sigma.shape[0] != n:
            print("Error of shape in Sigma") 
        # Number of external forcings considered
        l = self.X.shape[1]
        Sigma12 = sqrt_matrix(Sigma)
        #Sigma13 = matrix_power(Sigma,1/3)
        #Sigma12_inv = matrix_power(Sigma,-1/2)
        # Initial value of beta for the Monte Carlo simulations
        if method1 == 'TLS':
            """
            Z = np.hstack((multi_dot(Sigma12_inv,self.X,np.diag(np.sqrt(nsig))),np.dot(Sigma12_inv,self.Y)))
            u = R_svd(Z)
            v = u['v'][:,u['v'].shape[1]-1]
            beta_0 = R_matrix(-v[:l]*np.sqrt(nsig[:l])/v[l],l,1)
            """
            beta_0 = np.ones([l,1])
            # Monte Carlo simulations
        elif method1 == 'OLS':
            Cf1 = np.linalg.inv(Sigma) 
            Ft = T(multi_dot(np.linalg.inv(multi_dot(T(self.X),Cf1,self.X)),T(self.X),Cf1))
            beta_0 = T(np.dot(T(self.Y),Ft))
            # output stat_H0
        print(beta_0)
        stat_H0 = np.zeros([N,1]) 
            
        for ith in range(N):

            Yt = np.dot(self.X,beta_0)            
                
            # TLS algorithm
            if method1 == 'TLS':
                Y0 = Yt + np.dot(Sigma12,random_array([n,1],0,'norm'))
                X0 = self.X + np.dot(Sigma12,random_array([n,l],0,'norm'))/(np.ones(Yt.shape)*np.sqrt(nsig))
                Xc = (np.ones(Yt.shape)*np.sqrt(nsig)) * X0
                Z1 = np.dot(Sigma12,random_array([n,n1],0,'norm'))
                Z2 = np.dot(Sigma12,random_array([n,n2],0,'norm'))
    
                C1_hat = Creg(T(Z1))
                Cf12 = in_sqrt_matrix(C1_hat)
                Zm = np.dot(Cf12,np.column_stack([Xc,Y0]))
                # M = np.hstack((multi_dot(Cf12,Xc,np.diag(np.sqrt(nsig))),np.dot(C12_in,Ys)))
                #u_u = R_svd(T(Zm))['u']
                u_v = svd(T(Zm))['v']
                u_d = svd(T(Zm))['d']
                d = u_d**2
                nd = len(d)
                vi = u_v[:,nd-1]
                Z2w = T(np.dot(Cf12,Z2))
                if method2 == 'Allen03':
                    stat_H0[ith] = d[nd-1]/multi_dot(vi,np.dot(T(Z2w),Z2w)/n2,T(vi))
                elif method2 == 'ROF': 
                    stat_H0[ith] = d[nd-1]/(np.sum(multi_dot(vi**2,T(Z2w**2)))/n2)
                #print('TLS algorithm - distribution of ' + str(ith + 1) + ' simulations')
                #distribution(0).dis_pro_hist_depict(stat_H0,[0,int(np.max(stat_H0)) + 1],int(np.max(stat_H0)) + 1)
                # OLS algorithm
            elif method1 == 'OLS':

                Sigma_ver = Sigma12
                Y0 = Yt + np.dot(Sigma_ver,random_array([n,1],0,'norm'))
                X0 = self.X + np.dot(Sigma_ver,random_array([n,l],0,'norm'))
                Z1 = np.dot(Sigma_ver,random_array([n,n1],0,'norm')) 
                Z2 = np.dot(Sigma_ver,random_array([n,n2],0,'norm')) 
                if REG:
                    C1_hat = Creg(T(Z1))
                else:
                    C1_hat = np.dot(Z1,T(Z1))/n1
            
                C = C1_hat
                Cf1 = np.linalg.inv(C) 
                Ft = T(multi_dot(np.linalg.inv(multi_dot(T(X0),Cf1,X0)),T(X0),Cf1))
                beta_hat = np.dot(T(Y0),Ft)
                if method2 == 'REG':
                    var_valid = np.dot(Z2,T(Z2))/n2
                    epsilon = Y0 - np.dot(X0,T(beta_hat))
                    stat_H0[ith] = multi_dot(T(epsilon),np.linalg.inv(var_valid),epsilon)
                    #print('OLS algorithm - distribution of ' + str(ith + 1) + ' simulations')
                    #distribution(0).dis_pro_hist_depict(stat_H0,[0,int(np.max(stat_H0)) + 1],int(np.max(stat_H0)) + 1) 
                elif method2 == 'AT99':
                    nmin = np.min(np.array([n,n1,n2]))
                    eof1 = Sym_eigen(C)[1] # EOFs of the control / eigenvectors of Cn^
                    PX = np.dot(T(eof1),X0)
                    PY = np.dot(T(eof1),Y0)  
                    noise2t = np.dot(T(Z2),eof1)        
                    cn2t = np.dot(T(noise2t[:,:nmin]),noise2t[:,:nmin])/n2
                    ev2 = Sym_eigen(cn2t)[0] 
                    eof2 = Sym_eigen(cn2t)[1]                     
                    u = PY - np.dot(PX, beta_hat)
                    tmp = np.zeros(shape = nmin)
                    for j in range(nmin):
                        tmp[j] = np.sum(T(u[:nmin])*eof2[:nmin,j])
                    stat_H0[ith] = np.sum(tmp**2/ev2[:nmin])
                
        if type(self.savefig) == dict:        
            self.savefig['name'] = 'mc_simulation ' + method1 + '_' + method2 +  '_n_' + str(n) + '_n1_' + str(n1) + '_N_' + str(N) + '_' + figname_affix(REG,'REG') 
            path_dir(self.savefig['path'])
            np.save(self.savefig['name'],stat_H0)
        xy_length = [int(n*3),0.15/np.sqrt(n/10)]
        if n > 30:
            bar_width = int(xy_length[0]/30)
        else:
            bar_width = 1
        oo = Fig_7_map(stat_H0,l,n,n2,[0,xy_length[0]],[0,xy_length[1]],self.savefig,bar_width)
        del(oo)
        
        return stat_H0,self.savefig['name']

    def GKE(self,GK_para):
        
        d = GK_para['output']['RCstat'][-1]
        stat_H0 = GK_para['stat_H0']
        
        N = len(stat_H0)
        h = 1.06 * stdev(stat_H0) * N**(-1/5)
        p = stats.norm.cdf(d*np.ones(stat_H0.shape),stat_H0,h*np.ones(stat_H0.shape))
        pv = np.sum(1-p)/N
        
        return pv      
 
def Fig_7_map(stat_H0,l,n,n2,x_range,y_range,savenfig,barwidth):
    
    # Z --- input
    # l --- rank(X)
    # n --- EOF truncation
    # n2 --- ensemble num
    # x_range --- real x range
    # y_range --- y range
    
    temp = distribution(1).KF_depict(x_range[0],x_range[1],n - l) # Ribes_2013 f8
    RHS_f8_Ribes_2013 = temp[0]
    temp = distribution(1).f_depict(x_range[0],x_range[1],n - l,n2) # Ribes_2013 f8
    RHS_f14_Ribes_2013 = distribution(1).distribution_multiple(temp[0],n - l)
    temp = distribution(1).f_depict(x_range[0],x_range[1],n - l,n2 - n + 1) # Ribes_2013 f8
    RHS_f13_Ribes_2013 = distribution(1).distribution_multiple(temp[0],(n - l)*n2/(n2 - n + 1))

    # line 
    line_group = np.zeros(shape = (len(RHS_f13_Ribes_2013),3))
    line_group[:,0] = RHS_f8_Ribes_2013
    line_group[:,1] = RHS_f14_Ribes_2013
    line_group[:,2] = RHS_f13_Ribes_2013
    Fig_7_map_inf1 = depict_para_npy('Ribes_2013_7p_inf1')
    Fig_7_map_inf2 = depict_para_npy('Ribes_2013_7p_inf2')
    Fig_7_map_inf1[0][0]['y'] = line_group
    Fig_7_map_inf1[0][0]['xlim'] = [x_range[0]*10,x_range[1]*10]
    Fig_7_map_inf1[0][0]['ylim'] = y_range
    Fig_7_map_inf1[0][0]['x'] = np.linspace(x_range[0]*10, x_range[1]*10 - 1,x_range[1]*10 - x_range[0]*10)
    Fig_7_map_inf1[0][0]['x_label'] = [str(ss) for ss in np.linspace(int(x_range[0]), int(x_range[1]),7)]
    if type(savenfig) == dict:    
        Fig_7_map_inf1[0][0]['title'] = savenfig['name']
    
    # bar 
    bar_value,bar_xticks,bar_width = distribution(1).dis_pro_hist_depict(stat_H0,[0,int(np.max(stat_H0)) + 1],int((int(np.max(stat_H0)) + 1)/ barwidth))

    Fig_7_map_inf2['x_ticks'] = [int(ss * 10) for ss in np.linspace(int(x_range[0]), int(x_range[1]),7)] 
    Fig_7_map_inf1[0][0]['bar'] = {}
    Fig_7_map_inf1[0][0]['bar']['edgecolor'] = 'black' 
    Fig_7_map_inf1[0][0]['bar']['color'] = 'grey'    
    Fig_7_map_inf1[0][0]['bar']['alpha'] = 0.5   
    Fig_7_map_inf1[0][0]['bar']['linewidth'] = 1   
    Fig_7_map_inf1[0][0]['bar']['value'] = bar_value
    Fig_7_map_inf1[0][0]['bar']['xticks'] = bar_xticks * 10
    Fig_7_map_inf1[0][0]['bar']['width'] = bar_width * 10 

    # plot    
    if type(savenfig) == dict:
        savenfig['name'] = savenfig['name'] + '_' + Time().replace(":", "_")
    line_depict(Fig_7_map_inf1,Fig_7_map_inf2,savenfig)
        
    return Fig_7_map_inf1


def truncation_errorbar(output,row_col,bar_color,ylim,savefig):

    ticks_int = output['EOF']
    xticks = [str(i) for i in ticks_int]
    point_value = np.zeros(shape = (1,len(output['beta_hat'])))
    point_value[0,:] = output['beta_hat'][:,0]
    xx = np.zeros(shape=(point_value.shape[0],point_value.shape[1]))
    for i in range(point_value.shape[0]):
        for j in range(point_value.shape[1]):
            xx[i,j] = i*2+j*10    
    dy = np.zeros(shape = (1,len(output['beta_hat'])))
    dy[0,:]  = output['beta_up'][:,0] - output['beta_hat'][:,0]
    hy = np.zeros(shape = (1,len(output['beta_hat'])))
    hy[0,:]  = np.sqrt(output['beta2_max_95'])

    plt.figure(figsize=(14, 7))
    for m in range(1):

        map_location = int(str(row_col[0]) + str(row_col[1]) + str(m+1))
            
        plt.subplot(map_location) #确定第一个图的位置
            
        plt.style.use('seaborn-whitegrid')
             
        #fmt是一种控制线条和点的外观的代码格式,语法与plt.plot的缩写代码一样.
        plt.axhline(0,color = 'black',linestyle = '-',linewidth = 2.5)
        plt.axhline(1,color = 'grey',linestyle = '--',linewidth = 2)
        plt.axhline(-1,color = 'grey',linestyle = '--',linewidth = 2)
        plt.ylabel('scaling factors',fontsize=24) 
        plt.xlabel('Number of EOF patterns retained in the truncation',fontsize=24)        
        plt.title('Estimates of scaling factors for beta',fontsize = 28) 
        
        for i in range(point_value.shape[0]):
            plt.errorbar(xx[i,:],point_value[i,:] ,yerr=dy[0,:],fmt='none',ecolor=bar_color[i],elinewidth=4,capsize=1)
            plt.scatter(xx[i,:],point_value[i,:],s= 300,marker = 'o', color='', edgecolors = 'black',alpha=0.8,linewidths = 2)
            plt.scatter(xx[i,:],point_value[i,:]+ hy[i,:],s=1000,marker = '_',color='black', alpha=0.8,linewidths = 10)
            plt.scatter(xx[i,:],point_value[i,:]- hy[i,:],s=1000,marker = '_',color='black', alpha=0.8,linewidths = 10)        
        
        plt.xticks(xx[0,:],xticks,fontsize = 24)
        plt.yticks(fontsize = 20)
        plt.ylim(-ylim,ylim)
        #plt.legend(prop={'size':24},frameon = True,ncol=3,loc = 'lower left',shadow = True)
        plt.grid(None)        #删除网格线
        plt.tight_layout()
        if type(savefig) == dict:
            path_dir(savefig['path'])
            plt.savefig('truncation_errorbar_' + Time().replace(":", "_") + '.' + savefig['format'], dpi = savefig['dpi']) 




 
print('\nPlease prepare your data(.csv) as follows: \n')
print('Observation: 1 x n matrix')
print('Signal: nx x n matrix, nx is number of signals')
print('Noise 1 / Noise 2: n2 x n matrix, each row of this matrix is one piece of noise \n')

ss = raw_input('I have prepared my data 1[yes]/0[No] ','int')
if not ss:
    sys.exit(-1)
    
  
sig = 1
while sig:
    sig = 0
    loc = str(input('Please input your address: '))
    try:
        if '/' in loc:
            data_path = path_dir([r'' + loc + '/data'])# I:\new\tmh_exe_1  
            systype = 'os'
        elif "\\" in loc:
            data_path = path_dir([r'' + loc + '\data'])# I:\new\tmh_exe_1 
            systype = 'win'
        elif '\\' not in loc and '/' not in loc:
            raise Exception('Please input your address again: ')
    except:
        print('Location not found')
        sig = 1
        
        
print('Files location: ')         
print(data_path)       

cir = 1
while cir:  
    sim_Z = {}
    if ss:
        sig = 1
        while sig:
            try:
                pattern = raw_input('single-pattern or two-pattern? please print 1[single-pattern] or 2[two-pattern]: ','int')
                observation_name = input('please print filename of observation (with file extension(.csv/.dat)): ')
                sim_Z['Y'] = file_read(observation_name,[data_path]).T  
                signal_name = input('please print filename of signal (with file extension(.csv/.dat)): ')
                sim_Z['X'] = file_read(signal_name,[data_path]).T
                noise_1_name = input('please print filename of noise 1 (with file extension(.csv/.dat)): ')
                sim_Z['noise1'] = file_read(noise_1_name,[data_path])
                noise_2_name = input('please print filename of noise 2 (with file extension(.csv/.dat)): ')
                sim_Z['noise2'] = file_read(noise_2_name,[data_path])
                nsig = np.zeros(pattern)
                for i in range(pattern):
                    nsig[i] = int(input('please print the ' + str(i+1) + ' st/nd value of nsig (vector, length of nx, ensemble size for each signal): '))
                sig -= 1
            except:
                print('file not found')
        
        nt = len(sim_Z['Y'])
        ns = 1
        u = redop(nt,ns,nt)
        sim_Zr = redEOF(sim_Z,u,nt,ns)
        variety_Z(sim_Zr,['g','orange'],5,0.6)
        print(sim_Zr)
        
    #sim_Zr,nsig = Z_creat(['R','R'],[187,11,1],red = 1)
    #sim_Zr,nsig = Z_creat(['C','C'],0,red = 1)
    #sim_Zr,nsig = Z_creat(['A','A'],0,red = 1)
    #sim_Zr,nsig = Z_creat(['Cr','Cr'],0,red = 1)
    
    ols_para = {}
    ols_para['plev'] = 0.95
    ols_para['nsig'] = nsig
    ols_para['REG'] = 1
    ols_para['tilde'] = 1
    ols_para['All_EOF'] = 1
    
    tls_para = {}
    tls_para['plev'] = 0.9
    tls_para['nsig'] = nsig
    tls_para['REG'] = 1
    tls_para['tilde'] = 1
    tls_para['All_EOF'] = 1
    
    if systype == 'os':
        savefig = {'path':[r'' + loc + ''],\
                   'name':'','dpi':300,'format':'jpg'}  
    elif systype == 'win':
        savefig = {'path':[r'' + loc + ''],\
                   'name':'','dpi':300,'format':'jpg'}     
    
    Output = {}
    
    Output['OLS_output'] = tls_ols(sim_Zr,savefig).ols(ols_para)
    Output['TLS_output'] = tls_ols(sim_Zr,savefig).tls(tls_para)
    
    SF_para = {}
    RC_para = {}
    MC_para = {}
    SN_para = {}
    GK_para = {}
    RE_para = {}
    CE_para = {}
    
    for i_key in Output.keys():
        SF_para['output'],RC_para['output'],MC_para['output'],SN_para['output'],GK_para['output'],RE_para['output'],CE_para['output'] = Output[i_key],Output[i_key],Output[i_key],Output[i_key],Output[i_key],Output[i_key],Output[i_key]
        MC_para['nsig'],SN_para['nsig'] = nsig,nsig
        SF_para['color_list'] = ['red']
        RC_para['label_name'] = ['95%','5%']
        SN_para['ylim'] = [0,10]
        CE_para['xylim'] = 0
        if i_key == 'OLS_output':
            CE_para['s'] = 2
            CE_para['depict_group'] = ['beta2_max','beta2_max_95','r2_beta_r2_min_95','r2_beta_r2_min_90','chi2_confidence_95','chi2_confidence_90','cross_bar']
        elif i_key == 'TLS_output':
            CE_para['depict_group'] = ['tls_confidence','cross_bar']
            CE_para['s'] = 4.605
        CE_para['K'] = -1
        #GK_para['stat_H0'] = stat_H0
        
        if pattern == 1:
            tls_ols(sim_Zr,savefig).scaling_factors_for_beta(SF_para)
            tls_ols(sim_Zr,savefig).residual_consistency_test(RC_para)
            tls_ols(sim_Zr,savefig).regression(RE_para)
            if i_key == 'OLS_output':
                tls_ols(sim_Zr,savefig).Signal_to_noise_ratio(SN_para)
        if pattern == 2:
            tls_ols(sim_Zr,savefig).confidence_ellipse(CE_para)
    
    for i_key in list(Output.keys()):
        xls_file = xlwt.Workbook()
        for i_th in range(len(list(Output[i_key].keys()))):
            if i_th == 0:
                time_affix = Time().replace(":", "_")
                write_field_xls(r'' + loc + '\\'+ i_key + ' ' + time_affix +'.xls', 'EOFs', Output[i_key][list(Output[i_key].keys())[i_th]])
            else:
                write_sheet_xls(r'' + loc + '\\'+ i_key + ' ' + time_affix +'.xls', list(Output[i_key].keys())[i_th], Output[i_key][list(Output[i_key].keys())[i_th]])
        
    cir = int(input('print 1[continue] or 0[end]: '))
    
End = input('press any key to end')
'''
MC_para['N'] = 1000
MC_para['method1'] = 'TLS'
MC_para['method2'] = 'Allen03'
MC_para['REG'] = 0
stat_H0,stat_H0_npy_name = tls_ols(sim_Zr,savefig).mc_simulation(MC_para)


CE_para['output'] = TLS_output
CE_para['depict_group'] = ['tls_confidence','cross_bar']
CE_para['xylim'] = 0
CE_para['s'] = 4.605 #根据置信区间查卡方概率表 95% 5.991 99% 9.21 90% 4.605
CE_para['K'] = -1

tls_ols(sim_Zr,savefig).confidence_ellipse(CE_para)
'''

"""
def location_dir():
    systype = 0
    while not systype:
        loc = str(input('input location:'))
        try:
            os.chdir(r'' + loc)
            sys.path.append(r'' + loc) # os
            systype = 1
        except:
            print('Error : No such file or directory: ' + loc)
    print('location: ' + loc)
    if '/' in loc:
        systype = 'os'
    elif "\\" in loc or ':' in loc:
        systype = 'win'    
    print('systype: ' + systype)
    return loc,systype
    
loc,systype = location_dir()
"""

