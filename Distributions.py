#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 22:34:01 2020

@author: tang
"""
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore','(?s).*MATPLOTLIBDATA.*',category = UserWarning)

import pandas as pd
import numpy as np
import os
#import sympy
from sympy import *
from scipy.stats import t
from scipy.stats import f
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import math
import rpy2
#import rpy2.robjects
#from rpy2 import robjects
import os
import pandas as pd
from scipy.stats import wishart, chi2
#from scipy import integrate 
# 与 sympy 冲突
from sympy.abc import x,y,a,b,c
import seaborn as sns
import scipy.stats as stats


class distribution():
        
    def __init__(self,returnornot):
        self.returnornot = returnornot

    def KF_distribution(self,x,n):
        
        if x > 0:
    
            return 1/(2**(n/2) * self.F_function(n/2)) * x**(n/2 - 1) * math.e**(-x/2)
        
        else:
            
            return 0
        
    def KF_EX_DX(self,n):
        
        return n,2*n
    
    def F_function(self,z): # Γ(z)＝∫(x^(z-1)*e^(-x))dx (0,+∞)
    
        x = symbols('x')
        m = z-1
        fx = x**m*exp(-x) # 空格会出现逗号
        #integrate(fx,(x,0,oo)) 
        
        return integrate(fx,(x,0,oo)) 
    
    def n_distribution(self,x,miu,beta): # stats.norm.pdf(x,miu,beta)
        
        return (1/((math.pi*2)**0.5 * beta)) * math.e**(-((x - miu)**2)/(2*beta**2))
    
    def s_n_distribution(self,x):
        
        return self.n_distribution(x,0,1)

    def t_pdf(self,x,n): # n -> oo , t_pdf -> s_n_distribution
        
        return self.F_function((n+1)/2)/((n*math.pi)**0.5 * self.F_function(n/2)) * (1 + x**2/n)**(-(n+1)/2)

    def t_ppf(self,alpha,n): # n -> oo , t_pdf -> s_n_distribution

        x = symbols('x')        
        m1 = self.F_function((n+1)/2)
        m2 = self.F_function(n/2)
        m_1 = m1/((n*3.14159265)**0.5 * m2)
        m_2 = (n)**(-1)
        m_3 = (-(n+1)*0.5)
        fx = m_1 * (1 + x**2 * m_2)**m_3
        print(m_1)
        print(m_2)
        print(m_3)
        
        #integrate(fx,(x,-oo,alpha)) 
    
        return integrate(fx,(x,-oo,alpha)) 
    
    def t_EX_DX(self,n):
        
        if n > 1:
            
            print('EX:')
        
            return 0
        
        if n > 2:
            
            return 0 , n/(n - 2)            
    
    def F_pdf(self,x,n1,n2):
        
        if x > 0:
        
            return self.F_function((n1 + n2)/2)/(self.F_function((n1)/2) * self.F_function((n2)/2)) * n1**(n1/2) * n2**(n2/2) * x**(n1/2 - 1)/(n1*x + n2)**((n1 + n2)/2)

        else:
            
            return 0
        
    def t_depict(self,start,end,n):
        
        plt.figure(1)
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)        
        x = np.linspace(start,end,(end - start)*10)
        y = np.linspace(start,end,(end - start)*10)
        
        for i in range(len(x)):
            y[i] = self.t_pdf(x[i],n)
        
        plt.sca(ax1)
        plt.plot(x,y)
        plt.fill_between(x,y,alpha = 0.2)

        x1 = np.linspace(start,end,(end - start)*10)
        y1 = np.linspace(start,end,(end - start)*10)
        
        for i in range(len(x1)):
            y1[i] = t.cdf(x1[i],n)
            #y1[i] = self.t_pdf(x[i],n)

        plt.sca(ax2)        
        plt.plot(x1,y1)
        plt.fill_between(x1,y1,alpha = 0.2)    
        
        plt.show()
    
    def f_depict(self,start,end,n1,n2):
        
        plt.figure(1)
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)        
        x = np.linspace(start,end,(end - start)*10)
        y = np.linspace(start,end,(end - start)*10)
        
        for i in range(len(x)):
            y[i] = f.pdf(x[i],n1,n2)
        
        plt.sca(ax1)
        plt.plot(x,y)
        plt.fill_between(x,y,alpha = 0.2)

        x1 = np.linspace(start,end,(end - start)*10)
        y1 = np.linspace(start,end,(end - start)*10)
        
        for i in range(len(x1)):
            y1[i] = f.cdf(x1[i],n1,n2)
            #y1[i] = self.t_pdf(x[i],n)

        plt.sca(ax2)        
        plt.plot(x1,y1)
        plt.fill_between(x1,y1,alpha = 0.2)    
        
        plt.show()
        
        if self.returnornot:
            
            return y,y1

    def n_depict(self,start,end,miu,beta):
        
        plt.figure(1)
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)        
        X = np.linspace(start,end,(end - start)*10)
        Y = np.linspace(start,end,(end - start)*10)
        
        for i in range(len(X)):
            Y[i] = self.n_distribution(X[i],miu,beta)
        
        plt.sca(ax1)
        plt.plot(x,y)
        plt.fill_between(x,y,alpha = 0.2)

        x1 = np.linspace(start,end,(end - start)*10)
        y1 = np.linspace(start,end,(end - start)*10)
        
        for i in range(len(x1)):
            y1[i] = stats.norm.cdf(x[i],miu,beta)
            #y1[i] = self.t_pdf(x[i],n)

        plt.sca(ax2)        
        plt.plot(x1,y1)
        plt.fill_between(x1,y1,alpha = 0.2)    
        
        plt.show()
        
        if self.returnornot:
            
            return Y,y1

        
    def KF_depict(self,start,end,n):
        
        plt.figure(1)
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)        
        x = np.linspace(start,end,(end - start)*10)
        y = np.linspace(start,end,(end - start)*10)
        
        for i in range(len(x)):
            y[i] = self.KF_distribution(x[i],n)
        
        plt.sca(ax1)
        plt.plot(x,y)
        plt.fill_between(x,y,alpha = 0.2)

        x1 = np.linspace(start,end,(end - start)*10)
        y1 = np.linspace(start,end,(end - start)*10)
        
        for i in range(len(x1)):
            y1[i] = stats.chi2.cdf(x[i],n)
            #y1[i] = self.t_pdf(x[i],n)

        plt.sca(ax2)        
        plt.plot(x1,y1)
        plt.fill_between(x1,y1,alpha = 0.2)    
        
        plt.show()

        if self.returnornot:
            
            return y,y1

    def dis_hist_depict(self,u,range_group,bins_num):

        if type(u) == list:
            u = np.array(u)
            
        plt.hist(x = u, # 指定绘图数据
            range=(range_group[0],range_group[1]),bins = bins_num, # 指定直方图中条块的个数
            color = 'steelblue', # 指定直方图的填充色
            edgecolor = 'black', # 指定直方图的边框色
            align = 'mid',
            #label = str(round(np.std(u),2))
            )       
        
    def dis_pro_hist_depict(self,u,range_group,bins_num):
        
        if type(u) == list:
            u = np.array(u)
                
        n,bins,patches = plt.hist(u,bins_num, # 指定绘图数据
            (range_group[0],range_group[1]),density = True, # 指定直方图中条块的个数
            color = 'steelblue', # 指定直方图的填充色
            edgecolor = 'black', # 指定直方图的边框色
            align = 'mid',
            #label = str(round(np.std(u),2))
            ) 
        
        plt.show()
        
        bins_n = np.zeros(shape = len(bins)-1)
        for i in range(len(bins)-1):
            bins_n[i] = (bins[i]+bins[i + 1])/2
            
        pro_n = np.zeros(shape = n.shape)
        for i in range(len(n)):
            pro_n[i] = n[i]/np.sum(n)
            
        plt.bar(bins_n,pro_n,width = (range_group[1] - range_group[0])/bins_num , edgecolor='black')
        plt.show()
        
        if self.returnornot:
            return pro_n,bins_n,(range_group[1] - range_group[0])/bins_num 

    def probability_distribution(self,data):
        
        sns.kdeplot(data, shade = True)
                
    def n_features(self,mu,sigma):
        
        print('mean','var','std')
        
        return stats.norm.mean(mu,sigma),stats.norm.var(mu,sigma),stats.norm.std(mu,sigma)
    
    def KF_features(self,n):
        
        print('mean','var','std')        
        
        return stats.chi2.mean(n),stats.chi2.var(n),stats.chi2.std(n)
    
    def F_features(self,m,n):
        
        print('mean','var','std')    
        
        return stats.f.mean(m,n),stats.f.var(m,n),stats.f.std(m,n)

    def t_features(self,n):
        
        print('mean','var','std')    
        
        return stats.t.mean(n),stats.t.var(n),stats.t.std(n)
       
    def n_percentile(self,alpha,mu,sigma):
        
        return stats.norm.isf(1 - alpha, mu, sigma), stats.norm.ppf(alpha, mu, sigma)
    
    def KF_percentile(self,alpha,n):
        
        return stats.chi2.isf(1 - alpha, n), stats.chi2.ppf(alpha, n)

    def F_percentile(self,alpha,m,n):
        
        return stats.f.isf(1 - alpha, m, n), stats.f.ppf(alpha, m, n)
    
    def t_percentile(self,alpha,n):
        
        return stats.t.isf(1 - alpha, n), stats.t.ppf(alpha, n)    

    def creat_n_dis(self,num):
        
        return np.random.randn(num)

    def distribution_multiple(self,y,num):
        
        num = int(num)
        out = np.zeros(shape = len(y))
        if y[0] == 0:
            out[0] =  0.00000001 # 拟合 起始终止值不能为空值
        else:
            out[0] = y[0]
        for i in range(int(len(y)/num)-1):
            out[num*i + num] = y[i + 1]
        # 插值
        list_33f = pd.DataFrame(out)            
        list_33f = list_33f.mask(list_33f.applymap(str).eq('0.0'))#0不能保持三位小数            
        list_33i = list_33f.interpolate(method = 'cubic')            
        list_33v = list_33i.values            
        list_33v = (list_33v.T)[0]    
        list_33v = list_33v/num

        if str(out[-1]) == 'nan':
            out[-1] =  0.00000001 # 拟合 起始终止值不能为空值 
        # 修补
        list_33f = pd.DataFrame(list_33v)            
        list_33f = list_33f.mask(list_33f.applymap(str).eq('0.0'))#0不能保持三位小数            
        list_33i = list_33f.interpolate()            
        list_33v = list_33i.values            
        list_33v = (list_33v.T)[0]    
        
        return list_33v
    

        
