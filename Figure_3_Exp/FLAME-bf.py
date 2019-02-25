import numpy as np
import pandas as pd
import pickle
import time
import itertools
import operator
import matplotlib
matplotlib.rcParams.update({'font.size': 17.5})

import matplotlib.pyplot as plt

matplotlib.rc('axes.formatter', useoffset=False)

import operator
import sys
import os.path
sys.path.append( os.path.abspath(os.path.join( os.path.dirname('..') , os.path.pardir )) )

import numpy as np
import pandas as pd
import pickle
import time
import itertools
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from operator import itemgetter
import operator
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sqlalchemy import create_engine
from mpl_toolkits.mplot3d import Axes3D
import warnings; warnings.simplefilter('ignore')
from decimal import *
import random
from itertools import combinations
import re
from sklearn import linear_model
import statsmodels.formula.api as sm
from statsmodels.stats import anova
import pylab as pl
from multiprocessing import Pool
from functools import partial
import warnings
import pysal
from pysal.spreg.twosls import TSLS
import random

def construct_sec_order(arr):
    
    # an intermediate data generation function used for generating second order information
    
    second_order_feature = []
    num_cov_sec = len(arr[0])
    for a in arr:
        tmp = []
        for i in range(num_cov_sec):
            for j in range(i+1, num_cov_sec):
                tmp.append( a[i] * a[j] )
        second_order_feature.append(tmp)
        
    return np.array(second_order_feature)


# In[51]:

def data_generation(num_control, num_treated, num_cov_dense, 
                            num_covs_unimportant, control_m = 0.1,
                            treated_m = 0.9):
    
    ''' the data generating function that we will use; 
        includes second order information '''
    

    # generate data for control group 
    xc = np.random.binomial(1, 0.5, size=(num_control, num_cov_dense)) #bernouilli
    
    # generate data for treated group 
    xt = np.random.binomial(1, 0.5, size=(num_treated, num_cov_dense))   #bernouilli
     
    
    errors1 = np.random.normal(0, 0.1, size=num_control)    # some noise
    
    errors2 = np.random.normal(0, 0.1, size=num_treated)    # some noise
    
    dense_bs_sign = np.random.choice([-1,1], num_cov_dense) 
    
    dense_bs = [ np.random.normal(s * 10, 1) for s in dense_bs_sign ]   #alpha in the paper

    # y for control group 
    yc = np.dot(xc, np.array(dense_bs)) #+ errors1     
       
    # y for treated group 
    treatment_eff_coef = np.random.normal( 1.5, 0.15, size=num_cov_dense) #beta
    treatment_effect = np.dot(xt, treatment_eff_coef) 
    
    second = construct_sec_order(xt[:,:(num_covs_unimportant -1)])
    treatment_eff_sec = np.sum(second, axis=1)
    
    yt = np.dot(xt,np.array(dense_bs))+treatment_effect+treatment_eff_sec 
                                      # + errors2    

    # generate unimportant covariates for control group
    xc2 = np.random.binomial(1, control_m, size=(num_control,
                                                 num_covs_unimportant))  
    
    # generate unimportant covariates for treated group
    xt2 = np.random.binomial(1, treated_m, size=(num_treated,
                                                 num_covs_unimportant))   
        
    df1 = pd.DataFrame(np.hstack([xc, xc2]), 
                       columns=range(num_cov_dense + num_covs_unimportant))
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(np.hstack([xt, xt2]), 
                       columns=range(num_cov_dense + num_covs_unimportant)) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df, dense_bs, treatment_eff_coef

def compare_rows(treatment,control,covs):
    res = []
    for i in covs :
        if treatment[i] == control[i]:
            res.append(1)
        else:
            res.append(0)      
    return res

def run_bf(df,covs, w):
    s_start = time.time()
    treatments = df[df['treated']==1]
    controls = df[df['treated']==0]
    
    res = pd.DataFrame()
    cates = []
    #run over all treatments
    for i in range(len(treatments)):
        
        #get the treatment unit i
        cur_treatment = treatments.iloc[i]
        
        w_t = []
        for j in range(len(controls)):
            
            #get current control unit
            cur_control = controls.iloc[j]
            
            # find the v_tc
            v_tc = np.array(compare_rows(cur_treatment, cur_control, covs)  )
            
            #print(v_tc, w)
            w_tc = np.dot(v_tc,w)
            w_t.append((cur_control,w_tc))
            
        #now get the controls with the largest w_tc
        controls_c = max(w_t, key=itemgetter(1))
        best_control = controls_c[0]
        best_control = pd.DataFrame(data = best_control.values)
        best_control = best_control.transpose()
        best_control.columns = df.columns
        
        cur_treatment = pd.DataFrame(data = cur_treatment.values)
        cur_treatment = cur_treatment.transpose()
        cur_treatment.columns = df.columns
        
        group_t = pd.concat([cur_treatment,best_control])
        
        get_cate = group_t['outcome'].mean()
        cates.append(get_cate)
        res = pd.concat([res,group_t])
    
    s_end = time.time()
    print("time:", s_end - s_start)   
    return res, cates

if __name__ == '__main__':
    df,dense_bs,_ = data_generation(1500, 1500, 10, 0)
    holdout,_,_ = data_generation(1500, 1500, 10, 0)

    res = run_bf(df, range(10), dense_bs)
