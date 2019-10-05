import numpy as np
import pandas as pd
import pyodbc
import time
import pickle
import operator
from operator import itemgetter
from joblib import Parallel, delayed

from sklearn import linear_model
import statsmodels.formula.api as sm
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine

import psycopg2
from sklearn.utils import shuffle
from sqlalchemy.pool import NullPool

import sql
from sklearn import feature_selection

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

def data_generation(num_control, num_treated, num_cov):
    
    # the data generating function that we will use. include second order information
    
    xc = np.random.binomial(1, 0.5, size=(num_control, num_cov))   # data for conum_treatedrol group
    xt = np.random.binomial(1, 0.5, size=(num_treated, num_cov))   # data for treatmenum_treated group
        
    errors1 = np.random.normal(0, 0.1, size=num_control)    # some noise
    errors2 = np.random.normal(0, 0.1, size=num_treated)    # some noise
    
    dense_bs = [ 20.*((4./5)**(i+1)) for i in range(num_cov) ]

    yc = np.dot(xc, np.array(dense_bs)) #+ errors1     # y for conum_treatedrol group 
    catt_c = [0] * num_control

    treatment_eff_coef = np.random.normal( 1.5, 0.15, size=num_cov)
    treatment_effect = np.dot(xt, treatment_eff_coef) 
    
    second = construct_sec_order(xt[:,:5])
    treatment_eff_sec = np.sum(second, axis=1)
    
    yt = np.dot(xt, np.array(dense_bs)) + treatment_effect + treatment_eff_sec #+ errors2    # y for treated group 
    catt_t = treatment_effect + treatment_eff_sec
 
    df1 = pd.DataFrame(xc, columns = range(num_cov))
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(xt, columns = range(num_cov)) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    catt = catt_c + list(catt_t)
    df['true_effect'] = catt

    df = df.reset_index()
    df['index'] = df.index

    return df, df[list(range(num_cov))], df['treated'], df['outcome'], df['true_effect']

if __name__ == '__main__':
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    num_treated = 15000
    num_control = 15000
    num_cov = 18

    df,x,d,r, catt = data_generation(num_control, num_treated, num_cov)
    pickle.dump(df, open("data/df", 'wb'))
    np.savetxt("data/x"+".txt",x,fmt="%d")
    np.savetxt("data/d"+".txt",d,fmt="%d")
    np.savetxt("data/r"+".txt",r,fmt="%.8f")
    np.savetxt("data/catt"+".txt",catt,fmt="%.8f")
    holdout,x_,d_,r_, catt_ = data_generation(num_control, num_treated, num_cov)
    pickle.dump(holdout, open("data/holdout", 'wb'))