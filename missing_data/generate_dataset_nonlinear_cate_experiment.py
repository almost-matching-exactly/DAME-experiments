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
from scipy.stats import random_correlation
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

def data_generation(num_control, num_treated, num_covs):
    
    mean = [0] * num_covs
    eigenvalues = num_covs * np.random.dirichlet(np.ones(num_covs),size=1)[0]
    cov = random_correlation.rvs(eigenvalues)

    X = []
    for i in range(num_treated + num_control):
        z = list(np.random.multivariate_normal(mean,cov))
        x = [1 if elem > 0 else 0 for elem in z]
        X.append(x)

    
    xc = np.array(X[:num_control])
    xt = np.array(X[num_control:])
    #print(X[15000:])
    #print(xt)

    errors1 = np.random.normal(0, 0.1, size=num_control)    # some noise
    errors2 = np.random.normal(0, 0.1, size=num_treated)    # some noise
    
    dense_bs_sign = np.random.choice([-1,1], num_covs)
    #dense_bs = [ np.random.normal(dense_bs_sign[i]* (i+2), 1) for i in range(len(dense_bs_sign)) ]
    dense_bs = [ np.random.normal(s * 10, 10) for s in dense_bs_sign ]

    yc = np.dot(xc, np.array(dense_bs)) #+ errors1     # y for conum_treatedrol group 
    catt_c = [0] * num_control

    treatment_eff_coef = np.random.normal( 0.5, 0.15, size=num_covs)
    treatment_effect = np.dot(xt, treatment_eff_coef) 
    
    second = construct_sec_order(xt[:,:5])
    treatment_eff_sec = np.sum(second, axis=1)
    
    yt = np.dot(xt, np.array(dense_bs)) + treatment_effect + treatment_eff_sec #+ errors2    # y for treated group 
    catt_t = treatment_effect + treatment_eff_sec

   
    df1 = pd.DataFrame(xc, columns = range(num_covs))
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(xt, columns = range(num_covs)) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    #df['outcome'] += 2 * np.random.normal(0,5)
    catt = catt_c + list(catt_t)
    df['true_effect'] = catt

    miss = []
    for i in range(num_control + num_treated):
        miss.append(1 if random.randint(1,101) <= 20 else 0)

    miss = pd.DataFrame(np.zeros((num_control + num_treated, num_covs)))
    select = set()
    k = 0
    total_miss_num = (num_control + num_treated) * num_covs * 0.05
    while k < total_miss_num:
        row = random.randint(0,num_control + num_treated - 1)
        col = random.randint(0,num_covs - 1)
        if (row, col) in select:
            continue
        miss.iloc[row,col] = 1
        select.add((row,col))
        k += 1
    
    return df, df[list(range(num_covs))], df['treated'], df['outcome'], miss

if __name__ == '__main__':
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    
    num_treated = 30000
    num_control = 30000
    num_covs = 15
    df,x,d,r, miss = data_generation(num_control, num_treated, num_covs)
    pickle.dump(df, open("data/df2", 'wb'))
    np.savetxt("data/x2"+".txt",x,fmt="%d")
    np.savetxt("data/d2"+".txt",d,fmt="%d")
    np.savetxt("data/r2"+".txt",r,fmt="%.8f")
    holdout,x_,d_,r_, miss_ = data_generation(num_control, num_treated, num_covs)
    pickle.dump(holdout, open("data/holdout2", 'wb'))
    pickle.dump(miss, open("data/miss2", 'wb'))
    

    """
    with open("data/df1",'rb') as f:
        data = pickle.load(f) 
    data = data.reset_index()
    data['index'] = data.index
    data.to_csv("data/df1.csv")

    with open("data/miss1",'rb') as f:
        data = pickle.load(f) 
    data = data.reset_index()
    data['index'] = data.index
    data.to_csv("data/miss1.csv")
    """