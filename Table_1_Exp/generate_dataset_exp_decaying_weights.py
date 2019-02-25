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
    catt_t = treatment_effect
 
    yt = np.dot(xt,np.array(dense_bs))+treatment_effect #+ errors2

    df1 = pd.DataFrame(np.hstack([xc]), 
                       columns=range(num_cov))
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(np.hstack([xt]), 
                       columns=range(num_cov ) ) 
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

    num_treated = 2000
    num_control = 40000
    num_cov = 18

    df,x,d,r, catt = data_generation(num_control, num_treated, num_cov)
    pickle.dump(df, open("data/df", 'wb'))
    np.savetxt("data/x"+".txt",x,fmt="%d")
    np.savetxt("data/d"+".txt",d,fmt="%d")
    np.savetxt("data/r"+".txt",r,fmt="%.8f")
    np.savetxt("data/catt"+".txt",catt,fmt="%.8f")
    holdout,x_,d_,r_, catt_ = data_generation(num_control, num_treated, num_cov)
    pickle.dump(holdout, open("data/holdout", 'wb'))

    df1 = df.iloc[20000:42000,:] 
    df1 = df1.reset_index()
    df1['index'] = df1.index
    df1 = df1.iloc[:,1:]

    df2 = df.iloc[30000:42000,:] 
    df2 = df2.reset_index()
    df2['index'] = df2.index
    df2 = df2.iloc[:,1:]

    holdout1 = holdout.iloc[20000:42000,:] 
    holdout1 = holdout1.reset_index()
    holdout1['index'] = holdout1.index
    holdout1 = holdout1.iloc[:,1:]

    holdout2 = holdout.iloc[30000:42000,:]
    holdout2 = holdout2.reset_index()
    holdout2['index'] = holdout2.index
    holdout2 = holdout2.iloc[:,1:]
    
    pickle.dump(df1, open("data/df1", 'wb'))
    pickle.dump(df2, open("data/df2", 'wb'))
    pickle.dump(holdout1, open("data/holdout1", 'wb'))
    pickle.dump(holdout2, open("data/holdout2", 'wb'))
    
    print(df)
    np.savetxt("data/x_1"+".txt",df1.iloc[:,1:19],fmt="%d")
    np.savetxt("data/d_1"+".txt",df1.iloc[:,20],fmt="%d")
    np.savetxt("data/r_1"+".txt",df1.iloc[:,19],fmt="%.8f")
    np.savetxt("data/catt_1"+".txt",df1.iloc[:,22],fmt="%.8f")
    np.savetxt("data/x_2"+".txt",df2.iloc[:,1:19],fmt="%d")
    np.savetxt("data/d_2"+".txt",df2.iloc[:,20],fmt="%d")
    np.savetxt("data/r_2"+".txt",df2.iloc[:,19],fmt="%.8f")
    np.savetxt("data/catt_2"+".txt",df2.iloc[:,22],fmt="%.8f")
