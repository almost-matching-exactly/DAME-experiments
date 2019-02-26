import pyodbc
import time
import pickle
import operator
from operator import itemgetter
from joblib import Parallel, delayed
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
import psycopg2
from sklearn.utils import shuffle
import sql
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from decimal import *
import re
from functools import partial
from sortedcontainers import SortedDict

#get matching result for collapsing FLAME    
with open('res/FLAME-col-result','rb') as f:
    data = pickle.load(f) 
match_list_col = data

#print(match_list_col)

cates = {}
idx_0 = []
cates_num = {}
pos = []
neg = []
for match in match_list_col: 
    if match is None:
        continue
    for group in match:
        index_list = group[2]
        mean = group[0]
        for idx in index_list:
            if mean > 0.0:
                pos.append(idx)
            elif mean < 0.0:
                neg.append(idx)
            cates[idx] = mean

        cates_num[mean] = cates_num[mean] + len(index_list) if mean in cates_num else len(index_list)
cates_col = [None] * 382



for key in cates:
    cates_col[int(key) - 1] = cates[key]

for i in cates_col:
    if i is None:
        print(i)

df = pd.read_csv('data/MyBTCData_R2.csv', index_col=0, parse_dates=True)
df = df.rename(columns={'BTC': 'treated', 'outcome_matrix$ANY_NDRU': 'outcome'})
df_treated = df.loc[:,'treated']
df = df.drop('treated',1)
df_outcome = df.loc[:,'outcome']
df = df.drop('outcome',1)
shape = df.shape 
row_num = shape[0]
col_num = shape[1]
df.columns = np.arange(col_num)
df.columns = df.columns.astype(str) 

#merge covariates and outcomes
df = pd.concat([df, df_treated, df_outcome], axis=1)  
for label in df:
    if label == 'outcome':
        df[label][df[label] == 0] = -1
df['outcome'] = df['outcome'].astype('object')        
df['matched'] = 0

df = df.reset_index()
df['index'] = df.index

pos_indicator = [0] * 382
for i in range(382):
    if i in pos:
        pos_indicator[i] = 1
    else:
        pos_indicator[i] = 0


neg_indicator = [0] * 382
for i in range(382):
    if i in neg:
        neg_indicator[i] = 1
    else:
        neg_indicator[i] = 0

#print(pos_indicator.count(1))
for i in range(10):
    x = df[str(i)]
    print(x.tolist(), np.corrcoef(x, pos_indicator))
    