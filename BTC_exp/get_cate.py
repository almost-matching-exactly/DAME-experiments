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
with open('FLAME-gen-early','rb') as f:
    data = pickle.load(f) 
match_list_col = data

#print(match_list_col)

cates = {}
for match in match_list_col: 
    if match is None:
        continue
    for key, group in match.iterrows():
        index_list = group["index"]
        mean = group["mean"]
        for idx in index_list:
            cates[idx] = mean
cates_col = [None] * 382

for key in cates:
    cates_col[int(key) - 1] = cates[key]


f = open("SVM.txt", "r+")
predicts = [float(line.replace("\n","")) for line in f.readlines()]
f.close()

def get_keys_values(dict):
    return dict.keys(), dict.values()

neg = {}
neu = {}
pos = {}

for i in range(len(cates_col)):
    key = cates_col[i]
    if cates_col[i] is None:
        continue
    elif predicts[i] == -1:
        neg[key] = neg[key] + 1 if key in neg else 1
    elif predicts[i] == 1:
        pos[key] = pos[key] + 1 if key in pos else 1  
    else:
        neu[key] = neu[key] + 1 if key in neu else 1  

x1, y1 = get_keys_values(neg)    
x2, y2 = get_keys_values(neu)
x3, y3 = get_keys_values(pos)

#print(min([predicts[i] for i in idx_0]))
#print(max([predicts[i] for i in idx_0]))

plt.scatter(x1, y1, color = 'blue', label = 'Negative Effect by SVM')
plt.scatter(x2, y2, color = 'green', label = 'Neutral Effect by SVM')
plt.scatter(x3, y3, color = 'red', label = 'Positive Effect by SVM')
plt.legend()
plt.xlabel("Estimated CATE by DAME")
plt.ylabel("Number of Units")
plt.savefig("com_cates.png")

