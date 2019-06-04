# coding: utf-8

# In[46]:

import numpy as np
import pandas as pd
#import pyodbc
import pickle
import time
import itertools
from joblib import Parallel, delayed

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
import matplotlib.pyplot as plt
from decimal import Decimal
import warnings

"""
with open("data/df2",'rb') as f:
    df = pickle.load(f)  

x = list(df[df["treated"] == 1]["true_effect"])
"""

with open("res/DAME-x-2",'rb') as f:
    x = pickle.load(f) 

with open("res/DAME-y-2",'rb') as f:
    y = pickle.load(f) 

"""
f = open("data/ps_out.txt", "r+")
y = [float(line.replace("\n","")) for line in f.readlines()]
f.close()
"""

print(len(x))
print(len(y))
print(MSE(x,y))
