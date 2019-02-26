import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
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
for match in match_list_col: 
    if match is None:
        continue
    for group in match:
        index_list = group[2]
        mean = group[0]
        for idx in index_list:
            if mean <= 0.25 and mean >= -0.25:
                idx_0.append(idx)
            cates[idx] = mean

        cates_num[mean] = cates_num[mean] + len(index_list) if mean in cates_num else len(index_list)
cates_col = [None] * 382

for key in cates:
    cates_col[int(key) - 1] = cates[key]

x = [float(elem) for elem in cates_col if elem is not None]

print(x)
print(type(x))

num_bins = 20
# the histogram of the data
n, bins, patches = plt.hist(x, facecolor='blue', alpha=0.5)
 
# add a 'best fit' line
#y = mlab.normpdf(bins, mu, sigma)
#plt.plot(bins, y, 'r--')
plt.xlabel('Predicted CATE')
plt.ylabel('Probability')
plt.title(r'Histogram of Predicted CATE By DAME')
 
# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.savefig("cate_hist.png")