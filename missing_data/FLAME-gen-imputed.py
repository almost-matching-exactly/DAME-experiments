
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

def get_data(path):
    with open(path,'rb') as f:
        data = pickle.load(f) 
    data = data.reset_index()
    data['index'] = data.index 
    return data

def match(df, covs, covs_max_list, treatment_indicator_col = 'treated', match_indicator_col = 'matched'):
    covs = list(covs)
    covs_max_list = list(covs_max_list)
    
    # this function takes a dataframe, a set of covariates to match on, 
    # the treatment indicator column and the matched indicator column.
    # it returns the array indicating whether each unit is matched (the first return value), 
    # and a list of indices for the matched units (the second return value)

    arr_slice_wo_t = df[covs].values # the covariates values as a matrix
    arr_slice_w_t = df[ covs + [treatment_indicator_col] ].values # the covariate values together with the treatment indicator as a matrix
        
    lidx_wo_t = np.dot( arr_slice_wo_t, np.array([ covs_max_list[i]**(len(covs_max_list) - 1 - i) for i in range(len(covs_max_list))]) ) # matrix multiplication, get a unique number for each unit
    lidx_w_t = np.dot( arr_slice_w_t, np.array([ covs_max_list[i]**(len(covs_max_list) - i) for i in range(len(covs_max_list))] +                                               [1]
                                              ) ) # matrix multiplication, get a unique number for each unit with treatment indicator
        
    _, unqtags_wo_t, counts_wo_t = np.unique(lidx_wo_t, return_inverse=True, return_counts=True) # count how many times each number appears
    _, unqtags_w_t, counts_w_t = np.unique(lidx_w_t, return_inverse=True, return_counts=True) # count how many times each number appears (with treatment indicator)
    
    match_indicator = ~(counts_w_t[unqtags_w_t] == counts_wo_t[unqtags_wo_t]) # a unit is matched if and only if the counts don't agree
        
    return match_indicator, lidx_wo_t[match_indicator]

# In[54]:

# match_quality, the larger the better
def match_quality(df, holdout, covs_subset, match_indicator, ridge_reg = 0.1, tradeoff = 0.1):
    covs_subset = list(covs_subset)

    s = time.time()
    num_control = len(df[df['treated']==0]) # how many control units that are unmatched (recall matched units are removed from the data frame)
    num_treated = len(df[df['treated']==1]) # how many treated units that are unmatched (recall matched units are removed from the data frame)
    
    num_control_matched = np.sum(( match_indicator ) & (df['treated']==0) ) # how many control units that are matched on this level
    num_treated_matched = np.sum(( match_indicator ) & (df['treated']==1) ) # how many treated units that are matched on this level
        
    time_BF = time.time() - s
    
    # -- below is the regression part for PE
    s = time.time()
    ridge_c = Ridge(alpha=ridge_reg) 
    ridge_t = Ridge(alpha=ridge_reg) 
    #tree_c = DecisionTreeRegressor(max_depth=8, random_state=0)
    #tree_t = DecisionTreeRegressor(max_depth=8, random_state=0)
        
    n_mse_t = np.mean(cross_val_score(ridge_t, holdout[holdout['treated']==1][covs_subset], 
                                holdout[holdout['treated']==1]['outcome'] , scoring = 'neg_mean_squared_error' ) )
        
    n_mse_c = np.mean(cross_val_score(ridge_c, holdout[holdout['treated']==0][covs_subset], 
                                holdout[holdout['treated']==0]['outcome'] , scoring = 'neg_mean_squared_error' ) )
    
    #n_mse_t = np.mean(cross_val_score(tree_t, holdout[holdout['treated']==1][covs_subset], 
    #                            holdout[holdout['treated']==1]['outcome'] , scoring = 'neg_mean_squared_error' ) )
        
    #n_mse_c = np.mean(cross_val_score(tree_c, holdout[holdout['treated']==0][covs_subset], 
    #                            holdout[holdout['treated']==0]['outcome'] , scoring = 'neg_mean_squared_error' ) )
    
    time_PE = time.time() - s
    # -- above is the regression part for PE
    
    # -- below is the level-wise MQ
    return  (tradeoff * ( float(num_control_matched)/num_control + float(num_treated_matched)/num_treated ) +             ( n_mse_t + n_mse_c ) , time_PE , time_BF ) 
    # -- above is the level-wise MQ
    
    #return (balance_reg * (num_treated_matched + num_control_matched) * ( float(num_control_matched)/num_control +\
    #                       float(num_treated_matched)/num_treated ) +\
    #         (num_treated_matched + num_control_matched) * ( n_mse_t  + n_mse_c ) , time_PE , time_BF ) 
    
# In[55]:

def get_CATE_bit(df, match_indicator, index):
    d = df[ match_indicator ]
    if index is None: # when index == None, nothing is matched
        return None
    d.loc[:,'grp_id'] = index

    res = d.groupby(['grp_id'])
    res_list = []
    for key, item in res:
        df = res.get_group(key)
        idx_list = df[df['treated'] == 1]['index'].tolist()
        df_t = df[df['treated']==1]
        df_c = df[df['treated']==0]
        mean_c = df_c['outcome'].mean()
        mean_t = df_t['outcome'].mean()
        mean = mean_t - mean_c
        res_list.append([Decimal(mean),idx_list])
  
    return res_list

# In[56]:

def recover_covs(d, covs, covs_max_list, binary = True):
    covs = list(covs)
    covs_max_list = list(covs_max_list)

    ind = d.index.get_level_values(0)
    ind = [ num2vec(ind[i], covs_max_list) for i in range(len(ind)) if i%2==0]

    df = pd.DataFrame(ind, columns=covs ).astype(int)

    mean_list = list(d['mean'])
    size_list = list(d['size'])
        
    effect_list = [mean_list[2*i+1] - mean_list[2*i] for i in range(len(mean_list)/2) ]
    df.loc[:,'effect'] = effect_list
    df.loc[:,'size'] = [size_list[2*i+1] + size_list[2*i] for i in range(len(size_list)/2) ]
    
    return df

def num2vec(num, covs_max_list):
    res = []
    for i in range(len(covs_max_list)):
        num_i = num/covs_max_list[i]**(len(covs_max_list)-1-i)
        res.append(num_i)
        
        if (num_i == 0) & (num%covs_max_list[i]**(len(covs_max_list)-1-i) == 0):
            res = res + [0]*(len(covs_max_list)-1-i)
            break
        num = num - num_i* covs_max_list[i]**(len(covs_max_list)-1-i)
    return res


# In[57]:

def run_bit(df, holdout, covs, covs_max_list, tradeoff_param = 0.1):
    covs = list(covs)
    covs_max_list = list(covs_max_list)

    constant_list = ['outcome', 'treated','matched', 'true_effect', 'index']
    
    covs_dropped = []
    cur_covs = covs[:]
    cur_covs_max_list = covs_max_list[:]

    timings = [0]*5 # first entry - match (matrix multiplication and value counting and comparison), 
                    # second entry - regression (compute PE),
                    # third entry - compute BF, fourth entry - keep track of CATE,
                    # fifth entry - update dataframe (remove matched units)
    
    level = 1
    #print("level ", level)
    s = time.time()
    match_indicator, index = match(df, cur_covs, covs_max_list) # match without dropping anything
    timings[0] = timings[0] + time.time() - s
    
    s = time.time()

    res = get_CATE_bit(df, match_indicator, index) # get the CATEs without dropping anything
    timings[3] = timings[3] + time.time() - s
    
    matching_res = [res] # result on first level, None says nothing is dropped
    
    s = time.time()

    init_score, time_PE, time_BF = match_quality(df, holdout, cur_covs, match_indicator, tradeoff=tradeoff_param)

    df = df[~match_indicator][ cur_covs + constant_list ] # remove matched units
    timings[4] = timings[4] + time.time() - s
    
    level_scores = []
    
    
    while len(cur_covs)>1:
        
        #print(cur_covs)

        level += 1
        #print("level ", level)

        matching_result_tmp = []
        
        if (np.sum(df['treated'] == 0) == 0 ) | (np.sum(df['treated'] == 1) == 0 ): # the early stopping condition
            print('no more matches')
            break
        
        for i in range(len(cur_covs)):
            
            cur_covs_no_c = cur_covs[:i] + cur_covs[i+1:]
            
            cur_covs_max_list_no_c = cur_covs_max_list[:i] + cur_covs_max_list[i+1:]
            
            s = time.time()
            match_indicator, index = match(df, cur_covs_no_c, cur_covs_max_list_no_c)
            timings[0] = timings[0] + time.time() - s 
            
            score, time_PE, time_BF = match_quality(df, holdout, cur_covs_no_c, match_indicator, tradeoff=tradeoff_param)
            timings[1] = timings[1] + time_PE 
            timings[2] = timings[2] + time_BF 
                                    
            matching_result_tmp.append( (cur_covs_no_c, cur_covs_max_list_no_c, score, match_indicator, index) )
        
        best_res = max(matching_result_tmp, key=itemgetter(2)) # use the one with largest MQ as the one to drop
        
        best_score = max( [t[2] for t in matching_result_tmp] )
        
        """
        if (init_score < 0 and best_score < 1.05 * init_score) or (init_score >= 0 and best_score < 0.95 * init_score):
            print("early stop")
            break
        """

        level_scores.append(best_score)
        
        del matching_result_tmp
        
        new_matching_res = get_CATE_bit(df, best_res[-2], best_res[-1])
        
        cur_covs = best_res[0] 
        cur_covs_max_list = best_res[1]
        matching_res.append(new_matching_res)
        
        s = time.time()
        df = df[~ best_res[-2] ]
        timings[4] = timings[4] + time.time() - s
    
    return matching_res

def merge_results(matching_res, catt_res, num_t):
    for match_res_level in matching_res:
        for match_grp in match_res_level:
            estimated_effect = match_grp[0]
            for i in range(len(match_grp[1])):
                matched_index = match_grp[1][i]
                if matched_index >= num_t:
                    if matched_index not in catt_res:
                        catt_res[matched_index] = []
                    catt_res[matched_index].append(estimated_effect)

def get_avg_imputation(catt_res):
    avg_catt = []
    for matched_index in catt_res:
        if len(catt_res[matched_index]) == 0:
            avg_catt.append(None)
        else:
            catt_list = catt_res[matched_index]
            avg_catt.append(float(sum(catt_list)) * 1.0 / len(catt_list))
    return avg_catt

if __name__ == '__main__':
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    imputation_num = 10

    catt_res = {}
    for i in range(imputation_num):
        print("imputed: " + str(i))
        df = pd.read_csv("data/imputed_ " + str(i + 1) + " .csv") 
        df = df.iloc[:,1:]
        df = df.reset_index()
        df['index'] = df.index 
        for i in range(df.shape[1] - 5):
            df = df.rename(columns = {"X"+str(i):i})
        
        with open("data/holdout1",'rb') as f:
            holdout = pickle.load(f) 

        res = run_bit(df, holdout, range(15), [2]*15, tradeoff_param = 0.1)

        merge_results(res, catt_res, int(df.shape[0] * 0.5))

    avg_catt_res = get_avg_imputation(catt_res)
    print(avg_catt_res)
    pickle.dump(avg_catt_res, open("res/FLAME-imputation", 'wb'))
    

   