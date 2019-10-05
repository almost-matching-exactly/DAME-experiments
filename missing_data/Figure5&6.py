import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE


fexs=[]
feys=[]
titles=["DAME", "FLAME"]

#flame
with open("data/df1",'rb') as f:
    true_effect = pickle.load(f)

true_effect = true_effect['true_effect'].tolist()[1000:]
fexs.append(true_effect)
fexs.append(true_effect)

with open("res/DAME-imputation",'rb') as f:
    estimated_effect = pickle.load(f)  
feys.append(estimated_effect)

with open("res/FLAME-imputation",'rb') as f:
    estimated_effect = pickle.load(f)  
feys.append(estimated_effect)


"""
with open("res/DAME-x-miss",'rb') as f:
    x = pickle.load(f)
    fexs.append(x)

with open("res/DAME-y-miss",'rb') as f:
    y = pickle.load(f)  
    feys.append(y)

with open("res/FLAME-x-miss",'rb') as f:
    x = pickle.load(f)
    fexs.append(x)

with open("res/FLAME-y-miss",'rb') as f:
    y = pickle.load(f) 
    feys.append(y)
"""

f, axes = plt.subplots(1,2, sharex= True, sharey = True, figsize=(10,6))

colors = ['red', 'red']

for sp in range(len(fexs)):
    FLAME_x = fexs[sp]
    FLAME_y = feys[sp]

    axes[sp].scatter(FLAME_x, FLAME_y, c = colors[sp], alpha = 0.3, marker = 'o', edgecolor = 'black') # plotting t, a separately 
    axes[sp].plot(FLAME_x, FLAME_x, c = 'cyan') # plotting t, a separately 
    axes[sp].set_xticks(range(0, 16, 5))
    title = axes[sp].set_title(titles[sp], backgroundcolor="white", pad = 0.2, wrap = True, fontsize = 24)
    title._bbox_patch._mutation_aspect=0.05
    
    axes[sp].text(12,-60,s="MSE: {0:.2f}".format(MSE(FLAME_x,FLAME_y)), fontsize = 20)
    axes[sp].spines["top"].set_visible(False)
    axes[sp].spines["right"].set_visible(False)
    axes[sp].tick_params(axis='both', which='major', labelsize=18)
    
    axes[sp].set_xlabel("True CATT",fontweight="normal", fontsize = 24)
    if sp == 0:
        axes[sp].set_ylabel("Estimated CATT",fontweight="normal", fontsize = 24)

#plt.title("DAME VS FLAME: Imputation")
plt.savefig("Comparison_Imputation.png")

