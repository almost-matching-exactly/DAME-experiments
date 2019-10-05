import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE

fexs=[]
feys=[]
titles=["DAME 30%", "FLAME 30%", "DAME 50%", "FLAME 50%"]
mse=[0.39, 0.93, 0.54, 2.47]
ticksize = 14
labelsize=16
titlesize =16

with open("res/DAEMR-x-1",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)

with open("res/DAEMR-y-1",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
#flame
with open("res/FLAME-x-1",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)


with open("res/FLAME-y-1",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)

#flame early

#flame early
with open("res/DAEMR-x-2",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)

with open("res/DAEMR-y-2",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)


with open("res/FLAME-x-2",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)


with open("res/FLAME-y-2",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)



x = [18, 17, 16, 15, 14, 13]
x1 = [e - 0.2 for e in x]
x2 = [e + 0.2 for e in x]
daem = [1722, 8296, 0,0,0,0]
flame = [1722, 750,  1244, 2009, 2509, 2477]

f, axes = plt.subplots(2, 3, gridspec_kw = {'width_ratios':[1, 1, 2]}, figsize=(12,9))

def draw_hist(ax, x, x1, x2, daem, flame, percent):


    colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow']

    ax.bar(x1,  daem, width = 0.4, color="green", label = "DAME {}%".format(percent), hatch="/")
    ax.bar(x2, flame, width = 0.4, color = "orange", label = "FLAME {}%".format(percent), hatch = "\\")
    ax.legend(fontsize=labelsize)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel("number of covariates matched on", fontsize=labelsize)
    ax.set_ylabel("number of units matched", fontsize=labelsize)
    ax.set_xticks(x)
    ax.set_ylim(0,max(daem)*1.3)
    ax.tick_params(labelsize=ticksize)
    def autolabel(ax, x, y):
        """
        Attach a text label above each bar displaying its height
        """
        for i in range(len(x)):
            height = y[i]
            xl = x[i]
            ax.text(xl, height,
                    '%d' % int(height),
                    ha='center', va='bottom', rotation=90, fontsize=ticksize)
    autolabel(ax, x1, daem)
    autolabel(ax, x2, flame)
    #plt.savefig(filename)

def draw_scatter(ax, x, y, title, color, mse, yticks= False):
    ax.scatter(x, y, c = color, alpha = 0.3, marker = 'o', edgecolor = 'black') # plotting t, a separately 
    ax.plot(x,x, c = 'cyan') # plotting t, a separately 
    ax.set_xticks(range(0, 26, 5))
    ax.set_title(title, pad = 0.2, wrap = True, fontsize=labelsize)
    ax.set_xlim(-1, 26)
    ax.set_ylim(-1, 26)
    ax.tick_params(labelsize=ticksize)
    #ax.spines["top"].set_visible(False)
    if(yticks):
    #    ax.spines["right"].set_visible(False)
        ax.set_yticks(range(0, 26, 5))
        ax.set_ylabel("Estimated CATT", fontsize = labelsize)
    else:
        ax.get_yaxis().set_visible(False)
    #    ax.spines["left"].set_visible(False)
    ax.text(0, 23, "MSE: {}".format(mse), fontsize=labelsize)


draw_hist(axes[0][2], x, x1, x2, daem, flame, 30)
daem = [1633, 9109, 4895,0,0,0]
flame = [1633, 704,  1283, 2018, 2525, 2406]
draw_hist(axes[1][2], x, x1, x2, daem, flame, 50)
draw_scatter(axes[0][0], fexs[0], feys[0], titles[0],  "green", mse[0], True)
draw_scatter(axes[0][1], fexs[1], feys[1], titles[1],  "orange",mse[1], False)
draw_scatter(axes[1][0], fexs[2], feys[2], titles[2],  "green", mse[2], True)
draw_scatter(axes[1][1], fexs[3], feys[3], titles[3],  "orange",mse[3], False)
f.subplots_adjust(wspace=0.1)
f.subplots_adjust(hspace=0.5)
f.text(0.28, 0.06, "True CATT", fontsize=labelsize)
f.text(0.28, 0.51, "True CATT", fontsize = labelsize)
f.text(0.10, 0.91, "First time at least 9000 units are matched", fontweight="semibold", fontsize=labelsize)
f.text(0.10, 0.45, "First time at least 15000 units are matched", fontweight="semibold", fontsize = labelsize)

position = axes[0][2].get_position()
axes[0][2].set_position([0.65,0.572, 0.3033,  0.308])
print(position)
position = axes[1][2].get_position()
axes[1][2].set_position([0.65, 0.11, 0.3033, 0.308])
print(position)
plt.savefig("decay_expo_exp.png")


