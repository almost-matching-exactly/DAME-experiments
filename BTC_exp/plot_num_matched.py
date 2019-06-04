import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE

ticksize = 14
labelsize=16
titlesize =16

def draw_hist(ax, x, x1, x2, daem, flame, percent):


    colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow']

    ax.bar(x1,  daem, width = 0.4, color="green", label = "DAME", hatch="/")
    ax.bar(x2, flame, width = 0.4, color = "orange", label = "FLAME", hatch = "\\")
    ax.legend(fontsize=labelsize)
    ax.set_xlim(0.5, 11)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel("number of covariates matched", fontsize=labelsize)
    ax.set_ylabel("number of units matched", fontsize=labelsize)
    ax.set_xticks(x)
    ax.set_ylim(0,max(daem)*1.15)
    ax.tick_params(labelsize=ticksize)
    def autolabel(ax, x, y):
        """
        Attach a text label above each bar displaying its height
        """
        for i in range(len(x)):
            height = y[i]
            xl = x[i]
            if height>0:
                ax.text(xl, height,
                        '%d' % int(height),
                        ha='center', va='bottom', fontsize=9)
    autolabel(ax, x1, daem)
    autolabel(ax, x2, flame)
    #plt.savefig(filename)

x = [10,9,8,7,6,5,4,3,2,1]
#x1 = [e - 0.2 for e in x]
#x2 = [e + 0.2 for e in x]
f, axes = plt.subplots(1, 1,  figsize=(8,5))
dame = [287, 69, 21, 2, 1, 0, 0, 0, 0, 0]
flame = [287, 7,  25, 9, 7,12, 6, 5, 11, 5]
draw_hist(axes, x, x1, x2, dame, flame, 50)
f.subplots_adjust(wspace=0.1)
f.subplots_adjust(hspace=0.5)
#f.text(0.10, 0.45, "First time at least 15000 units are matched", fontweight="semibold", fontsize = labelsize)


plt.savefig("num_matched_btc.png")


