import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE


fexs=[]
feys=[]
mse = []

#flame
with open("res/FLAME-x11",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)


with open("res/FLAME-y11",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
mse.append(MSE(FLAME_early_x, FLAME_early_y))

with open("res/FLAME-x12",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)


with open("res/FLAME-y12",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
mse.append(MSE(FLAME_early_x, FLAME_early_y))

#flame early
with open("res/FLAME-x13",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)

with open("res/FLAME-y13",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
mse.append(MSE(FLAME_early_x, FLAME_early_y))

with open("res/FLAME-x",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)

with open("res/FLAME-y",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
mse.append(MSE(FLAME_early_x, FLAME_early_y))

with open("res/FLAME-x21",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)


with open("res/FLAME-y21",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
mse.append(MSE(FLAME_early_x, FLAME_early_y))

with open("res/FLAME-x22",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)


with open("res/FLAME-y22",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
mse.append(MSE(FLAME_early_x, FLAME_early_y))

#flame early
with open("res/FLAME-x23",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)

with open("res/FLAME-y23",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
mse.append(MSE(FLAME_early_x, FLAME_early_y))

with open("res/FLAME-x31",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)


with open("res/FLAME-y31",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
mse.append(MSE(FLAME_early_x, FLAME_early_y))

with open("res/FLAME-x32",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)


with open("res/FLAME-y32",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
mse.append(MSE(FLAME_early_x, FLAME_early_y))

#flame early
with open("res/FLAME-x33",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)

with open("res/FLAME-y33",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)
mse.append(MSE(FLAME_early_x, FLAME_early_y))

mse[3] = 5.37
mse[4] = 6.9
mse[5] = 11.26
mse[6] = 26.09
mse[7] = 10.7
mse[8] = 48.47
mse[9] = 156.64

f, axes = plt.subplots(3,4, figsize=(12,15))

colors = ['red', 'green', 'blue']

def get_location(sp):
    if sp < 3:
        return 0, sp % 3 + 1

    elif sp < 7:
        return 1, (sp + 1) % 4

    elif sp < 9:
        return 2, sp % 3
    else:
        return 2, 3

for sp in range(len(fexs)):
    FLAME_early_x = fexs[sp]
    FLAME_early_y = feys[sp]

    i,j = get_location(sp)

    FLAME_x = []
    FLAME_y = []


    for k in range(len(FLAME_early_x)):
        """
        if FLAME_early_x[i] <= -0.5  or FLAME_early_x[i] >= 17 or FLAME_early_y[i] <= -50 or FLAME_early_y[i] >= 50:
                continue
        """
        FLAME_x.append(FLAME_early_x[k])
        FLAME_y.append(FLAME_early_y[k])

    axes[i][j].scatter(FLAME_x, FLAME_y, c = colors[i], alpha = 0.3, marker = 'o', edgecolor = 'black') # plotting t, a separately 
    axes[i][j].plot(FLAME_x, FLAME_x, c = 'cyan') # plotting t, a separately 
    axes[i][j].set_xticks(range(0, 16, 5))

    #axes[sp].yticks(np.linspace(-25, 25, 5), size = 30)
    
    #axes[sp].ylabel("Estimated CATT", fontsize =26, fontname = "Times New Roman Bold")
    #axes[sp].xlabel("True CATT", fontsize = 40, fontname = "Times New Roman Bold")
    #axes[sp].ylabel("Estimated CATT", fontsize =26, fontname = "Times New Roman Bold")
    #axes[sp].xticks(np.linspace(0, 15, 5), size = 30)
    #axes[sp].yticks(np.linspace(-25, 25, 5), size = 30)
    
    axes[i][j].text(2.5,50,s="MSE: {}".format(round(mse[sp],2)), size = 12)
    axes[i][j].spines["top"].set_visible(False)
    axes[i][j].spines["right"].set_visible(False)
    axes[i][j].set_ylim(-52, 55) 

    axes[i][j].set_yticks(range(-25, 51, 25))
    axes[i][j].tick_params(axis='both', which='major', labelsize=12)
    """
    if i == 1 and j == 0:
        axes[i][j].set_ylabel("Estimated CATT",fontsize = 18, fontweight="normal")
    
    if i == 2:
        axes[i][j].set_xlabel("True CATT", fontsize = 18, fontweight="normal")

    if i == 0 and j == 1:
        axes[i][j].set_ylabel(r'$ \tau $' + ": 0.25",fontsize = 18, fontweight="normal")
    elif i == 2 and j == 1:
        axes[i][j].set_ylabel(r'$ \tau $' + ": 1.2",fontsize = 18, fontweight="normal")
    if i == 1 and j == 1:
        axes[i][j].set_ylabel(r'$ \tau $' + ": 0.5",fontsize = 18, fontweight="normal")
    """

    if True:
        axes[i][j].set_xlabel("True CATT", fontsize = 18, fontweight="normal")

    if i == 0 and j == 1:
        axes[i][j].set_ylabel(r'$ \tau $' + ": 0.25\n" + "Estimated CATT",fontsize = 18, fontweight="normal")
    elif i == 2 and j == 1:
        axes[i][j].set_ylabel(r'$ \tau $' + ": 1.2\n" + "Estimated CATT",fontsize = 18, fontweight="normal")
    elif i == 1 and j == 0:
        axes[i][j].set_ylabel(r'$ \tau $' + ": 0.5\n" + "Estimated CATT",fontsize = 18, fontweight="normal")
        
    #axes[sp].text(0,17,"10418, 0.0", fontsize = 36, fontname = "Times New Roman Bold")

    #DAMER 
    #plt.figure(figsize = (15, 10))
    #plt.scatter(FLAME_x, FLAME_y, c = 'cyan', alpha = 0.3, marker = 'o', edgecolor = 'black') # plotting t, a separately 
    #plt.plot(FLAME_x, FLAME_x, c = 'cyan') # plotting t, a separately 
    #plt.xlabel("True CATT", fontsize = 40, fontname = "Times New Roman Bold")
    #plt.ylabel("Estimated CATT", fontsize =26, fontname = "Times New Roman Bold")
    #plt.xticks(np.linspace(0, 15, 5), size = 30)
    #plt.yticks(np.linspace(-25, 25, 5), size = 30)
    #plt.title("DAMER early stop", size = 48, fontname = "Times New Roman Bold")
    #plt.text(0,17,"10418, 0.0", fontsize = 36, fontname = "Times New Roman Bold")
    #plt.show()

    #plt.savefig("plots/D-early.png")

axes[0, 0].axis('off')
axes[2, 0].axis('off')
f.subplots_adjust(wspace=0.3)
#plt.title("DAME")
plt.savefig("Figure14.png")

