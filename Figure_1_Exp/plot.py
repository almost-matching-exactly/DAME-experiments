import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE


fexs=[]
feys=[]
titles=["DAME early", "FLAME early", "DAME end", "FLAME end", "1-PSNNM", "Mahalanobis"]
num_matched=[]
padding=[0.2, 0.4, 0.6, 0.5, 1.5, 0.2]

#flame
with open("data/DAEMR-early-x",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)


with open("data/DAEMR-early-y",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)

num_matched.append((15000,0.0))

with open("data/FLAME-early-x",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)


with open("data/FLAME-early-y",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)

num_matched.append((11863,0.0))

#flame early
with open("res/DAEMR-x",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)

with open("res/DAEMR-y",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)


with open("data/df",'rb') as f:
    df = pickle.load(f)  

num_matched.append((15000, 0.0))

#flame early
with open("data/FLAME-x",'rb') as f:
        FLAME_early_x = pickle.load(f) 
        fexs.append(FLAME_early_x)

with open("data/FLAME-y",'rb') as f:
        FLAME_early_y = pickle.load(f) 
        feys.append(FLAME_early_y)


with open("data/df",'rb') as f:
    df = pickle.load(f)  

num_matched.append((14579, 21.92))

#FLAME_early_x = list(df[df["treated"] == 1]["true_effect"])
#fexs.append(FLAME_early_x)
#
#
#f = open("data/cf_out.txt", "r+")
#FLAME_early_y = [float(line.replace("\n","")) for line in f.readlines()]
#f.close()
#feys.append(FLAME_early_y)
#num_matched.append(100)

FLAME_early_x = list(df[df["treated"] == 1]["true_effect"])
fexs.append(FLAME_early_x)


f = open("data/ps_out.txt", "r+")
FLAME_early_y = [float(line.replace("\n","")) for line in f.readlines()]
f.close()
feys.append(FLAME_early_y)
num_matched.append((15000, 278.04))

FLAME_early_x = list(df[df["treated"] == 1]["true_effect"])
fexs.append(FLAME_early_x)


f = open("data/maha_out.txt", "r+")
FLAME_early_y = [float(line.replace("\n","")) for line in f.readlines()]
f.close()
feys.append(FLAME_early_y)
num_matched.append((15000, 7.54))

cf_x = list(df[df["treated"] == 1]["true_effect"])


f = open("data/cf_out.txt", "r+")
cf_y = [float(line.replace("\n","")) for line in f.readlines()]
f.close()

f, axes = plt.subplots(1,len(fexs) + 1, sharex= True, figsize=(15,4))
print(titles)

colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow']

for sp in range(len(fexs)):
    FLAME_early_x = fexs[sp]
    FLAME_early_y = feys[sp]

    FLAME_x = []
    FLAME_y = []


    for i in range(len(FLAME_early_x)):
            if FLAME_early_x[i] <= -0.5  or FLAME_early_x[i] >= 17 or FLAME_early_y[i] <= -50 or FLAME_early_y[i] >= 50:
                    continue
            FLAME_x.append(FLAME_early_x[i])
            FLAME_y.append(FLAME_early_y[i])

    axes[sp].scatter(FLAME_x, FLAME_y, c = colors[sp], alpha = 0.3, marker = 'o', edgecolor = 'black') # plotting t, a separately 
    axes[sp].plot(FLAME_x, FLAME_x, c = 'cyan') # plotting t, a separately 
    axes[sp].set_xticks(range(0, 16, 5))
    #axes[sp].yticks(np.linspace(-25, 25, 5), size = 30)
    #axes[sp].xlabel("True CATT", fontsize = 40, fontname = "Times New Roman Bold")
    #axes[sp].ylabel("Estimated CATT", fontsize =26, fontname = "Times New Roman Bold")
    #axes[sp].xlabel("True CATT", fontsize = 40, fontname = "Times New Roman Bold")
    #axes[sp].ylabel("Estimated CATT", fontsize =26, fontname = "Times New Roman Bold")
    #axes[sp].xticks(np.linspace(0, 15, 5), size = 30)
    #axes[sp].yticks(np.linspace(-25, 25, 5), size = 30)
    wid = axes[sp].get_window_extent().width

    title = axes[sp].set_title(titles[sp], backgroundcolor="white", pad = 0.2, wrap = True)
    title._bbox_patch._mutation_aspect=0.05
    title.get_bbox_patch().set_boxstyle("square", pad=padding[sp])

    #bb = title.get_bbox_patch()
    #bb.set_width(axes[sp].get_window_extent().width)
    #title.set_bbox(bb)


    axes[sp].text(2.5,-45,s="{}".format(num_matched[sp]))
    axes[sp].spines["top"].set_visible(False)
    axes[sp].spines["right"].set_visible(False)
    axes[sp].set_ylim(-52, 55) 
    if sp != 0:
        axes[sp].get_yaxis().set_visible(False)
        axes[sp].spines["left"].set_visible(False)
    else:
        axes[sp].set_yticks(range(-25, 51, 25))
        axes[sp].set_ylabel("Estimated CATT",fontweight="normal")
    #axes[sp].savefig("CATT"+str(s)+str(p)+".png")

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

#f.text(0.135, 0.82, "Number of treatment units matched", fontsize =14)
f.subplots_adjust(wspace=0.1)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
f.text(0.435, 0.03, "True CATT", fontweight="normal")
ax = axes[-1]
ax.scatter(cf_x, cf_y, c = 'cyan', alpha = 0.3, marker = 'o', edgecolor = 'black') # plotting t, a separately 
ax.plot(cf_x, cf_x, c = 'cyan') # plotting t, a separately 
title = ax.set_title("Causal Forest", backgroundcolor = "white", wrap = True)
title._bbox_patch._mutation_aspect=0.05
title.get_bbox_patch().set_boxstyle("square", pad=1.2)
ax.set_ylim(-5, 127)
ax.set_yticks(range(0, 127, 25))
axes[-1].text(2.5,115,s="{}".format((15000, 470.84)))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
position = ax.get_position()
print(position)
ax.set_position([0.85, 0.10999999999999, 0.13, 0.75])
f.text(0.90, 0.03, "True CATT", fontweight="normal")
#f.text(0.37, 0.73, "NOT RECOMMENDED", fontweight="semibold", color="red")
#arrow = axes[2].arrow(16,39, -5, 8, head_width = 1, head_length = 2, fc='k', ec='k')
#f.patches.append(arrow)
#arrow = axes[3].arrow(0,39, 5, 8, head_width = 1, head_length = 2, fc='k', ec='k')
#f.patches.append(arrow)
plt.savefig("CATT.png")

