import numpy
from matplotlib import pyplot

"""
x = [5,8,10]
y_gen = [0.50,0.94,1.45]
y_col = [0.32,3.26,12.67]
y_bf = [11.17,518.87,1459.60]

fig,axes = pyplot.subplots(1, 2, figsize=(15,9))
ax = axes[0]
ax.set_xlim(4.9,10.1)
ax.set_xticks(range(5, 11, 1))
#ax.xticks(fontsize = 20)
#ax.set_ylim(-5,200)
#ax.set_yticks(range(0, 201, 50))
#ax.yticks(fontsize = 20)
ax.plot(x,y_gen,marker = 'o', linewidth=4, markersize = 15, c = 'orange', label = "FLAME")
ax.plot(x,y_col, marker = '^', linewidth=4, markersize = 15, c = 'green', label = "DAME")
ax.plot(x,y_bf, marker = 's', linewidth=4, markersize = 15, c = 'red', label = "Brute Force")
ax.set_xlabel('number of covariates(n = 3k units fixed)', fontsize = 20)
ax.set_ylabel('running time(in seconds)', fontsize = 20)
ax.legend(prop={'size': 20})
ax.tick_params(labelsize=20)

for i,j in zip(x,y_gen):
    ax.annotate(str(j),xy=(i,j), xytext = (i, j - 55), fontsize = 18)

for i,j in zip(x,y_col):
    ax.annotate(str(j),xy=(i,j), xytext = (i, j + 5), fontsize = 18)

for i,j in zip(x,y_bf):
    ax.annotate(str(j),xy=(i,j), xytext = (i, j + 25), fontsize = 18)

#pyplot.savefig("running time 3k.png")

x = [1000,3000,5000]
y_gen = [1.30,1.37,1.48]
y_col = [13.76,13.59,13.99]
y_bf = [389.29,468.35,526.53]

#fig = pyplot.figure(figsize=(9,7))
ax = axes[1]
ax.set_xlim(900,5100)
ax.set_xticks(range(1000, 5100, 1000))
ax.tick_params(labelsize=20)
#ax.xticks(fontsize = 20)
#ax.set_yticks(range(0, 801, 200))
#ax.yticks(fontsize = 20)
ax.plot(x,y_gen,marker = 'o', linewidth=4, markersize = 15, c = 'orange', label = "FLAME")
ax.plot(x,y_col, marker = '^',linewidth=4, markersize = 15, c = 'green', label = "DAME")
ax.plot(x,y_bf, marker = 's', linewidth=4,markersize = 15, c = 'red', label = "Brute Force")
ax.set_xlabel('number of units(p = 10 covariates)', fontsize = 20)
ax.set_ylabel('running time(in seconds)', fontsize = 20)
ax.legend(loc='center', prop={'size': 20})

for i,j in zip(x,y_gen):
    ax.annotate(str(j),xy=(i,j), xytext = (i, j - 25), fontsize = 18)

for i,j in zip(x,y_col):
    ax.annotate(str(j),xy=(i,j), xytext = (i, j + 5), fontsize = 18)

for i,j in zip(x,y_bf):
    ax.annotate(str(j),xy=(i,j), xytext = (i, j + 25), fontsize = 18)

fig.subplots_adjust(wspace=0.4)

pyplot.savefig("running_time.png")


"""

x = [20000,30000,40000]
y_gen = [22.19,28.74,44.09]

fig = pyplot.figure(figsize=(9,7))
ax = fig.add_subplot(111)
ax.set_xlim(19000,41000)
ax.set_xticks(range(20000, 41000, 5000))
pyplot.xticks(fontsize = 20)
#ax.set_ylim(-100,2100)
#ax.set_yticks(range(0, 2000, 500))
pyplot.yticks(fontsize = 20)
pyplot.plot(x,y_gen,marker = 's',linewidth=4,  markersize = 15, c = 'green', label = "DAME")
ax.set_xlabel('number of units(p = 12 covariates)', fontsize = 20)
ax.set_ylabel('running time(in seconds)', fontsize = 20)
ax.legend(loc='upper center', fontsize = 20)

for i,j in zip(x,y_gen):
    ax.annotate(str(j),xy=(i,j), xytext = (i, j), fontsize = 18)

pyplot.savefig("running_time_12 covs.png")



