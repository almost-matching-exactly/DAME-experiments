import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE

"""
with open("data/df1",'rb') as f:
    true_effect = pickle.load(f)

true_effect = true_effect['true_effect'].tolist()[1000:]

with open("res/DAME-imputation",'rb') as f:
    estimated_effect = pickle.load(f)  

print(len(estimated_effect))

"""
with open("res/DAME-x-miss",'rb') as f:
    x = pickle.load(f)

with open("res/DAME-y-miss",'rb') as f:
    y = pickle.load(f)  

print(x)
print(y)
plt.scatter(x,y,color='r')
plt.ylim((-10,40))
#plt.text(15,80,"MSE: " + "{0:.2f}".format(MSE(x,y)))
plt.plot(x,x,color='b')
plt.xlabel("True CATT")
plt.ylabel("Estimated CATT")
plt.title("DAME Without Imputation")
plt.savefig("No_Imputation_DAME.png")
