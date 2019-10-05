import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
import seaborn as sns

#flame
with open("data/df1",'rb') as f:
    df = pickle.load(f)

df = df.iloc[:,:15]
f, ax = plt.subplots(figsize=(6, 6))
corr = df.corr()
print(corr)

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.title("Correlation Matrix with 5% Missing Data", fontsize = 16)
plt.savefig("Col_Corr.png")


