# https://studfiles.net/all-vuz/145/folder:8845/#4494688
# https://medium.com/@drnesr/filling-gaps-of-a-time-series-using-python-d4bfddd8c460

import pandas as pd
import numpy as np
from numpy import genfromtxt
from scipy.interpolate import griddata
from sklearn.preprocessing import Imputer

df = genfromtxt(open("./dataToRestore.csv", "r"), delimiter=",", skip_header=1, dtype=float)
print df

x1 = np.linspace(-1,1,10)
x2 =  np.linspace(-1,1,10)
x3 =  np.linspace(-1,1,10)
X1, X2, X3 = np.meshgrid(x1,x2,x3)

res = griddata((df['x1'],df['x2'], df['x3']), df['y'], (X1,X2,X3), method='linear', fill_value='NaN', rescale=False)

print df['y']
print res
