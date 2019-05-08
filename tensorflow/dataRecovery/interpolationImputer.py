# https://studfiles.net/all-vuz/145/folder:8845/#4494688
# https://medium.com/@drnesr/filling-gaps-of-a-time-series-using-python-d4bfddd8c460

import pandas as pd
import numpy as np
from numpy import genfromtxt
from scipy.interpolate import griddata
from sklearn.preprocessing import Imputer

df = genfromtxt(open("./dataToRestore.csv", "r"), delimiter=",", skip_header=1, dtype=float)
print df

x = df.copy()
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
result = imp.fit_transform(x)
print result
