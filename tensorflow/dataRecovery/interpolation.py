# https://studfiles.net/all-vuz/145/folder:8845/#4494688

import pandas as pd
import numpy as np
from numpy import genfromtxt
from scipy.interpolate import griddata
from sklearn.preprocessing import Imputer

df = pd.read_csv("./dataToRestore.csv", sep=",", dtype={'x1': np.float64, 'x2': np.float64, 'x3': np.float64, 'y': np.float64})
# df = genfromtxt(open("./dataToRestore.csv", "r"), delimiter=",", skip_header=1, dtype=float)
# print df

# x = df.copy()
# imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# result = imp.fit_transform(x)
# print result


# FOLLOWING IS USING GRID DATA

# x1 = np.linspace(-1,1,10)
# x2 =  np.linspace(-1,1,10)
# x3 =  np.linspace(-1,1,10)
# X1, X2, X3 = np.meshgrid(x1,x2,x3)
#
# res = griddata((df['x1'],df['x2'], df['x3']), df['y'], (X1,X2,X3), method='linear', fill_value='NaN', rescale=False)
#
# print df['y']
# print res




# FOLOWING ARE INTERPOLATION, MEAN, MEDIAN:

df_interpolated = df.assign(InterpolateLinear=df['y'].interpolate(method='linear'))
print df_interpolated

# df_mean = df.assign(FillMean=df['y'].fillna(df['y'].mean()))
# print df_mean
#
# df_median = df.assign(FillMedian=df['y'].fillna(df['y'].median()))
# print df_median

# results = [(method, r2_score(df.reference, df[method])) for method in list(df)[3:]]
# results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])
# results_df.sort_values(by='R_squared', ascending=False)


# final_df= df[['reference', 'target', 'missing', 'InterpolateTime' ]]
# final_df.plot(style=['b-.', 'ko', 'r.', 'rx-'], figsize=(20,10));
# plt.ylabel('Temperature');
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           fancybox=True, shadow=True, ncol=5, prop={'size': 14} );













# https://medium.com/@drnesr/filling-gaps-of-a-time-series-using-python-d4bfddd8c460

