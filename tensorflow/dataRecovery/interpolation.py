# https://studfiles.net/all-vuz/145/folder:8845/#4494688
# https://medium.com/@drnesr/filling-gaps-of-a-time-series-using-python-d4bfddd8c460

import pandas as pd
import numpy as np
from numpy import genfromtxt
from scipy.interpolate import griddata
from sklearn.preprocessing import Imputer

df = pd.read_csv("./dataToRestore.csv", sep=",", dtype={'x1': np.float64, 'x2': np.float64, 'x3': np.float64, 'y': np.float64})

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
