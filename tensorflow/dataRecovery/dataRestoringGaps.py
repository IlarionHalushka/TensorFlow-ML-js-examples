import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./dataToRestore.csv", sep=",")

print(data)

data.fillna(method='bfill',inplace=True)

print(data)
