import pandas as pd

df = pd.read_csv('wine.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().any())
