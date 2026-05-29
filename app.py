import pandas as pd
df = pd.read_csv("data/diabetes.csv")
x=df.iloc[:,0:8]
y=df.iloc[:,-1]