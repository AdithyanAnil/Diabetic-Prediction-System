import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/diabetes.csv")
x=df.iloc[:,0:8]
y=df.iloc[:,-1]

mm=MinMaxScaler()
mm.fit(x)
mx=mm.transform(x)