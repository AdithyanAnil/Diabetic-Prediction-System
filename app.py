import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/diabetes.csv")
x=df.iloc[:,0:8]
y=df.iloc[:,-1]

mm=MinMaxScaler()
mm.fit(x)
mx=mm.transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)