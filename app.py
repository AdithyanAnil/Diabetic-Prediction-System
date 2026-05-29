import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
df = pd.read_csv("data/diabetes.csv")
x=df.iloc[:,0:8]
y=df.iloc[:,-1]

mm=MinMaxScaler()
mm.fit(x)
mx=mm.transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)


st.set_page_config(page_title="Diabetic Predictor")

st.markdown(
    f"<h1 style='color:violet;'>Diabetic Prediction System</h1>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"<h1 style='color:violet;'>Diabetic Prediction System</h1>",
    unsafe_allow_html=True
)


st.sidebar.write("Enter the result values correctly.")

pregnancy = st.sidebar.number_input("Number of Pregnancies", min_value=0)

glucose = st.sidebar.number_input("Glucose Level", min_value=0)

bp = st.sidebar.number_input("Blood Pressure", min_value=0)

skin = st.sidebar.number_input("Skin Thickness", min_value=0)

insulin = st.sidebar.number_input("Insulin Level", min_value=0)

bmi = st.sidebar.number_input("BMI", min_value=0.0, )

dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, )

age = st.sidebar.number_input("Age", min_value=1)

submit = st.sidebar.button("Test")



st.markdown(
    f"<h1 style='color:green;'>This System has {accuracy*100:.2f}% accuracy</h1>",
    unsafe_allow_html=True
)
if submit:

    

    if glucose == 0:
        st.error("Enter glucose level")

    elif bp == 0:
        st.error("Enter blood pressure")

    elif skin == 0:
        st.error("Enter skin thickness")

    

    elif bmi == 0:
        st.error("Enter BMI")

    

    elif age == 0:
        st.error("Enter age")
    else:
        sample=[[pregnancy,glucose,bp,skin,insulin,bmi,dpf,age]]
        sample=mm.transform(sample)
        prediction=knn.predict(sample)
        if prediction[0]==1:
           st.error('You might Diabetic')

        else:
            st.success('You are not Diabetic')