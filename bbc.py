import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

#data cleaning and loading 
data = pd.read_csv("data.csv")
data = data.drop(columns=['id', 'Unnamed: 32'])
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data = data.drop_duplicates()

# Prepare Data
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = AdaBoostClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100


#save the model for  future use
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)




# Streamlit App
st.title("Breast Cancer Prediction App")
st.write(f"Model Accuracy: {accuracy:.2f}%")

# Input Fields
st.sidebar.header("Enter Feature Values")
input_values = []
for feature in X.columns:
    value = st.sidebar.number_input(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    input_values.append(value)

# Prediction
if st.sidebar.button("Predict"):
    result = model.predict([input_values])
    diagnosis = "Malignant (Cancer)" if result[0] == 1 else "Benign (No Cancer)"
    st.subheader("Prediction Result:")
    st.write(f"### {diagnosis}")



st.markdown(
    '<a href="https://www.linkedin.com/in/suvashis-panda-05a209315/" target="_blank">'
    '<button style="background-color:#008CBA;color:white;padding:10px 24px;'
    'border:none;border-radius:5px;cursor:pointer;font-size:16px;">'
    'Suvashis Panda</button></a>',
    unsafe_allow_html=True
)