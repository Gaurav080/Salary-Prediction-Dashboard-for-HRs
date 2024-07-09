import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('Salary_Data.csv')
df = df.dropna(subset=['Salary'])

# Define features and target variable
X = df[['Years of Experience', 'Age']]
y = df['Salary']

# Impute missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Streamlit app
st.title('Salary Prediction App')

# User inputs
experience = st.number_input('Enter your experience (in years):', min_value=0, step=1)
age = st.number_input('Enter your age:', min_value=0, step=1)

# Prediction
if st.button('Predict Salary'):
    # Impute missing values for the input data
    input_data = np.array([[experience, age]])
    input_data = imputer.transform(input_data)
    prediction = model.predict(input_data)
    st.write(f'Predicted Salary: ${prediction[0]:,.2f}')
