# Multiple Linear Regression (predicting the profit in from given startup data from New-York, California, Florida)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import streamlit as st


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Encoding categorical data

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



# Define the options for the dropdown menu
options = ["New York", "California", "Florida"]
research_spend = st.number_input(
    "Enter a R&D Spend in Dollars($)",
    min_value=1000,  # Minimum value
    max_value=10000000,  # Maximum value (1 crore)
    value=100000,  # Default value
    step=1000  # Step size for increment/decrement
)
admistration_spend = st.number_input(
    "Enter a Administration Spend in Dollars($)",
    min_value=1000,  # Minimum value
    max_value=10000000,  # Maximum value (1 crore)
    value=100000,  # Default value
    step=1000  # Step size for increment/decrement
)
Marketing = st.number_input(
    "Enter a Marketing Spend in Dollars($)",
    min_value=1000,  # Minimum value
    max_value=10000000,  # Maximum value (1 crore)
    value=100000,  # Default value
    step=1000  # Step size for increment/decrement
)

# Create the dropdown menu in the sidebar
selected_option = st.sidebar.selectbox("Choose a City ", options)

# Create a submit button
if st.sidebar.button("Submit"):
    # Display the selected value when the button is clicked
    if (selected_option == 'New York'):
        y_pred = regressor.predict([[0,0,1,research_spend,admistration_spend,Marketing]])
        st.success(f'Estimated Profit for the Startup in New York is {y_pred[0].round()}')
    if (selected_option == "California"):
        y_pred = regressor.predict([[1,0,0,research_spend,admistration_spend,Marketing]])
        st.success(f'Estimated Profit for the Startup in California is {y_pred[0].round()}')
    if (selected_option == "Florida"):
        y_pred = regressor.predict([[0,1,0,research_spend,admistration_spend,Marketing]])
        st.success(f'Estimated Profit for the Startup in Florida is {y_pred[0].round()}')
    

