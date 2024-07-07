# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st


# Set desired size in pixels
width_pixels = 400
height_pixels = 200

# Convert pixels to inches (assuming default DPI of 100)
dpi = 100
width_inches = width_pixels / dpi
height_inches = height_pixels / dpi 


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print("/////// predicted single value //////")
years_experience = st.sidebar.number_input('Enter number of years of experience:', min_value=0, max_value=50, value=5)
years_experience_predicted = regressor.predict([[years_experience]])

st.title("Predicting Salary by Experience")
if st.sidebar.button('Submit'):
    years_experience_predicted = regressor.predict([[years_experience]])
    st.success(f'Predicted Salary for Given Experience is {years_experience_predicted[0].round()} $')
    


# Create a plot
fig, ax = plt.subplots(figsize=(width_inches, height_inches))
ax.plot(X_train, regressor.predict(X_train), color = 'blue')
ax.scatter(X_train, y_train, color = 'red')
ax.set(title='Salary vs Experience (Training set)', xlabel='Years of Experience', ylabel='Salary')
st.pyplot(fig)

# Create a plot
fig2, ax2 = plt.subplots(figsize=(width_inches, height_inches))
ax2.plot(X_train, regressor.predict(X_train), color = 'blue')
ax2.scatter(X_test, y_test, color = 'red')
ax2.set(title='Salary vs Experience (Test set)', xlabel='Years of Experience', ylabel='Salary')
st.pyplot(fig2)

# # Visualising the Training set results if required in graph plot.show()
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')

# # Visualising the test set results if required in graph plot.show()
# plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Test set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
