# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st


# Set desired size in pixels
width_pixels = 400
height_pixels = 200

# Convert pixels to inches (assuming default DPI of 100)
dpi = 100
width_inches = width_pixels / dpi
height_inches = height_pixels / dpi 


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values



# Training the Polynomial Regression model on the whole dataset
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)



# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

st.title("Predicting Salary According to given Experience(Years)")
years_experience = st.sidebar.number_input("Enter number of years of experience:", min_value=0, max_value=50, step=1)

labels = []
for i in range(10):
    labels.append(dataset.iloc[i][0])



# Create a submit button
if st.sidebar.button("Submit"):
    salary_predicted = lin_reg_2.predict(poly_reg.fit_transform([[years_experience]]))
    st.success(f'Predicted Salary for Given Experience is {salary_predicted[0].round()} $')
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))
    ax.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
    ax.scatter(X, y, color = 'red')
    ax.scatter(years_experience, salary_predicted, color="green")
    ax.set(title='Salary vs Experience (Polynomial Regression)', xlabel='Years of Experience', ylabel='Salary')
    st.pyplot(fig)