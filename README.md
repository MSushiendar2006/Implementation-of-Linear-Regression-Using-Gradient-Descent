# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sushiendar M
RegisterNumber:212223040217 
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data, skipping the header row
data = pd.read_csv("/content/50_Startups.csv", header=0)

# Ensure we're working with numeric data
data = data.select_dtypes(include=[np.number])

# Plot initial data
plt.figure(figsize=(10, 6))
plt.scatter(data.iloc[:, 0], data.iloc[:, -1])
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[-1])
plt.title("Data Visualization")
plt.show()

def feature_normalize(X):
    """
    Normalize the features to have zero mean and unit variance
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

def computeCost(X, y, theta):
    """
    Compute the cost function for linear regression.
    """
    m = len(y)
    h = X.dot(theta)
    square_err = (h - y) ** 2
    return 1 / (2 * m) * np.sum(square_err)

# Prepare data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)
m = X.shape[0]

# Normalize features
X_norm, X_mean, X_std = feature_normalize(X)

# Add intercept term to X
X_norm = np.hstack([np.ones((m, 1)), X_norm])

# Initialize theta
theta = np.zeros((X_norm.shape[1], 1))

# Compute initial cost
initial_cost = computeCost(X_norm, y, theta)
print(f"Initial cost: {initial_cost}")

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.T, (predictions - y))
        descent = alpha * 1 / m * error
        theta -= descent
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

# Perform gradient descent
alpha = 0.01
num_iters = 1500
theta, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)

# Print the equation of the fitted line
print("Fitted line equation (with normalized features):")
print(f"y = {theta[0,0]:.2f}", end="")
for i, coef in enumerate(theta[1:], 1):
    print(f" + {coef[0]:.2f} * x{i}", end="")
print()

# Plot cost function
plt.figure(figsize=(10, 6))
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()

def predict(x, theta, mean, std):
    x_norm = (x - mean) / std
    x_norm = np.insert(x_norm, 0, 1)  # Add intercept term
    return np.dot(theta.T, x_norm)

# Make predictions
print("\nMaking predictions:")
x_input = np.array([float(input(f"Enter value for {col}: ")) for col in data.columns[:-1]])
prediction = predict(x_input, theta, X_mean, X_std)[0]
print(f"Prediction: {prediction:.2f}")

# Calculate R-squared
y_pred = X_norm.dot(theta)
ss_tot = np.sum((y - np.mean(y))**2)
ss_res = np.sum((y - y_pred)**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\nR-squared: {r_squared:.4f}")
```

## Output:
![2024-09-10](https://github.com/user-attachments/assets/56aed327-aaff-423d-bc7f-be8e74f1db1b)

![2024-09-10 (2)](https://github.com/user-attachments/assets/07bc84fb-7a17-4a49-bc83-2bed2766bc97)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
