# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1)Data Preparation:
Load the dataset and extract input features (X) and target variable (y).
Convert these arrays to float type and reshape as needed.

2)Data Scaling:
Apply feature scaling using StandardScaler on both input features and target variable to standardize their values.

3)Model Training (Linear Regression):
Implement a linear regression model with gradient descent, initializing parameters (theta) and iteratively updating them based on the errors between predictions and actual values.

4)Prediction:
Prepare new data, scale it, and use the trained model to make predictions based on the scaled inputs.

5)Inverse Scaling and Output:
Inverse transform the prediction to revert it back to the original scale and print the predicted value.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sushiendar M
RegisterNumber:212223040217 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta -=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv');
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([16539.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value:{pre}")
```

## Output:
![2024-09-10](https://github.com/user-attachments/assets/56aed327-aaff-423d-bc7f-be8e74f1db1b)

![2024-09-10 (2)](https://github.com/user-attachments/assets/07bc84fb-7a17-4a49-bc83-2bed2766bc97)

![2024-09-10 (3)](https://github.com/user-attachments/assets/998e0a56-327f-4fe9-ada5-bc63d22c379d)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
