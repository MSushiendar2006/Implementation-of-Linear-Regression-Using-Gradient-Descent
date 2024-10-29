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
data=pd.read_csv('/content/50_Startups.csv');
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
![Screenshot 2024-10-29 123952](https://github.com/user-attachments/assets/1c3e7d22-8727-4197-bf3c-69d9659aa57c)

![image](https://github.com/user-attachments/assets/5a533466-d604-4519-9e46-385c4cc531d3)

![image](https://github.com/user-attachments/assets/4aadde6c-dfd8-4b39-8ddf-96cb5c1e6b95)

![Screenshot 2024-10-29 124252](https://github.com/user-attachments/assets/ae0b42b6-1174-47b1-aea7-7f497e5b1c03)

![image](https://github.com/user-attachments/assets/3a6e3f76-114f-46e7-a2ab-2b726f384af9)

![image](https://github.com/user-attachments/assets/1a75fd48-0537-4597-8e7b-7a37f5298f59)

![image](https://github.com/user-attachments/assets/56854b2a-3bc6-4688-a645-facd0d1f4877)

![image](https://github.com/user-attachments/assets/4b06bf08-6852-4360-a95e-2b57a3ba842f)

![image](https://github.com/user-attachments/assets/32b9051f-f53b-424a-a294-1cfbe6ac9aa7)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
