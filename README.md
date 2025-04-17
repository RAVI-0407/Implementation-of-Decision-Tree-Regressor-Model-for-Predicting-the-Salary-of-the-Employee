# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program & Output:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RAVIPRASATH K
RegisterNumber:  212224230225
*/
```
```
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("drive/MyDrive/ML/Salary.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/399c2015-46f5-493f-be3a-7536b112d137)
```
data.info()
```
![image](https://github.com/user-attachments/assets/098388e0-9a12-4520-ab3d-7c90ea67b275)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/a3372068-9d05-4704-9f7b-c9693e4da3a7)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
![image](https://github.com/user-attachments/assets/b6d27983-3eec-4a31-8ba1-b171c6d83737)
```
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
```
![image](https://github.com/user-attachments/assets/60038868-9f58-47e0-9b4c-e03c71f5a00c)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/6173abfe-8ca4-4396-bf70-0f5c59876f6e)
```
from sklearn import metrics
r2=metrics.r2_score(y_test,y_pred)
r2
```
![image](https://github.com/user-attachments/assets/922cc946-5ae8-460d-82c5-79e58a6b42dc)
```
dt.predict([[5,6]])
```
![image](https://github.com/user-attachments/assets/fba1fdb3-2813-42fe-8f29-d70e1680ae82)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
