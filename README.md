# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIKRAM K
RegisterNumber:  212222040180
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120624033/2f1716c2-f010-4101-ac18-61b9352bf44c)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120624033/6dfdb32b-336b-49f2-90eb-4940297e2580)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120624033/9ccfc97b-a4ab-44ea-b31f-6c738c26877f)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120624033/dd15b314-1a32-4638-adb4-b8fa9e73042b)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120624033/6f426799-cb37-4960-8bf7-5d2f3a198cca)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120624033/5d48811f-60e8-4eb4-9d80-48983c88ad8a)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120624033/63593b17-6c4f-4789-b436-ecc196647f2a)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120624033/7b25ea0f-3ace-440e-90f0-5ffa6fa025b9)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
