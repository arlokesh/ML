# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 20:27:55 2021

@author: AG07256
"""

#Decission Trees Implementation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:\\Users\\AG07256\\Downloads\\User_Data.csv")
print(df.head())

x=df.iloc[:,2:3]
y=df.iloc[:,4]

print(x.head())
print(y.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling

from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
X_train=st_x.fit_transform(X_train)
X_test=st_x.fit_transform(X_test)

print(X_train)
print(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
print(y_pred)

#confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print("Confusion matrix for Decession Trees is :\n",cm)

#f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(data=cm,annot=True)

#Random Forest Regrssor

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy')
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print("Confusion matrix for Randomforest is :\n",cm)

sns.heatmap(data=cm,annot=True)    