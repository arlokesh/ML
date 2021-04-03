# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 20:34:50 2021

@author: AG07256
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#df = pd.read_csv("C:\\worked\\Data_science\\Mtech\\Sem-4\\DeepLearning\\kaagle\\archive\\uci-secom.csv")
df = pd.read_csv("C:\\Users\\AG07256\\Downloads\\BostonHousing.csv")

x=df.iloc[:,0:13]
print(x.head())
y=df.iloc[:,-1]
print(y.head())

corel=df.corr().round(2)
print(corel)
f,ax=plt.subplots(figsize=(30,30))
corelation_matrix=corel
sns.heatmap(data=corelation_matrix,annot=True)

df2=corelation_matrix['medv']
df3=df2[abs(df2) >= 0.6]
print(df3)

x=df[['rm','lstat']]
y=df['medv']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lm=LinearRegression()
lm.fit(X_train,y_train)

print(lm.coef_)
print(lm.intercept_)


from sklearn.metrics import r2_score
output=4.58*0.6+2.73
print(output)

y=pd.DataFrame({'lstat':[12.8],'rm':[6.3]})
print(y)
output_predict=lm.predict(y)
print(output_predict)

#rmse
import numpy as np
from sklearn.metrics import r2_score

y_train_predict=lm.predict(X_train)
rmse=(np.sqrt(mean_squared_error(y_train,y_train_predict)))
r2=r2_score(y_train,y_train_predict)
print('rmse is {}'.format(rmse))
print('r2 is {}'.format(r2))
print("\n")

print("===Rmse for y_test and y_predicted===")

y_test_predict=lm.predict(X_test)
rmse=(np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2=r2_score(y_train, y_train_predict)
print('rmse is {}'.format(rmse))
print('r2 is {}'.format(r2))

#dumping the model to pickle

import pickle
filename='boston_lm.pkl'
pickle.dump(lm,open(filename,'wb'))

