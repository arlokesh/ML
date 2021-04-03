# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 21:32:48 2021

@author: Lokesh Rudraradhya
"""

import pandas as pd
import numpy as np
from sklearn import datasets

#import cancer dataset into pandas dataframe

cancer=datasets.load_breast_cancer()
data=np.c_[cancer.data,cancer.target]
columns=np.append(cancer.feature_names,["target"])
df=pd.DataFrame(data,columns=columns)
print(df.head())
print(df.describe())

print(df)

print("Features",cancer.feature_names)
print("Labels",cancer.target_names)

print(cancer.data.shape)
print(cancer.data[0:5])

print(cancer.target)

from sklearn.model_selection import train_test_split
#70% training 30%test data
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,test_size=0.30,random_state=109)

from sklearn import svm

clf=svm.SVC(kernel='linear')
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision is :",metrics.precision_score(y_test, y_pred))
print("Recall is : ",metrics.recall_score(y_test, y_pred))

