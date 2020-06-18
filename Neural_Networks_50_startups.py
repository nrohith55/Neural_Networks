# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:21:08 2020

@author: Rohith
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Neural_Networks\\50_Startups.csv")

df=df.iloc[:,[4,0,1,2,3]]
df.isnull().sum()
df=pd.get_dummies(df,columns=['State'],drop_first=True)

df.loc[df.Profit <= 100000,'Profit']= 0
df.loc[df.Profit != 0,'Profit'] = 1


X=df.iloc[:,1:6]
y=df.iloc[:,0]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

X_train_new=scale(X_train)
print(X_train)
X_test=scale(X_test)
X_test
X_train=scale(X_train)

help(MLPClassifier)
model=MLPClassifier(hidden_layer_sizes=(30,))
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)



























