# -*- coding: utf-8 -*-
"""
Created on Tue May 19 22:42:16 2020

@author: Rohith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Neural_Networks\\wbcd.csv")

df.loc[df.diagnosis=='B','diagnosis']=1
df.loc[df.diagnosis=='M','diagnosis']=0

X=df.iloc[:,2:32]
y=df.iloc[:,1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.preprocessing import scale

X_test=scale(X_test)
X_train=scale(X_train)

help(MLPClassifier)
model=MLPClassifier(hidden_layer_sizes=(100))
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)






























































































