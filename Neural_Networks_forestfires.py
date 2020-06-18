# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:53:49 2020

@author: Rohith
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Neural_Networks\\forestfires.csv")

df=df.iloc[:,[30,6,7,8,9]]
df.shape
df.describe()
X=df.iloc[:,1:5]
y=df.iloc[:,0]

df.size_category.value_counts()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

X_train=scale(X_train)
X_test=scale(X_test)

model=MLPClassifier(hidden_layer_sizes=(100,))
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)








































