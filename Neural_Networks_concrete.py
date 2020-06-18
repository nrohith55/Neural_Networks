# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:33:00 2020

@author:Rohith
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Neural_Networks\\concrete.csv")

df.loc[df.strength <= 25,'strength']='Weak'
df.loc[df.strength != 'Weak','strength']='Strong'


X=df.iloc[:,0:8]
y=df.iloc[:,8]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

X_train=scale(X_train)
X_test=scale(X_test)
model=MLPClassifier(hidden_layer_sizes=(30,30))
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

accuracy_score(y_test,y_pred)
classification_report(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))
np.mean(y_test==y_pred)




