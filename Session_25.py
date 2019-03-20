# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:43:18 2019

@author: CNsasi
"""

# XGBoost

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
train_set = pd.read_csv('adult.data.txt', header = None)
test_set = pd.read_csv('adult.test.txt', skiprows = 1, header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels

#Let's concatenate our dataset
dataset=pd.concat([train_set,test_set], axis=0)

#Formatting wage_class column
dataset["wage_class"]= dataset["wage_class"].replace([' <=50K.',' <=50K'], '<=50K')
dataset["wage_class"]= dataset["wage_class"].replace([' >50K',' >50K.'], '>50K')

#Split our dependant and independant feature
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,14]

#Encoding categorical independant feature
workclass=pd.get_dummies(X['workclass'], drop_first=True)
education=pd.get_dummies(X['education'], drop_first=True)
marital_status=pd.get_dummies(X['marital_status'], drop_first=True)
occupation=pd.get_dummies(X['occupation'], drop_first=True)
relationship=pd.get_dummies(X['relationship'], drop_first=True)
race=pd.get_dummies(X['race'], drop_first=True)
sex=pd.get_dummies(X['sex'], drop_first=True)
native_country=pd.get_dummies(X['native_country'], drop_first=True)

#Encoding categorical dependant feature
wage_class=pd.get_dummies(y, drop_first=True)
y=wage_class

#Drop the columns
X.drop(['workclass','education','marital_status','occupation','relationship','race','sex','native_country'],axis=1,inplace=True)

#concat the dummy variables
X=pd.concat([X,workclass,education,marital_status,occupation,relationship,race,sex,native_country], axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Fitting XGBoost to the Training set
from  xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
