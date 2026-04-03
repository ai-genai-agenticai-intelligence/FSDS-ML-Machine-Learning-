# k-Fold Cross Validation--------------------------

#Random Forest xGboost-------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset-------------------
dataset = pd.read_csv(r"D:\Ml-MACHINE LEARNING DATA\Churn_Modelling.csv")
x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)


#Encoding categorical data
#Label Encoding the 'Gender' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] =le.fit_transform(x[:,2])
print(x)


'''One Hot Encoding the 'Geography' column'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)


''' Spliting the dataset into the Training set and Test set'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.20,random_state=0)


''' in Random Forest Scalling Technique are not required
Training XGBoost on the Training set'''
from xgboost import XGBClassifier
classifier =XGBClassifier(random_state=0)
classifier.fit(x_train ,y_train)


'''Predicting the test set result'''
y_pred = classifier.predict(x_test)


''' Marginal matrix'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias=classifier.score(x_train, y_train)
bias

variance = classifier.score(x_test, y_test)
variance


# you can add implement auc & roc 


''' Applying k-Fold Cross Validation'''
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))









