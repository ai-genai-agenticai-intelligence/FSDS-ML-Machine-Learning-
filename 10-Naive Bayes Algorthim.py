#Naive Bayis Algorthim-----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset-------
dataset=pd.read_csv(r"D:\Download\ML-DATA\logit classification.csv")
x = dataset.iloc[:,[2,3]].values #independent variable
y = dataset.iloc[:,-1].values #dependent variable


#split data----
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.20,random_state=0)



#Feature Scaling------------------------
'''
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
x_train=sc.fit_transform(x_train)
x_test =sc.transform(x_test)
'''

from sklearn.preprocessing import Normalizer
sc=Normalizer()
x_train=sc.fit_transform(x_train)
x_test =sc.transform(x_test)



#Traning the SVM model on the traning set
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train,y_train)



'''
#REMEMBER
-----------------------
#when you Apply GaussianNB that time SCALLING TECHNIQUE not required---------
#Training the Naive Bayes model on the Traning set------------
'''
'''
from sklearn.naive_bayes import BernoulliNB
classifier=BernoulliNB()
classifier.fit(x_train ,y_train)
'''

'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)
'''

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)


# Predicting the Test set result
y_pred =classifier.predict(x_test)


#Making cofusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)



#This is to get the Models Accuracy
from sklearn.metrics import accuracy_score
ac =accuracy_score(y_test, y_pred)
print(ac)



#classification---------
from sklearn.metrics import classification_report
cr =classification_report(y_test, y_pred)
print(cr)



#bias-----------
bias =classifier.score(x_train, y_train)
print(bias)


#variance----------
variance =classifier.score(x_test,y_test)
print(variance)
