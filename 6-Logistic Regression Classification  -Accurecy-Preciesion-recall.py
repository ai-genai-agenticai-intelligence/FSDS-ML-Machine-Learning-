import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\Download\ML-DATA\logit classification.csv")


#devided data----
x = dataset.iloc[:,[2,3]].values #independent variable
y = dataset.iloc[:,-1].values #dependent variable



#split data----------
#here we 25% split data------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=41)



#Scalling technique------
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
x_train=sc.fit_transform(x_train)
x_test =sc.transform(x_test)


'''
from sklearn.preprocessing import Normalizer
sc =Normalizer()
x_train=sc.fit_transform(x_train)
x_test =sc.transform(x_test)
'''



#Logistic Regression-----------------
from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression()
classifier.fit(x_train,y_train)



#i want to how many record are miss classification -----
y_pred = classifier.predict(x_test)





#------------------------CONFUSIN MATRIX---------------------------------------


#how many data missclassification -----
#confusion matrix build on y_test,y_pred alwaya-------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


#Acurecy--
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



#------------------------FUTURE DATA--------------------------------------


dataset1 = pd.read_csv(r"D:\Download\ML-DATA\Future prediction1.csv")

d2 = dataset1.copy()


dataset1=dataset1.iloc[:,[2,3]].values


#Scale the data------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M =sc.fit_transform(dataset1)


y_pred1= pd.DataFrame()

#model predicted future data
d2['y_pred1'] =classifier.predict(M)

d2.to_csv('final1.csv')


'''
#to get the path 
import os
os.getcwd()
'''

#AUC/ROC-------------------------
from sklearn.metrics import roc_auc_score ,roc_curve
y_pred_prob = classifier.predict_proba(x_test)[:,1]

auc_score =roc_auc_score(y_test,y_pred_prob)
auc_score

fpr ,tpr,thershold =roc_curve(y_test ,y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Random line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()












