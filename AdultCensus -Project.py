#ADULT CENCUS PROJECT
#-------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =pd.read_csv(r"D:\Download\ML-DATA\adult\adult.csv")

#Data cleaning
dataset[dataset =='?'] =np.nan


columns = ['workclass', 'occupation', 'native.country']
dataset[columns] = dataset[columns].fillna(dataset[columns].mode().iloc[0])

x = dataset.drop(['income'], axis=1)
y = dataset['income']


#split data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=0)



from sklearn import preprocessing
categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        x_train[feature] = le.fit_transform(x_train[feature])
        x_test[feature] = le.transform(x_test[feature])
        
        
        
#scalling technique-------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns = x.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns = x.columns)



#model built--------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Logistic Regression accuracy score with all the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(x_train)
pca.explained_variance_ratio_


#Traning the SVM model on the traning set
#------------------------------------------------------
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svm_model = SVC(kernel='rbf',degree=3,gamma='scale')
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)
ac_svm = accuracy_score(y_test, y_pred_svm)
print(ac_svm)



# Train KNN Model (FIXED)
#-------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)
ac_knn = accuracy_score(y_test, y_pred_knn)
print(ac_knn)


from sklearn.metrics import classification_report
cr =classification_report(y_test,y_pred)
print(cr)



#visualization
#----------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlim(0, 14)  
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()






