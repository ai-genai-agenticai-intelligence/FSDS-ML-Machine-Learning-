'''GRIDSEARCH CV'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset
dataset =pd.read_csv(r"D:\Ml-MACHINE LEARNING DATA\Social_Network_Ads.csv")
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values


#Sclling technique
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X=sc.fit_transform(X)

#split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,train_size=0.25,random_state=0)


# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train)


#predict model
y_pred =classifier.predict(X_test)


#making confusion matrix
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_pred,y_test)
cm

from sklearn.metrics import accuracy_score
ac =accuracy_score(y_pred,y_test)
print(ac)

bias =classifier.score(X_train,y_train)
bias

variance =classifier.score(X_test,y_test)



#Apply k1-folder cross validation
#Also it will retuen mean of accuracy combine all dataset
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier,  X = X_train,y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracy.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracy.std()*100))




# Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           )
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)



#Applying Random Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

parameters_random = {
    'C': uniform(1, 1000),           # continuous values between 1 and 1000
    'kernel': ['linear', 'rbf'],
    'gamma': uniform(0.01, 1)        # for rbf kernel
}

random_search = RandomizedSearchCV(
    estimator = classifier,
    param_distributions = parameters_random,
    n_iter = 50,                     # number of random combinations
    scoring = 'accuracy',
    cv = 10,
    random_state = 0,
    n_jobs = -1
)

random_search = random_search.fit(X_train, y_train)

best_accuracy_rs = random_search.best_score_
best_parameters_rs = random_search.best_params_

print("Best Accuracy using Random Search: {:.2f} %".format(best_accuracy_rs*100))
print("Best Parameters using Random Search:", best_parameters_rs)


'''Visualization'''

# Visualising the Training set results-----------------------
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# Visualising the  Testing  set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



