import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\Download\ML-DATA\emp_sal.csv")


x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#SPLIT DATA
from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test =train_test_split(x,y)
#every model give different type of result
# We we Using Same datatype with Different Model
#SVR MODEL(Support Vector REGRESSION)== SVM(SUPPORT VECTOR MODEL)-----------
#Calcuate Maximum Marzin HyperPlane
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree=2,gamma='auto',C=10.0,coef0=0.0)
svr_regressor.fit(x,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)





#KNN MODEL2 Types = Regressor Model/Classification Model
#KNN Regresseor Model-----------
#KNN MODEL(Knearest neighbor Model)--------------------
#We Calculate Distance
from sklearn.neighbors import KNeighborsRegressor
Knn_reg_model = KNeighborsRegressor(n_neighbors=2)
Knn_reg_model.fit(x,y)

Knn_reg_pred = Knn_reg_model.predict([[6.5]])
print(Knn_reg_pred)





#Decision Tree Regression ------------------------------
#We calculate Zenni index
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion="absolute_error", random_state=0)
dt_reg.fit(x, y)

dt_pred = dt_reg.predict([[6.5]])
print(dt_pred)





#Random forest Algorthim Model--------------------------
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0,n_estimators=200,max_depth=10,min_samples_split=5,min_samples_leaf=2)
rf_reg.fit(x,y)

rf_pred =rf_reg.predict([[6.5]])
print(rf_pred)




#FEATURE SCALLING TECHNIQUE-----------
# StandardScaler
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# NormalScaler
from sklearn.preprocessing import Normalizer

sc_x = Normalizer()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
#feature Engineering-------------
#feature Elimination and Sletion Techniqe----------













