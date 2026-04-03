#Polynomial Features------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv(r"D:\Download\ML-DATA\emp_sal.csv")

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)


lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

#it is not best fit moddle 
#because Actual Predict are not equal
plt.scatter(x,y ,color='red')
plt.plot(x ,lin_reg.predict(x),color ='blue')
plt.title('Linear Regression Graph')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()



#degree increase ony indipendent variable
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly =poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)

lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly,y)

#Non linear regression visualization
#EXAMPLE--is it best fit moddle--- no Actual and predict are not equal...
plt.scatter(x,y ,color='red')
plt.plot(x ,lin_reg_2.predict( poly_reg.fit_transform(x)),color ='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred





#---------------------SVR model----------------------------------
from sklearn.svm import SVR
svr_reg =SVR(kernel='poly',degree=2,gamma='auto',C=10.0,coef0=0.0)
svr_reg.fit(x,y)

svr_pred = svr_reg.predict([[6.5]])
svr_pred




#-------------------KNN model------------------------------------
from sklearn.neighbors import KNeighborsRegressor
Knn_reg_model = KNeighborsRegressor(n_neighbors=2)
Knn_reg_model.fit(x,y)

Knn_reg_pred = Knn_reg_model.predict([[6.5]])
print(Knn_reg_pred)




#----------------DecisionTree------------------------------------
from sklearn.tree import DecisionTreeRegressor
dt_reg_model=DecisionTreeRegressor(criterion="absolute_error" ,random_state=0)
dt_reg_model.fit(x,y)

dt_reg_model=dt_reg_model.predict([[6.5]])
dt_reg_model




#------------------Randomforest----------------------------------
from sklearn.ensemble import RandomForestRegressor
rf_reg_model=RandomForestRegressor(random_state=0,n_estimators=200,max_depth=10,min_samples_split=5,min_samples_leaf=2)
rf_reg_model.fit(x,y)

rf_reg_model=rf_reg_model.predict([[6.5]])
rf_reg_model












