# Regulazitation Technique---------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score


data =pd.read_csv(r'D:\Ml-MACHINE LEARNING DATA\car-mpg.csv')
data

data=data.drop(['car_name'],axis=1)

#replace original values
data['origin']=data['origin'].replace({1:'america',2:'europe',3:'asia'})

# convert origin to dummy variable

data=pd.get_dummies(data,columns=['origin'],dtype=int)

#replace '?' with nan

data=data.replace('?',np.nan)

# converted columns to numeric where possible
data=data.apply(pd.to_numeric,errors='ignore')

#fill missing values with median
data=data.apply(lambda x: x.fillna(x.median()) if x.dtype != 'object' else x)


X =data.drop(['mpg'],axis=1) #indipendent variables
y = data[['mpg']]



#Scalling the Data
X_s = preprocessing.scale(X)
X_s = pd.DataFrame(X_s, columns = X.columns) #converting scaled data into dataframe
y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s, columns = y.columns) #ideally train, test data should be in columns


#Split data
X_train, X_test, y_train,y_test = train_test_split(X_s, y_s, test_size = 0.30, random_state = 1)
X_train.shape



# 2.a Simple Linear Model
#Fit simple linear model and find coefficients
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
for idx, col_name in enumerate(X_train.columns):
print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx])):
    intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))



## 2.b Regularized Ridge Regression
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff
ridge_model = Ridge(alpha = 0.3)
ridge_model.fit(X_train, y_train)
print('Ridge model coef: {}'.format(ridge_model.coef_))



## 2.c Regularized Lasso Regression
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff
lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(X_train, y_train)
print('Lasso model coef: {}'.format(lasso_model.coef_))




# 3. Score Comparison
#Model score - r^2 or coeff of determinant
#r^2 = 1-(RSS/TSS) = Regression error/TSS
#Simple Linear Model
print(regression_model.score(X_train, y_train))
print(regression_model.score(X_test, y_test))
print('*************************')
#Ridge
print(ridge_model.score(X_train, y_train))
print(ridge_model.score(X_test, y_test))
print('*************************')
#Lasso
print(lasso_model.score(X_train, y_train))
print(lasso_model.score(X_test, y_test))

