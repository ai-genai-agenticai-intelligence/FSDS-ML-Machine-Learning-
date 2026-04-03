#multiple linear regression model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"D:\Download\ML-DATA\Investment.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]

x = pd.get_dummies(x,dtype=int)

#split data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=0)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)


y_pred = regressor.predict(x_test)


#CONSTANT [c=y-mx]
c_intercept =regressor.intercept_
print(c_intercept)

#()
m_coef = regressor.coef_
print(m_coef)


x = np.append(arr=np.full((50,1), 42467).astype(int), values=x, axis=1)
x = np.append(arr=np.full((50,1), 42467).astype(int), values=x, axis=1)




#STATSMODEL API------
#FEATURE ELIMINATION OR FEATURE SELECTION-----
#RECURSIVE FEATURE ELEMINATION----------(RFE)
#BACKWORD ELIMINATION
#FEATURE SELECTIO TECHNIQUE IN MACHINE LEARNING----
#p>value remove value <p value add value
#we reject the null hypithesis that means we eliminated the attributed---------


import statsmodels.api as sm
x_opt = x[:, [0,1,2,3,4,5]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()



import statsmodels.api as sm
x_opt = x[:, [0,1,2,3,5]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()



import statsmodels.api as sm
x_opt = x[:, [0,1,2,3]]
# Ordinary Least Squares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()



