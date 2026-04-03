import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"D:/Download/ML-DATA/Salary_Data.csv")

x=dataset.iloc[:,:1]
y=dataset.iloc[:,-1]

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)


y_pred = regressor.predict(x_test)

comparision = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
print(comparision)


plt.scatter(x_test,y_test,color='Red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary of based on experience')
plt.xlabel('Experience')
plt.ylable('Salary')
plt.show()



#predict future
m_coef = regressor.coef_
print(m_coef)


c_intercept =regressor.intercept_
print(c_intercept)

y_12 = m_coef *12 + c_intercept
print(y_12)


y_20 = m_coef *20 + c_intercept
print(y_20)



#BIAS
bias=regressor.score(x_train,y_train)
print(bias)
#VARIANCE
variance=regressor.score(x_test,y_test)
print(variance)
# bias-94,variance-98 (best fitted model)






#[D-4-3-2026]

# STATS IMPLITATION TO THIS CODE
#MEAN.
dataset.mean()
dataset['Salary'].mean()


#MEDIAN
dataset.median()
dataset['Salary'].median()


#MODE
dataset.mode()
dataset['Salary'].mode()


#VARIANCE
dataset.var()
dataset['Salary'].var()


#STANDARD DEVIATION
dataset.std()
dataset['Salary'].std()




#COEFICIENT OF VARIATION
from scipy.stats import variation

variation(dataset.values)
variation(dataset['Salary'])


#CORELATION
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])


#SKEWNESS
dataset.skew()
dataset['Salary'].skew()


#STANDARD ERROR
from scipy.stats import stats
dataset.apply(stats.zscore)



#SSR(sum of square regressior)
y_mean =np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

#r2
r_square=1-SSR/SST
print(r_square)




