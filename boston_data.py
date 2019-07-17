
# coding: utf-8

# In[334]:


# First let's import all the packages that wer will need in order to analyze and  visualize our data

import seaborn as sns
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

boston_dataset = load_boston()


# In[341]:


# We will try to make a Linear Regression model on the Boston data.
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
boston.head()

# In order to know more about the data's features we can use the DESCR:
print(boston_dataset.DESCR)


# In[340]:


# The correlation matrix will help us see if there  are any strong correlations between the features and the target:
correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
# We can see that there is a strong correlation (0.7) between the variables MEDV and RM. 
# We will try to predict the median price (MEDV) using the variable RM  (average number of rooms).


# In[373]:


# Splitting our data into train and test sets:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = boston['RM']
X = X.values.reshape(-1, 1)
y  = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=23)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Starting our linear regression model:

regr = LinearRegression()
plt.scatter(X_train, Y_train)
regr.fit(X_train, Y_train)
y_predicted = regr.predict(X_train)
print(regr.coef_)
print(regr.intercept_)
y_line = [9.14438088*x-34.75540260183401 for x in X]
plt.plot(X, y_line, c='r')


# In[377]:


#  Let's evaluate our model using RMSE:
Y_train_predict = regr.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("The model performance for training set")
print("-----------------")
print('RMSE is {}'.format(rmse))
print("\n")

# model evaluation for testing set
Y_test_predict = regr.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, Y_test_predict)))

print("The model performance for testing set")
print("-----------------")
print('RMSE is {}'.format(rmse))

