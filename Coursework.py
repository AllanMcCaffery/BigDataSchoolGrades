# -*- coding: utf-8 -*-
"""
@author: Allan McCaffery S1632551
Big Data Coursework
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Reading csv data separated by semi-colon
df = pd.read_csv('student-mat.csv', sep =';')

# Removing G1 and G2 grade values from dataframe as they are correlated with target value
del df['G1']
del df['G2']

#Replace missing data in target value of 0 with dataset mean value
df['G3'].replace(0, 10, inplace = True)

print (df.shape) # RETURNS COUNT OF INSTANCES AND ATTRIBUTES

print(df.dtypes) # Displays the name and datatypes of Attributes

print(df.head(5)) # Displays Top 5 rows of dataset

#Check for null values in dataset
print(df.isnull().sum())
print(df.describe())
#
#Check for duplicate records in dataset
duplicate_rows_df = df[df.duplicated()]
print("The number of duplicated rows is: ", duplicate_rows_df.shape)
#
#Split all non numeric cariables into individual variable column values
df = pd.get_dummies(df, columns=['school', 'sex','guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason'])

del df['Medu']
del df['guardian_mother']
del df['Dalc']
del df['Fjob_services']

corr = df.corr()
#print("Correlation Matrix: ", corr.to_string())
#
plt.figure(figsize=(40, 20))
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap="RdBu", annot=True)
#
#
#set all variables except target variable as input to variable X
X = df.drop(['G3'], axis=1)

#set output G3 to variabley 
y = df['G3'].values
print(y[0:25])
#
plt.figure(figsize=(20,10))
sns.distplot(y)
#
##Create the training and testing data. Training data accounts for 80% and testing data 20% 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Calculate the instances used in training and testing
print(" ")
print("Training Data = ", X_train.shape)
print("Test Data = ", X_test.shape)
print(" ")
#
regressor = LinearRegression(fit_intercept=False)
#
regressor.fit(X_train, y_train)
#
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
#
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#
df1 = df.head(20)
print(df1)
#
df1.plot(kind='bar', figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.show()
#
##evaluation
print(' ')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

