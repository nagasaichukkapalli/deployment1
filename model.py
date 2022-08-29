#!/usr/bin/env python
# coding: utf-8

# In[92]:


# Importing required modules and loading dataset
import numpy as np
import pandas as pd
from datetime import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid",{'grid.linestyle': '--'})


from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[93]:



df=pd.read_csv("SeoulBikeData.csv",encoding= 'unicode_escape')

df.head(10)


# In[94]:


# Getting shape of the data
df.shape


# In[95]:


# Getting details about all the features present in the dataset
df.info()


# # **Processing the dataset**

# In[96]:


#Using Lambda function to strip date from string to Datetime format so to retrieve d,m,y
df['Date'] = df['Date'].apply(lambda x:dt.strptime(x, "%d/%m/%Y"))
df['Day'] = df['Date'].dt.day_name()
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year


# In[97]:



df


# In[98]:


# Lets add a new column named Weekend with binary values, indicating 1 for weekend and 0 for a weekday

df['Weekend']=df['Day'].apply(lambda x : 1 if x=='Saturday' or x=='Sunday' else 0 )


# In[99]:


df.head()


# In[100]:


#Dropping columns in vertical axis
df=df.drop(columns=['Date','Day','Year'],axis=1)


# * Here we can assume that Bikes were rented more when there is no holiday and very less as on Holidays

# # **Analyze numerical Variables**

# In[101]:


#Storing all the numeric features in a variable list

numeric_features =['Rented Bike Count', 'Temperature(°C)', 'Humidity(%)',
       'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)',
       'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']


# In[102]:


# Ordinal encoding

df['Functioning Day']=df['Functioning Day'].map({'Yes':1,'No':0})
df['Holiday']=df['Holiday'].map({'No Holiday':0,'Holiday':1})


# In[103]:


df


# In[104]:


# One Hot Encoding

df_seasons=pd.get_dummies( df['Seasons'] )
df_month=pd.get_dummies( df['Month'] , prefix = 'Month')
df_hour=pd.get_dummies( df['Hour'] ,prefix = 'Hour' )


# In[105]:


# Join one hot encoded columns

df=df.join([df_seasons,df_month,df_hour])


# In[106]:


df=df.drop(columns = ['Hour', 'Seasons' ,'Month'])


# In[107]:


df.columns


# In[108]:


# Applying square root to Rented Bike Count column

df['Rented Bike Count']=np.sqrt(df['Rented Bike Count'])


# In[109]:


# Creating copy of data

data= df.copy()


# In[114]:


# Create the data of dependent and independent variables

y = data['Rented Bike Count']
X = data.drop(['Rented Bike Count', 'Temperature(°C)',
       'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)',
       'Rainfall(mm)', 'Snowfall (cm)', 'Holiday',
       'Weekend', 'Autumn', 'Spring', 'Summer',
       'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
       'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_12',
       'Hour_0', 'Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6',
       'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12',
       'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17',
       'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23'], axis=1)


# In[115]:


X.columns


# In[116]:


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12)
print(X_train.shape)
print(X_test.shape)


# In[117]:


# Transforming data

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # **Linear Regression**

# In[118]:


# Fitting onto Linear regression Model 

reg= LinearRegression().fit(X_train, y_train)


# In[119]:


# Getting the X_train and X-test value

y_pred_train=reg.predict(X_train)
y_pred_test=reg.predict(X_test)


# Evluation Matrix for Linear Regression

# In[120]:


# Calculate MSE, MAE, R2 for training data


MSEl = mean_squared_error((y_train), (y_pred_train))
MAEl= mean_absolute_error(y_train, y_pred_train)
r2l = r2_score(y_train, y_pred_train)


# In[121]:


# Calculate MSE, MAE, R2 for testing data


MSEtestl = mean_squared_error((y_test), (y_pred_test))
MAEtestl= mean_absolute_error(y_test, y_pred_test)
r2testl = r2_score(y_test, y_pred_test)


# In[122]:


# Printing Errors

print('Training Errors\nMSE:', MSEl , '\nMAE:' , MAEl , '\nR2:',round((r2l),3))
print('\n\nTesting Errors\nMSE:', MSEtestl , '\nMAE:' , MAEtestl , '\nR2:',round((r2testl),3))


# # **Gradient Boost with Gridsearch**

# In[123]:


param_dict = {'n_estimators' : [50,80,100],
              'max_depth' : [4,6,8,10],
              'min_samples_split' : [50,80,100],
              'min_samples_leaf' : [40,50]}


# In[124]:


gb = GradientBoostingRegressor()


# In[125]:


# Grid search
gb_grid = GridSearchCV(estimator=gb,
                       param_grid = param_dict,
                       cv = 5, verbose=0)


# In[126]:


gb_grid.fit(X_train,y_train)


# In[127]:


gb_grid.best_estimator_


# In[128]:


# Putting Best possible paramteres into model

gb_optimal_model = gb_grid.best_estimator_


# In[129]:


# Making predictions on train and test data

y_pred_traingbg = gb_optimal_model.predict(X_train)
y_predgbg= gb_optimal_model.predict(X_test)


# In[130]:


# Calculate MSE, MAE, R2 for training data


MSEGBG = mean_squared_error((y_train), (y_pred_traingbg))
MAEGBG = mean_absolute_error(y_train, y_pred_traingbg)
r2GBG = r2_score(y_train, y_pred_traingbg)


# In[131]:


# Calculate MSE, MAE, R2 for testing data


MSEtestGBG = mean_squared_error((y_test), (y_predgbg))
MAEtestGBG = mean_absolute_error(y_test, y_predgbg)
r2testGBG = r2_score(y_test, y_predgbg)


# In[132]:


# Printing Errors

print('Training Errors\nMSE:', MSEGBG , '\nMAE:' , MAEGBG , '\nR2:',round((r2GBG),3))
print('\n\nTesting Errors\nMSE:', MSEtestGBG , '\nMAE:' , MAEtestGBG , '\nR2:',round((r2testGBG),3))


# In[133]:


X.columns


# In[134]:


import pickle


# In[135]:


pickle.dump(gb_grid, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model)


# In[ ]:



