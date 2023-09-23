#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings("ignore")


# In[2]:


bike=pd.read_csv("day.csv")
bike


# In[3]:


len(bike)


# In[4]:


## check the shape of data set
bike.shape


# In[5]:


bike.head()


# In[6]:


## check the descriptive information 
bike.info()


# In[7]:


## check the missing values 
bike.isnull().sum()

bike.isna().any()
# In[8]:


bike.describe()


# In[9]:


## checking is there any duplicated values in data set 
bike_dup = bike.duplicated()
bike_dup


# In[10]:


## droping the duplicated values in data set 
bike_dup = bike_dup.drop_duplicates()
bike_dup


# # Exploratory Data Analysis(EDA)

# ## Data Cleaning
# - Checking the value counts for entire data set 
# - This will be hepl to identify any unknown/junk values present in the data set 

# In[11]:


bike['yr'].value_counts()


# In[12]:


bike["hum"].value_counts()


# In[13]:


bike["instant"].value_counts()


# In[14]:


bike["registered"].value_counts()


# In[15]:


bike["cnt"].value_counts()


# In[16]:


bike["dteday"].value_counts()


# In[17]:


bike["season"].value_counts()


# In[18]:


bike["mnth"].value_counts()


# In[19]:


bike["holiday"].value_counts()


# In[20]:


bike["weekday"].value_counts()


# In[21]:


bike["workingday"].value_counts()


# In[22]:


bike["weathersit"].value_counts()


# In[23]:


bike["temp"].value_counts()


# In[24]:


bike["atemp"].value_counts()


# In[25]:


bike["windspeed"].value_counts()


# In[26]:


bike["casual"].value_counts()


# In[27]:


bike["registered"].value_counts()


# In[28]:


bike.columns


# In[29]:


## remove unwanted columns in dat set
bike=bike[['season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'cnt']]
bike


# In[30]:


bike.info() , bike.shape


# ## Dummy Variables
# - We will create DUMMY variables for 4 categorical variables 'mnth', 'weekday', 'season' & 'weathersit'.
# - Before creating dummy variables, we will have to convert them into 'category' data types.

# In[31]:


# Convert to category data type
bike['season']=bike['season'].astype('category')
bike['weathersit']=bike['weathersit'].astype('category')
bike['mnth']=bike['mnth'].astype('category')
bike['weekday']=bike['weekday'].astype('category')


# In[32]:


bike


# In[33]:


# Create Dummy variable
# Drop original variable
# Drop first dummy variable for each set of dummies created.
bike_dummy = pd.get_dummies(bike, drop_first=False)
bike_dummy


# In[34]:


bike_dummy.info() , bike_dummy.shape


# In[35]:


## Creating a box plot of the "cnt" column
plt.boxplot(bike_dummy["cnt"])
plt.title("Box plot of daily bike rental count")
plt.xlabel("Bike rentals")
plt.show()


# In[36]:


pairplot=bike_dummy[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]
sns.pairplot(pairplot, diag_kind='kde')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Preprocessing the data 
# - Standardisation 
# - Normalisation

# In[37]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[38]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[39]:


num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']
bike_dummy[num_vars] = scaler.fit_transform(bike_dummy[num_vars])
bike_dummy.head()


# In[40]:


from statsmodels.graphics.regressionplots import influence_plot 
import statsmodels.formula.api as smf


# In[41]:


bike_dummy.corr()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


# split the data into x and y
x = bike_dummy.drop(["cnt"], axis=1)
x


# In[43]:


y = bike_dummy["cnt"]
y 


# ## Spliting the data into train and test data sets 
# - we will split the data into train and test(70:30 ratio) respectively

# In[44]:


## split the data 
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=42)


# In[45]:


print('Training x Shape:', x_train.shape)
print('Training y Shape:', y_train.shape)
print('Testing x Shape:', x_test.shape)
print('Testing y Shape:', y_test.shape)


# # Model Building

# In[46]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[47]:


# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(x_train, y_train)

rfe = RFE(lm, n_features_to_select=15,step=15)             # running RFE
rfe = rfe.fit(x_train, y_train)


# In[48]:


list(zip(x_train.columns,rfe.support_,rfe.ranking_))


# In[49]:


x_train.columns[rfe.support_]


# In[50]:


#x_train.columns
col = x_train.columns[~rfe.support_]
col 


# In[51]:


# Creating X_test dataframe with RFE selected variables
x_train_rfe = x_train[col]
x_train_rfe


# In[52]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[53]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = x_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe.values, i) for i in range(x_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[54]:


import statsmodels.api as sm

# Add a constant
x_train_lm1 = sm.add_constant(x_train_rfe)

# Create a first fitted model
lr1 = sm.OLS(y_train, x_train_lm1).fit()


# In[55]:


# Check the parameters obtained

lr1.params


# In[56]:


# Print a summary of the linear regression model obtained
print(lr1.summary()) 


# # Model 2
# - Removing the variable 'atemp' based on its High p-value & High VIF

# In[57]:


x_train_new2 = x_train_rfe.drop(["mnth_3"], axis = 1)
x_train_new2


# In[58]:


x_train_new2.columns


# # VIF check 

# In[59]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = x_train_new2.columns
vif['VIF'] = [variance_inflation_factor(x_train_new2.values, i) for i in range(x_train_new2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[60]:


# Add a constant
x_train_lm2 = sm.add_constant(x_train_new2)

# Create a first fitted model
lr2 = sm.OLS(y_train, x_train_lm2).fit()
lr2


# In[61]:


# Check the parameters obtained

lr2.params


# In[62]:


# Print a summary of the linear regression model obtained
print(lr2.summary())


# # model 3

# In[63]:


x_train_new3 = x_train_new2.drop(["temp"], axis = 1)
x_train_new3


# In[64]:


x_train_new3.columns


# # VIF check

# In[65]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = x_train_new3.columns
vif['VIF'] = [variance_inflation_factor(x_train_new3.values, i) for i in range(x_train_new3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[66]:


# Add a constant
x_train_lm3 = sm.add_constant(x_train_new3)

# Create a first fitted model
lr3 = sm.OLS(y_train, x_train_lm3).fit()
# Check the parameters obtained

lr3.params


# In[67]:


# Print a summary of the linear regression model obtained
print(lr3.summary())


# # model 4

# In[68]:


x_train_new4 = x_train_new3.drop(["mnth_5","mnth_6"], axis = 1)
x_train_new4


# In[69]:


x_train_new4.columns


# # VIF 

# In[70]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = x_train_new4.columns
vif['VIF'] = [variance_inflation_factor(x_train_new4.values, i) for i in range(x_train_new4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[71]:


# Add a constant
x_train_lm4 = sm.add_constant(x_train_new4)

# Create a first fitted model
lr4 = sm.OLS(y_train, x_train_lm4).fit()
# Check the parameters obtained

lr4.params


# In[72]:


# Print a summary of the linear regression model obtained
print(lr4.summary())


# # model 5 

# In[73]:


x_train_new5 = x_train_new4.drop(["mnth_8","workingday"], axis = 1)
x_train_new5


# In[74]:


x_train_new5.columns


# # VIF 

# In[75]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = x_train_new5.columns
vif['VIF'] = [variance_inflation_factor(x_train_new5.values, i) for i in range(x_train_new5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[127]:


# Add a constant
x_train_lm5 = sm.add_constant(x_train_new5)

# Create a first fitted model
lr5 = sm.OLS(y_train, x_train_lm5).fit()
# Check the parameters obtained

lr5.params


# In[128]:


# Print a summary of the linear regression model obtained
print(lr5.summary()) 


# In[130]:


#Selecting the variables that were part of final model.
col1=x_train_new5.columns
x_test=x_test[col1]
x_test


# In[131]:


# Adding constant variable to test dataframe
x_test_lm5 = sm.add_constant(x_test)
x_test_lm5.info()


# In[132]:


# Making predictions using the final model (lr6)
y_pred = lr5.predict(x_test_lm5)
y_pred


# In[133]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:





# In[78]:


## fitting a random forest regressor model 
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(x_train, y_train)


# In[79]:


rfr.estimators_


# In[80]:


rfr.n_features_in_


# In[81]:


rfr.n_outputs_


# In[82]:


# Using KFold and Cross validation score 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[83]:


kfold = KFold(n_splits=10, random_state=42,shuffle = True)
kfold


# In[84]:


results = cross_val_score(rfr, x, y, cv=kfold)
results


# In[85]:


print(results.mean()) 


# In[86]:


# Use the forest's predict method on the test data
predictions = rfr.predict(x_test)
errors = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.') 


# In[87]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') 


# In[88]:


# Predictions on train data
pred_train = rfr.predict(x_train)
pred_train


# In[89]:


from sklearn.metrics import r2_score
print(r2_score(y_train,pred_train)) 


# In[90]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_train, pred_train)
print('Mean squared error: ', mse)


# In[91]:


# Prediction on test data
pred_test = rfr.predict(x_test)
pred_test


# In[92]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred_test)) 


# In[93]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, pred_test)
print('Mean squared error: ', mse)


# In[94]:


rmse = np.sqrt(mse)
print("RMSE:", rmse)


# In[95]:


## fitting a decision tree regressor model
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(criterion='absolute_error',max_depth=3)
dtr.fit(x_train,y_train)


# In[96]:


dtr.feature_importances_


# In[97]:


dtr.get_params


# In[98]:


dtr.n_features_in_


# In[99]:


y_pred=dtr.predict(x_test)
y_pred


# In[100]:


r2_score(y_test,y_pred)


# In[101]:


from sklearn import tree
tree.plot_tree(dtr)
plt.show()


# In[102]:


dtr1=DecisionTreeRegressor(criterion="squared_error",max_depth=3)
dtr1.fit(x_train,y_train)


# In[103]:


y_pred1=dtr1.predict(x_test)
y_pred1


# In[104]:


r2_score(y_test,y_pred1)


# In[105]:


from sklearn import tree
tree.plot_tree(dtr1)
plt.show()


# In[106]:


dtr2=DecisionTreeRegressor(criterion="friedman_mse",max_depth=3)
dtr2.fit(x_train,y_train)


# In[107]:


y_pred2=dtr2.predict(x_test)
y_pred2


# In[108]:


r2_score(y_test,y_pred1)


# In[109]:


from sklearn import tree
tree.plot_tree(dtr2)
plt.show()


# In[110]:


## ftting a ridge regresion model
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.5)
ridge.fit(x_train, y_train)


# In[111]:


y_pred = ridge.predict(x_test)
y_pred


# In[112]:


mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)


# In[113]:


r2_score(y_test,y_pred)


# In[114]:


## laso regression model fitt
from sklearn.linear_model import Lasso, ElasticNet
lasso = Lasso(alpha=0.1)
lasso.fit(x_train, y_train)


# In[115]:


lasso_pred = lasso.predict(x_test)
lasso_pred


# In[116]:


lasso_mse = mean_squared_error(y_test, lasso_pred)
print('Lasso regression MSE:', lasso_mse)


# In[117]:


r2_score(y_test,lasso_pred)


# In[118]:


## fitt elasticnet model
from sklearn.linear_model import ElasticNet
enet = ElasticNet(alpha=0.5, l1_ratio=0.5)
enet.fit(x_train, y_train)


# In[119]:


enet_pred = enet.predict(x_test)
enet_pred


# In[120]:


r2_score(y_test,enet_pred)


# # SVM using regressor

# In[121]:


# model fitting or building 
from sklearn.svm import SVR
model_rbf = SVR(kernel = "rbf",gamma=0.1,C=1.0)
model_rbf


# In[122]:


model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)
pred_test_rbf


# In[123]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred_test))


# In[124]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, pred_test_rbf)
print('Mean squared error: ', mse)


# In[125]:


rmse = np.sqrt(mse)
print("RMSE:", rmse)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




