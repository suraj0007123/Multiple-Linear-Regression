# Multilinear Regression
import pandas as pd
import numpy as np
import seaborn as sns

# loading the data
startup = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\multiple linear regression\Datasets_MLR\50_Startups.csv")
startup.describe()

startup['State'].replace(['New York','California','Florida'],[0,1,2],inplace=True)

startup.columns

startup.info()

startup.columns="RD_Spend","Administration","Marketing_Spend","State","Profit"

startup.columns

startup.info()

#Graphical Representation
import matplotlib.pyplot as plt

plt.bar(height=startup.RD_Spend,x=np.arange(1,51,1)) 
plt.hist(startup.RD_Spend) 
sns.boxplot(startup.RD_Spend)

plt.bar(height=startup.Administration, x=np.arange(1,51,1))
plt.hist(startup.Administration)
sns.boxplot(startup.Administration)

plt.bar(height=startup.Marketing_Spend, x=np.arange(1,51,1))
plt.hist(startup.Marketing_Spend)
sns.boxplot(startup.Marketing_Spend)

plt.bar(height=startup.State, x=np.arange(1,51,1))
plt.hist(startup.State)
sns.boxplot(startup.State)

plt.bar(height=startup.Profit, x=np.arange(1,51,1))
plt.hist(startup.Profit)
sns.boxplot(startup.Profit)


# Jointplot
import seaborn as sns
sns.jointplot(x=startup['RD_Spend'], y=startup['Profit'])

sns.jointplot(x=startup['Administration'], y=startup['Profit'])

sns.jointplot(x=startup['Marketing_Spend'], y=startup['Profit'])

# Countplot
plt.figure(1, figsize=(30, 10))
sns.countplot(startup['Profit'])

plt.figure(1, figsize=(30, 10))
sns.countplot(startup['RD_Spend'])

plt.figure(1, figsize=(30, 10))
sns.countplot(startup['Administration'])

plt.figure(1, figsize=(30, 10))
sns.countplot(startup['Marketing_Spend'])


# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(startup.Profit, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startup.iloc[:,:])

# Correlation matrix 
startup.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
ml1 = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + State',data=startup).fit() # regression model
ml1.summary()


# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)

startup_new = startup.drop(startup.index[[45]])

ml_new = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + State',data=startup_new).fit() # regression model
ml_new.summary()
    

#if Coefficients are insignificant then check the model for individual inputs
m_RD=smf.ols('Profit ~ RD_Spend',data=startup).fit()
m_RD.summary()

m_ADM=smf.ols('Profit ~ Administration',data=startup).fit()
m_ADM.summary()

m_MRSPEND=smf.ols('Profit ~ Marketing_Spend',data=startup).fit()
m_MRSPEND.summary()

#Check combinatin of input variables
m_comb=smf.ols('Profit ~ RD_Spend+Administration+Marketing_Spend',data=startup).fit()
m_comb.summary() #one of independent variable p value not statistically significant

# calculating VIF's values of independent variables
rsq_RD_Spend = smf.ols('RD_Spend ~ Administration + Marketing_Spend + State',data=startup).fit().rsquared  
vif_RD_Spend = 1/(1-rsq_RD_Spend)

rsq_Administration = smf.ols('Administration ~ RD_Spend + Marketing_Spend + State',data=startup).fit().rsquared  
vif_Administration = 1/(1-rsq_Administration)


rsq_Marketing_Spend = smf.ols('Marketing_Spend ~ RD_Spend + Administration + State',data=startup).fit().rsquared  
vif_Marketing_Spend = 1/(1-rsq_Marketing_Spend)

rsq_State = smf.ols('State ~ RD_Spend + Administration + Marketing_Spend',data=startup).fit().rsquared  
vif_State = 1/(1-rsq_State)

# Storing vif values in a data frame
d1 = {'Variables':['RD_Spend','Administration','Marketing_Spend','State'],'VIF':[vif_RD_Spend,vif_Administration,vif_Marketing_Spend,vif_State]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As State and Administration is having higher VIF value, we are not going to include this prediction model

# final model
final_ml= smf.ols('Profit ~ RD_Spend + Marketing_Spend ',data=startup).fit()
final_ml.summary()

### Predictions 
pred = final_ml.predict(startup)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = startup.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train,startup_test  = train_test_split(startup,test_size = 0.3) # 30% test data

# preparing the model on train data 
model_train = smf.ols('Profit ~ RD_Spend  +  Marketing_Spend',data=startup_train).fit()


# train_data prediction
train_pred = model_train.predict(startup_train)

# train residual values 
train_resid  = train_pred - startup_train.Profit
train_resid 

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))  
train_rmse

# prediction on test data set 
test_pred = model_train.predict(startup_test)

# test residual values 
test_resid  = test_pred - startup_test.Profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))   
test_rmse
