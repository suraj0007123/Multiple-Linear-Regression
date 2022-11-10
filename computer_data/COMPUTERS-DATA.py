import pandas as pd
import numpy as np
import seaborn as sns

# loading the data
LAP = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\multiple linear regression\Datasets_MLR\Computer_Data.csv")

LAP.columns

LAP.rename(columns={"Unnamed: 0":"X"},inplace=True)

LAP.columns

LAP = LAP.iloc[:, 1:12]

LAP.drop(['cd','multi','ads','trend'],axis =1,inplace =True)

from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
lb = LabelEncoder()

LAP.columns

LAP['premium']= lb.fit_transform(LAP['premium'])

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

LAP.describe()

LAP.info()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# speed 
plt.bar(height = LAP.speed, x = np.arange(1, 6260, 1))
plt.hist(LAP.speed) #histogram
sns.boxplot(LAP.speed) #boxplot

# hd
plt.bar(height = LAP.hd, x = np.arange(1, 6260, 1))
plt.hist(LAP.hd) #histogram
sns.boxplot(LAP.hd) #boxplot

# ram
plt.bar(height = LAP.ram, x = np.arange(1, 6260, 1))
plt.hist(LAP.ram) #histogram
sns.boxplot(LAP.ram) #boxplot

# screen
plt.bar(height = LAP.screen, x = np.arange(1, 6260, 1))
plt.hist(LAP.screen) #histogram
sns.boxplot(LAP.screen) #boxplot

# premium
plt.bar(height = LAP.premium, x = np.arange(1, 6260, 1))
plt.hist(LAP.premium) #histogram
sns.boxplot(LAP.premium) #boxplot

# Jointplot for RAM and HD which seem to have outliers and might effect the models ahead
import seaborn as sns
sns.jointplot(x = LAP['speed'], y = LAP['ram'])


sns.jointplot(x = LAP['hd'], y = LAP['ram'])

# Countplot
plt.figure(1, figsize=(10, 5))
sns.countplot(LAP['speed'])

plt.figure(1, figsize=(30, 5))
sns.countplot(LAP['hd'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(LAP.price, dist = "norm", plot = pylab)
plt.show()

stats.probplot(LAP.ram, dist = "norm", plot = pylab)
plt.show()

stats.probplot(LAP.hd, dist = "norm", plot = pylab)
plt.show()


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(LAP.iloc[:, :])
                             
# Correlation matrix 
LAP.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price ~ speed + hd + ram + screen + premium', data = LAP).fit() # regression model

# Summary
ml1.summary()
# the R square value is high denoting the existence of multicollinearity

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 4477, 3783, 900 is showing high influence so we can exclude that entire row

LAP_new = LAP.drop(LAP.index[[900, 3783, 4477]])

# Preparing model                  
ml_new = smf.ols('price ~ speed + hd + ram + screen + premium', data = LAP_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_sp = smf.ols('speed ~ hd + ram + screen + premium', data = LAP).fit().rsquared  
vif_sp = 1/(1 - rsq_sp) 

rsq_hd = smf.ols('hd ~ speed + ram + screen + premium', data = LAP).fit().rsquared  
vif_hd = 1/(1 - rsq_hd)

rsq_ram = smf.ols('ram ~ speed + hd + screen + premium', data = LAP).fit().rsquared  
vif_ram = 1/(1 - rsq_ram) 

rsq_sc = smf.ols('screen ~ speed + ram + hd + premium', data = LAP).fit().rsquared  
vif_sc = 1/(1 - rsq_sc) 

rsq_pr = smf.ols('premium ~ speed + ram + hd + screen', data = LAP).fit().rsquared  
vif_pr = 1/(1 - rsq_pr) 


# Storing vif values in a data frame
d1 = {'Variables':['speed', 'hd', 'ram', 'screen', 'premium'], 'VIF':[vif_sp, vif_hd, vif_ram, vif_sc, vif_pr]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As hd is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('price ~ speed + premium + screen', data = LAP).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(LAP)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = LAP.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
LAP_train, LAP_test = train_test_split(LAP, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("price ~ speed + premium + screen", data = LAP_train).fit()

model_train.summary()

# prediction on test data set 
test_pred = model_train.predict(LAP_test)

# test residual values 
test_resid = test_pred - LAP_test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(LAP_train)

# train residual values 
train_resid  = train_pred - LAP_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

# As we saw earlier the columns 'hd' and 'ram' were effecting the data and the rmse values just shot up
# now with the latest model having other features the Train RMSE = 532.388 and Test RMSE = 526.8222
