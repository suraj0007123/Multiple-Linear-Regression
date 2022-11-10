# Multilinear Regression
import pandas as pd
import numpy as np
 
# loading the data
apdata = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\multiple linear regression\Datasets_MLR\Avacado_Price.csv")
apdata.columns
apdata.columns = 'AveragePrice', 'Total_Volume', 'tot_ava1', 'tot_ava2', 'tot_ava3','Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge_Bags', 'type', 'year','region' #renaming so that no sapces is there otherwise error.
 
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

import seaborn as sns
import matplotlib.pyplot as plt

byRegion=apdata.groupby('region').mean()
byRegion.sort_values(by=['AveragePrice'], ascending=False, inplace=True)
plt.figure(figsize=(17,8),dpi=100)
sns.barplot(x = byRegion.index,y=byRegion["AveragePrice"],data = byRegion,palette='rocket')
plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.title('Average Price According to Region')


#The barplot shows the average price of avocado at various regions in a ascending order. 
#Clearly Hartford Springfield, SanFrancisco, NewYork are the regions with the highest avocado prices.

# Converting Categorical Variable into numeric.
from sklearn import preprocessing 
 
label_encoder = preprocessing.LabelEncoder() 
apdata['type']= label_encoder.fit_transform(apdata['type']) 
apdata

apdata.shape

apdata.type.value_counts()

#Featuring Engineering- Handle Categorical Features Many Categories(Count/Frequency Encoding)
len(apdata["region"].unique())

# Let's obtain the counts for each one of the labels in variable "region"
# Let's capture this in a dictionary that we can use to re-map the labels

apdata.region.value_counts().to_dict()

# And now let's replace each label in "region" by its count
# first we make a dictionary that maps each label to the counts

apdata_frequency_map = apdata.region.value_counts().to_dict()

# and now we replace "region" lables in the dataset ap
apdata.region = apdata.region.map(apdata_frequency_map)

apdata.head()

apdata.describe()
apdata.columns

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import seaborn as sns

####Histogram
apdata.columns
plt.hist(apdata.AveragePrice)
plt.hist(apdata.Total_Volume)
plt.hist(apdata.tot_ava1)
plt.hist(apdata.tot_ava2)
plt.hist(apdata.tot_ava3)
plt.hist(apdata.Total_Bags)
plt.hist(apdata.Small_Bags)
plt.hist(apdata.Large_Bags)
plt.hist(apdata.XLarge_Bags)
plt.hist(apdata.type)
plt.hist(apdata.year)
plt.hist(apdata.region)


####Boxplot

sns.boxplot(apdata.AveragePrice)
sns.boxplot(apdata.Total_Volume)
sns.boxplot(apdata.tot_ava1)
sns.boxplot(apdata.tot_ava2)
sns.boxplot(apdata.tot_ava3)
sns.boxplot(apdata.Total_Bags)
sns.boxplot(apdata.Small_Bags)
sns.boxplot(apdata.Large_Bags)
sns.boxplot(apdata.XLarge_Bags)
sns.boxplot(apdata.type)
sns.boxplot(apdata.year)
sns.boxplot(apdata.region)


#barplot    
plt.bar(height = apdata["AveragePrice"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["Total_Volume"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["tot_ava1"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["tot_ava2"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["tot_ava3"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["Total_Bags"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["Small_Bags"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["Large_Bags"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["XLarge_Bags"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["type"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["year"], x = np.arange(1, 18250, 1))
plt.bar(height = apdata["region"], x = np.arange(1, 18250, 1))


# Jointplot
import seaborn as sns
sns.jointplot(x=apdata['Total_Volume'], y=apdata['AveragePrice']) #both univariate and bivariate visualization.
sns.jointplot(x=apdata['tot_ava1'], y=apdata['AveragePrice'])
sns.jointplot(x=apdata['tot_ava2'], y=apdata['AveragePrice'])
sns.jointplot(x=apdata['tot_ava3'], y=apdata['AveragePrice'])
sns.jointplot(x=apdata['Total_Bags'], y=apdata['AveragePrice'])
sns.jointplot(x=apdata['Small_Bags'], y=apdata['AveragePrice'])
sns.jointplot(x=apdata['Large_Bags'], y=apdata['AveragePrice'])
sns.jointplot(x=apdata['XLarge_Bags'], y=apdata['AveragePrice'])
sns.jointplot(x=apdata['type'], y=apdata['AveragePrice'])
sns.jointplot(x=apdata['year'], y=apdata['AveragePrice'])
sns.jointplot(x=apdata['region'], y=apdata['AveragePrice'])

# Countplot
plt.figure(1, figsize=(55, 10))
sns.countplot(apdata['AveragePrice']) 

plt.figure(1, figsize=(16, 10))
sns.countplot(apdata['type'])

plt.figure(1, figsize=(16, 10))
sns.countplot(apdata['year'])

plt.figure(1, figsize=(16, 10))
sns.countplot(apdata['region']) 
 
 

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(apdata.AveragePrice, dist = "norm", plot = pylab) # data is NOT normally distributed
plt.show()

stats.probplot(np.log(apdata['AveragePrice']),dist="norm",plot=pylab) #best transformation, Now data is normally distributed.

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(apdata.iloc[0:1000,:])
                             
# Correlation matrix 
a = apdata.corr() 
# we see there exists High collinearity between input variables 
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
#here ignoring the collinearity problem    
# We try to eliminate reasons of those varibales being insignificant.try to look into various scenario:
#1st scenario is, Is this because of the relation between y and x, we apply simple linear regression between, y and x1, y and x2.....so on.
#If it showing that there is no problem, we proceed further for influential observation.
ml1 = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + XLarge_Bags + type + year + region', data = apdata).fit() # regression model

# Summary
ml1.summary()
# P-values of all variables are more than 0.05 except type
# R-squared:                       0.408
# Adj. R-squared:                  0.408
# Here we can clearly see that r-squared is also very low
# So this model is not accppetable

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1) ##System not supporting this much calculations, getting crashed in between

apdata_new = apdata.drop(apdata.index[[14125, 15560, 17468]])


#Applying transformations => CubeRoot(x)

model2=smf.ols('AveragePrice~np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)', data=apdata_new).fit()

model2.summary()

#now we can see that
# R-squared:                       0.502
# Adj. R-squared:                  0.502
# R-squared has improved
# p-value of all the variables are now below than 0.05 

# So checking for collinearity to remove variable using VIF

rsq_region = smf.ols('np.cbrt(region) ~ np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)', data = apdata).fit().rsquared  
vif_region = 1/(1 - rsq_region)  
vif_region #  1.0220383403449556

rsq_year = smf.ols('np.cbrt(year)~np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)', data = apdata).fit().rsquared  
vif_year = 1/(1 - rsq_year)  
vif_year # 1.3220141296382328

rsq_type = smf.ols('np.cbrt(type)~np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)', data = apdata).fit().rsquared  
vif_type = 1/(1 - rsq_type)  
vif_type # 1.806011362190048

rsq_XLarge_Bags = smf.ols('np.cbrt(XLarge_Bags)~np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)', data = apdata).fit().rsquared  
vif_XLarge_Bags = 1/(1 - rsq_XLarge_Bags)  
vif_XLarge_Bags # 2.674814560074929

rsq_Large_Bags = smf.ols('np.cbrt(Large_Bags)~np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)', data = apdata).fit().rsquared  
vif_Large_Bags = 1/(1 - rsq_Large_Bags)  
vif_Large_Bags # 12.815223224015904

rsq_Small_Bags = smf.ols('np.cbrt(Small_Bags)~np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)', data = apdata).fit().rsquared  
vif_Small_Bags = 1/(1 - rsq_Small_Bags)  
vif_Small_Bags # 97.05464049893034

rsq_Total_Bags = smf.ols('np.cbrt(Total_Bags)~np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)', data = apdata).fit().rsquared  
vif_Total_Bags = 1/(1 - rsq_Total_Bags)  
vif_Total_Bags # 183.66105535482146

rsq_tot_ava3 = smf.ols('np.cbrt(tot_ava3)~np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)', data = apdata).fit().rsquared  
vif_tot_ava3 = 1/(1 - rsq_tot_ava3)  
vif_tot_ava3 # 4.6103041635745425

rsq_tot_ava2 = smf.ols('np.cbrt(tot_ava2)~np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)+np.cbrt(tot_ava1)', data = apdata).fit().rsquared  
vif_tot_ava2 = 1/(1 - rsq_tot_ava2)  
vif_tot_ava2 # 58.198269554686426

rsq_tot_ava1= smf.ols('np.cbrt(tot_ava1)~np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)+np.cbrt(Total_Volume)', data = apdata).fit().rsquared  
vif_tot_ava1 = 1/(1 - rsq_tot_ava1)  
vif_tot_ava1 # 35.15902390407044

rsq_Total_Volume = smf.ols('np.cbrt(Total_Volume)~np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)', data = apdata).fit().rsquared  
vif_Total_Volume = 1/(1 - rsq_Total_Volume)  
vif_Total_Volume # 306.81074681924065

# Here we can clearly see that VIF of Total_Volume is very high
# So we will remove this variable from our calculations

apdata.columns
# Storing vif values in a data frame
d1 = {'Variables':['Total_Volume','tot_ava1','tot_ava2','tot_ava3','Total_Bags','Small_Bags','Large_Bags',' XLarge_Bags',' type','year','region'],
      'VIF':[vif_Total_Volume, vif_tot_ava1, vif_tot_ava2, vif_tot_ava3, vif_Total_Bags, vif_Small_Bags, vif_Large_Bags, vif_XLarge_Bags, vif_type,vif_year,vif_region]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# So following variables have very high vif Total_Volume, Total_Bags
# Removing these variables from our calculation

apdata1=apdata.drop(['Total_Volume','Total_Bags'], axis=1)
apdata1.columns

#creating the model again
model3=smf.ols('AveragePrice~np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)', data=apdata1).fit()
model3.summary()

# coefficients are statistically significant,but R^2 And Adjusted r^2 is reduced so, model2 is the good model.

# with Total_Volume, Total_Bags: R-squared:0.502, Without Total_Volume, Total_Bags:R-squared:0.458
#with Total_Volume, Total_Bags: Adj. R-squared:0.502, Without Total_Volume, Total_Bags:Adj. R-squared:0.458
#R-squared and Adj. R-squared: Decreased

"model2 =  Final Model"

# Prediction - ml_new
pred = model2.predict(apdata)

# Q-Q plot
res = model2.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = apdata.AveragePrice, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(model2) #System not supporting this much calculations, getting crashed in between

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
apdata_train, apdata_test = train_test_split(apdata, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("AveragePrice~np.cbrt(Total_Volume)+np.cbrt(tot_ava1)+np.cbrt(tot_ava2)+np.cbrt(tot_ava3)+np.cbrt(Total_Bags)+np.cbrt(Small_Bags)+np.cbrt(Large_Bags)+np.cbrt(XLarge_Bags)+np.cbrt(type)+np.cbrt(year)+np.cbrt(region)", data = apdata_train).fit()

# prediction on test data set 
test_pred = model_train.predict(apdata_test)

test_pred
# test residual values 
test_resid = test_pred - apdata_test.AveragePrice
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse # 0.2821091442088658


# train_data prediction
train_pred = model_train.predict(apdata_train)
train_pred

# train residual values 
train_resid  = train_pred - apdata_train.AveragePrice
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse #0.2846329705373221

# Training Error and Test Error is approximately equal then we can say it is right fit.
#So this model can be accepted