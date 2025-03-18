#Lead_Scoring_Case_Study

#Import libraries  for data load and  reading
import numpy as np
import pandas as pd 

#import libraries for  visualization
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

##############    STEP 1 : Read  Dataset   ################
print('############# STEP 1 : Read  Dataset    ')

df = pd.read_csv('Leads.csv')

# Details about the dataset
print("Shape of the dataset ",df.shape)  #  (9240, 37)
print("Check head the dataset ")
print(df.head(2))
print(df.info())

print(df.isnull().sum())

##############  STEP 2 : Data Cleaning   ####################

print('############# STEP 2 : Data Cleaning   ########  ')

# check the precentage of the null values in columns 
#print(len(df.index))
#print(df.shape[0])

print("Percentage of null values ")
print(round((df.isnull().sum()*100)/len(df.index),2))

# Drop all columns with more than 45 percent of null values

cols  = []
cols  =  df.columns[(round((df.isnull().sum()*100)/len(df.index),2))>45]
print("Columns to drop ", cols)

df  =df.drop(cols  , axis=1)

#check the percentage of nulls in remaining columns
print(round((df.isnull().sum()*100)/len(df.index),2))

print("New Shape of the dataset ", df.shape)  # 7 columns were dropped ,  (9240, 32)

# Still we have  multiple columns with high  null values  

#  Handel city column

print(df.City.value_counts(dropna=False))  
#Mumbai has highest leads  and 2249 didn't selected the city  , changing the unselected city toi mumbai

df['City'] = df['City'].replace('Select', 'Mumbai')
df['City'] = df['City'].replace(np.nan,'Mumbai')

# Handel country column
print(df.Country.value_counts(dropna=False))
#India has very high value as compare to other countries hence country will not have major impact on values (showing huge data imbalance )we can drop the  country 
df = df.drop('Country' , axis=1)

# Handel Specialization column
print(df.Specialization.value_counts(dropna=False))  # Values are distributed , Fill NaN with 'Not Provided'

df['Specialization'] = df['Specialization'].replace(np.nan,'Not Provided')


#Check values for  "How did you hear about X Education"
print(df['How did you hear about X Education'].value_counts(dropna=False))  
# Huge value  belongs to Select or NaN , hence  we can drop this column
df = df.drop('How did you hear about X Education', axis=1)

# check values for  "What matters most to you in choosing a course"
print(df['What matters most to you in choosing a course'].value_counts(dropna=False)) 
 # Huge value  belongs to 'Better Career Prospects' or NaN , hence  we can drop this column
df = df.drop('What matters most to you in choosing a course', axis=1)

# check values for "What is your current occupation"
print(df['What is your current occupation'].value_counts(dropna=False)) 
# Huge value  belongs to 'Unemployed' or NaN , Replacing NaN with 'Not provided'
df['What is your current occupation'] = df['What is your current occupation'].replace(np.nan,'Not Provided')


#check values for 'Lead Profile' column
print(df['Lead Profile'].value_counts(dropna=False))
# Huge value  belongs to 'Unemployed' or NaN , Replacing NaN with 'Not provided'
df['Lead Profile'] = df['Lead Profile'].replace(np.nan,'Not Provided')

#check values for  Tags Columns
print(df['Tags'].value_counts(dropna=False))
# Huge value  belongs to  NaN , Replacing NaN with 'Not provided'
df['Tags'] = df['Tags'].replace(np.nan,'Not Provided')


# We can assign maximum frequency values for  the rows which has less number of missing values

cols = df.columns[round((df.isnull().sum()*100)/len(df.index),2)>0]
print(cols)

#print( df['Lead Source'].value_counts(dropna=False))
df['Lead Source'] = df['Lead Source'].replace(np.nan,'Google')
#print( df['TotalVisits'].value_counts(dropna=False))
df['TotalVisits'] = df['TotalVisits'].replace(np.nan,'0.0')
#print( df['Page Views Per Visit'].value_counts(dropna=False))
df['Page Views Per Visit'] = df['Page Views Per Visit'].replace(np.nan,'0.00')

print( df['Last Activity'].value_counts(dropna=False))
df['Last Activity'] = df['Last Activity'].replace(np.nan,'Not Provided')


# check again the null values 
print(round((df.isnull().sum()*100)/len(df.index),2))
print(df.shape)   # (9240, 29)

# Copy  DataFrame to Leads dataframe  

Leads = df 
print(Leads.shape) # (9240, 29)

#############  STEP 3 : Exploratory Data Analysis      #######################

print('############# STEP 3 : Exploratory Data Analysis     ########  ')

print(Leads.info())
# 'Prospect ID' and 'Lead Number' are just numerical identity variables , hence can be dropped
Leads = Leads.drop('Prospect ID' , axis=1)
Leads = Leads.drop('Lead Number' , axis=1)
print(Leads.shape)  #  (9240, 27)

###############  STEP 3.1 : Univariate Analysis     #######################

# Categorical Variables
plt.figure(figsize = (20,40))

plt.subplot(4,2,1)
sns.countplot(Leads['Lead Origin'])
plt.title('Lead Origin')

plt.subplot(4,2,2)
sns.countplot(Leads['Lead Source']).tick_params(axis='x', rotation =90 )
plt.title("Lead Source Visualization")

plt.subplot(4,2,3)
sns.countplot(Leads['Newspaper Article'])
plt.title('Newspaper Article')

plt.subplot(4,2,4)
sns.countplot(Leads['X Education Forums'])
plt.title('X Education Forums')

plt.subplot(4,2,5)
sns.countplot(Leads['Newspaper'])
plt.title('Newspaper')

plt.subplot(4,2,5)
sns.countplot(Leads['Digital Advertisement'])
plt.title('Digital Advertisement')

plt.subplot(4,2,6)
sns.countplot(Leads['Through Recommendations'])
plt.title('Through Recommendations')

plt.subplot(4,2,7)
sns.countplot(Leads['A free copy of Mastering The Interview'])
plt.title('A free copy of Mastering The Interview')

plt.subplot(4,2,8)
sns.countplot(Leads['Last Notable Activity']).tick_params(axis='x', rotation = 90)
plt.title('Last Notable Activity')

plt.show()

# categorical variables and Converted graph
plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Lead Origin', hue='Converted', data= Leads).tick_params(axis='x', rotation = 90)
plt.title('Lead Origin')

plt.subplot(1,2,2)
sns.countplot(x='Lead Source', hue='Converted', data= Leads).tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()  

# Specialization and current occupation with converted
plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Specialization', hue='Converted', data= Leads).tick_params(axis='x', rotation = 90)
plt.title('Specialization')

plt.subplot(1,2,2)
sns.countplot(x='What is your current occupation', hue='Converted', data= Leads).tick_params(axis='x', rotation = 90)
plt.title('What is your current occupation')
plt.show()

#Last activity and Converted
sns.countplot(x='Last Notable Activity', hue='Converted', data= Leads).tick_params(axis='x', rotation = 90)
plt.title('Last Notable Activity')
plt.show()

## Numerical variables , First convert them to numeric if needed

Leads['TotalVisits'] = pd.to_numeric(Leads['TotalVisits'], errors='coerce')
Leads['Page Views Per Visit'] = pd.to_numeric(Leads['Page Views Per Visit'], errors='coerce')

plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
plt.hist(Leads['TotalVisits'], bins = 200)
plt.title('Total Visits')
plt.xlim(0,25)

plt.subplot(2,2,2)
plt.hist(Leads['Total Time Spent on Website'], bins = 10)
plt.title('Total Time Spent on Website')

plt.subplot(2,2,3)
plt.hist(Leads['Page Views Per Visit'], bins = 20)
plt.title('Page Views Per Visit')
plt.xlim(0,20)
plt.show()


################ STEP 4 : Dummy Variables ##################

print('############# STEP 4 : Dummy Variables    ########  ')

# Get all categorical variables 
cat_cols  = Leads.select_dtypes(include='object').columns
print(cat_cols)

# Create dummy variables  from get_dummies() method

dummy  = pd.get_dummies(Leads[cat_cols], drop_first=True)
dummy = dummy.astype(int)
Leads_Final  = pd.concat([Leads, dummy], axis=1)
#print(Leads_Final.head(2)) 

Leads_Final = Leads_Final.drop(cat_cols ,axis=1)
print(Leads_Final.shape)  # (9240, 131)

######################  STEP 5 : TRAIN - TEST SPLIT ######################

print('#############  STEP 5 : TRAIN - TEST SPLIT    ########  ')

# import test train split library from scikit lean package 
from sklearn.model_selection import train_test_split


Leads_Final = Leads_Final.apply(pd.to_numeric, errors='coerce')

# Optionally, fill NaN values with a specific value (e.g., 0) or drop them
Leads_Final = Leads_Final.fillna(0)  # or Leads.dropna()

X = Leads_Final.drop('Converted' , axis=1)
y = Leads_Final['Converted']
print(X.head(2))
print(y.head())

# split the data into training and test data by 70 and 30 percent respectively 
X_train , X_test , y_train , y_test = train_test_split(X,y, train_size=0.7 , random_state=42)
print('Training data set shape = ', X_train.shape)
print('Test dataset  shape =  ',X_test.shape)

# Rescaling numeric variables using MinMaxScaler

#Import MinMaxSclaer 
from sklearn.preprocessing import MinMaxScaler

#scale the numeric features
Scaler  = MinMaxScaler()
X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = Scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] )
print(X_train.head())

# To check the correlation among varibles in training dataset
plt.figure(figsize=(20,30))
sns.heatmap(X_train.corr())
plt.show()

# Since  features are high in numbers hence we will analyze and  drop irrelevant features later

####################  STEP 6  : MODEL BUILDING - LOGISTIC REGRESSION #################

print('#############   STEP 6  : MODEL BUILDING - LOGISTIC REGRESSION    ########  ')

#Import Logistic Regression
from sklearn.linear_model import  LogisticRegression
log_reg  = LogisticRegression()

# Feature selection through RFE 
from sklearn.feature_selection import  RFE

print(X_train.info())

# Running feature sleection for 15 top relevant  features 
rfe  = RFE(estimator=log_reg , n_features_to_select=15)
rfe = rfe.fit(X_train,y_train)

# get features column slected  by RFE
rfe_cols  = X_train.columns[rfe.support_]
print(rfe_cols)

# Progressing  with columns selected  by RFE 
#X_train = X_train[rfe_cols]

# Import statsmodel
import statsmodels.api as sm

# Model -1 
X_train_sm = sm.add_constant( X_train[rfe_cols])
logm1 = sm.GLM(y_train, X_train_sm,family=sm.families.Binomial())
res1  = logm1.fit()
print(res1.summary())

# Evaluate P-Value and  VIFs for feature varaibles

#P-Value of "Lead Source_Welingak Website" feature variable is too high hence dropping it  
#dropping column with high p-value

rfe_cols = rfe_cols.drop('Lead Source_Welingak Website')

# Model -2 
X_train_sm2 = sm.add_constant( X_train[rfe_cols])
logm2 = sm.GLM(y_train, X_train_sm2,family=sm.families.Binomial())
res2  = logm2.fit()
print(res2.summary())

# Feature Variable  "Tags_wrong number given " has very high p-value hence dropping it 

rfe_cols = rfe_cols.drop('Tags_wrong number given')

# Model -3 
X_train_sm3 = sm.add_constant( X_train[rfe_cols])
logm3 = sm.GLM(y_train, X_train_sm3,family=sm.families.Binomial())
res3  = logm3.fit()
print(res3.summary())

# Feature  variable "Tags_invalid number" has high p-value hence dropping it 

rfe_cols = rfe_cols.drop('Tags_invalid number')

# Model -4 
X_train_sm4 = sm.add_constant( X_train[rfe_cols])
logm4 = sm.GLM(y_train, X_train_sm4,family=sm.families.Binomial())
res4  = logm4.fit()
print(res4.summary())

# Importing 'variance_inflation_factor'
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train_sm4.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm4.values, i) for i in range(X_train_sm4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)

# feature variable  'Lead Profile_Student of SomeSchool' has low VIF but high P-value hence dropping it now 

rfe_cols = rfe_cols.drop('Lead Profile_Student of SomeSchool')

# Model -5 
X_train_sm5 = sm.add_constant( X_train[rfe_cols])
logm5 = sm.GLM(y_train, X_train_sm5,family=sm.families.Binomial())
res5  = logm5.fit()
print(res5.summary())
# Make a VIF dataframe for all the variables now
vif = pd.DataFrame()
vif['Features'] = X_train_sm5.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm5.values, i) for i in range(X_train_sm5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)


# feature variable  'Tags_switched off' has low VIF but high P-value hence dropping it now 

rfe_cols = rfe_cols.drop('Tags_switched off')

# Model -6 
X_train_sm6 = sm.add_constant( X_train[rfe_cols])
logm6 = sm.GLM(y_train, X_train_sm6,family=sm.families.Binomial())
res6  = logm6.fit()
print(res6.summary())
# Make a VIF dataframe for all the variables now
vif = pd.DataFrame()
vif['Features'] = X_train_sm6.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm6.values, i) for i in range(X_train_sm6.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)

# Now P-value and VIF both are as per expectation, hence we can fix this model

########### STEP 7 : CREATING PREDICTION ###############


print('#############   STEP 7 : CREATING PREDICTION    ########  ')

#predicting the probablities on the training dataset
y_train_pred = res6.predict(X_train_sm6)
print(y_train_pred.head())

# Reshaping to an Array
y_train_pred = y_train_pred.values.reshape(-1)
#print(y_train_pred.head())

# creating dataframe with given conversion and  probablities of conversion
y_train_pred_final = pd.DataFrame({'Converted' :y_train.values,'Conversion_Prob':y_train_pred})
#print(y_train_pred_final.head())

# Create a predicted column using Conversion_Prob column using a cut off of 0.5 
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if  x> 0.5 else 0)
print("Final Prediction table  ")
print(y_train_pred_final.head())

################## STEP 8  : MODEL EVALUATION ####################

print('#############   STEP 8  : MODEL EVALUATION   ########  ')

# import metrics libraries 
from sklearn import metrics

# Confusion Matrix
confusion  = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
print(confusion)

# Accuracy score
print('Accuracy Score')
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))

# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]

# Calculating the sensitivity
print('Sensitivity')
print(TP/(TP+FN))

# Calculating the specificity
print('Specificity')
print(TN/(TN+FP))

######################  STEP 9 : ROC Curve ############

print('#############   STEP 9  : ROC CURVE   ########  ')

# Calculate FPT , TPR and Threshold
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )

print('fpr- ',fpr  ,'tpr  - ',tpr , 'thresholds - ',thresholds)

# ROC Function 
def draw_roc( actual, probs ):
    fpr , tpr,  thresholds  = metrics.roc_curve(actual , probs , drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual,probs)
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

# Call the ROC function
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)

#### Finding Optimal Cutoff Point for balanced  sensitivity and specificity

# Creating columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)

print(y_train_pred_final.head())


# Now calculate  accuracy , sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

# Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity
from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]


print("Cut off  data frame is as below ")

print( cutoff_df)


#Plot accuracy, sensitivity and specificity for various probabilities.
plt.figure(figsize=(10,6))
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()

#
#From the curve above, 0.35 is the optimum point to take it as a cutoff probability for accuracy, sensitivity and specificity 

# Recreate  predicted column with 0.3 as cutoff

y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if  x> 0.35 else 0)
print("Final Prediction table after putting cut off as 0.35 ")
print(y_train_pred_final.head())

# Check the overall accuracy
print('New accuracy score with 0.35 as cut off  is  = ',metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))

# Creating confusion matrix 
confusion_2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion_2)


TP = confusion_2[1,1] # true positive # Which are actually converted and we have predicted them as converted
TN = confusion_2[0,0] # true negatives # Which are actually not converted and we have predicted them as not converted
FP = confusion_2[0,1] # false positives # Which are actually not converted but we have predicted them as converted
FN = confusion_2[1,0] # false negatives # Which are actually converted but we have predicted them as not converted

accurcay2 = (TP + TN) / (TP + TN + FP + FN)
sensitivity2 = TP / (TP + FN)
specificity2 = TN / (TN + FP)
fpr = FP / (TN + FP)
tpr = TP / (TP + FN)
ppv = TP / (TP + FP)
npv = TN / (TN + FN)

print('Accuracy - ',accurcay2)
print('Sensitivity - ',sensitivity2)
print('Specificity - ',specificity2)
print('False Positive Rate - ',fpr)
print('True Positive Rate - ',tpr)
print('Positive Predictive Value - ',ppv)
print('Negative Predictive Value - ',npv)


## With the current cut off as 0.35 we have accuracy, sensitivity and specificity of around 80%.

##################  STEP 10 : Prediction on Test Dataset ###############

print('#############   STEP 10  :  Prediction on Test Dataset   ########  ')

# Scaling numeric values
X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = Scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

# Substituting all the columns in the final train model
col = X_train.columns

# Select the columns in X_train for X_test as well
X_test = X_test[rfe_cols]
# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res6.predict(X_test_sm)

# Coverting y_test_pred to dataframe
y_pred_df = pd.DataFrame(y_test_pred)

# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)

# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)

# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
print(y_pred_final.head())

# Prediction with cut off 0.35 as calculated from training dataset
y_pred_final['Predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.35 else 0)
print(' y_pred_final head')
print(y_pred_final.head())

# Check the overall accuracy for test data
print('Accuracy for test data  ' ,metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.Predicted))

#  confusion matrix  for test data
confusion_test = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.Predicted )
print(confusion_test)

# true positive
TP = confusion_test[1,1]
#  true negatives
TN = confusion_test[0,0]
#  false positives
FP = confusion_test[0,1] 
#  false negatives
FN = confusion_test[1,0]

accurcay_test = (TP + TN) / (TP + TN + FP + FN)
sensitivity_test = TP / (TP + FN)
specificity_test = TN / (TN + FP)
fpr = FP / (TN + FP)
tpr = TP / (TP + FN)

print(' Test Accuracy - ',accurcay_test)
print('Test Sensitivity - ',sensitivity_test)
print('Test Specificity - ',specificity_test)
print('Test data False Positive Rate - ',fpr)
print('Test data True Positive Rate - ',tpr)

############## STEP 11  : Precision Recall  ################


print('#############   STEP 11  :  Precision Recall    ########  ')

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print('precision_score = ' ,precision_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)  )
print('recall_score = ',recall_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)  )

# plot a precision-recall curve

from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()

y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.4 else 0)
print(y_train_pred_final.head())

# Accuracy
print('New accuracy score = ',metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))

# Creating confusion matrix again
confusion_new = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion_new)

# ] true positive
TP = confusion_new[1,1]
#  true negatives
TN = confusion_new[0,0]
#  false positives
FP = confusion_new[0,1] 
# false negatives
FN = confusion_new[1,0]

print(' Precision = ', TP /( TP + FP))
print('Recall = ',TP / (TP + FN))

#With the current cut off as 0.4 we have Precision around 74% and Recall around 76%

#################      STEP 10  : Prediction on Test Data set ###########

# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res6.predict(X_test_sm)

# Coverting it to dataframe
y_pred_df = pd.DataFrame(y_test_pred)

# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)

# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)

# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
print(y_pred_final.head())

# Making prediction using cut off 0.4
y_pred_final['Predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.4 else 0)
print(y_pred_final)

# Check the overall accuracy
print('Overall Accuracy  score = ',metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.Predicted))

# Creating confusion matrix 
confusion_final = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.Predicted )
print(confusion_final)

#  true positive
TP = confusion_final[1,1]
#  true negatives
TN = confusion_final[0,0]
#  false positives
FP = confusion_final[0,1] 
#  false negatives
FN = confusion_final[1,0]

print('Overall Precision = ', TP / (TP + FP))
print('Ovaerall Recall = ', TP / (TP + FN))

#With the current cut off as 0.4 we have Accuracy around 93% , Precision  is 90%  and Recall around 84%

