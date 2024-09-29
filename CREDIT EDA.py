#Importing libraries
import warnings

import inline as inline
import matplotlib
import numpy as np

warnings.filterwarnings("ignore")
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns",None)

##READING CSV
app_data_train = pd.read_csv("application_train.csv")
print(app_data_train.head(1))

###INSPECTION OF TRAINING DATASET
print(app_data_train.info())
##QUALITY CHECK
##CHECKING FOR NULL VALUES
pd.set_option("display.max_rows",200)
print(app_data_train.isnull().mean()*100)
###CONCLUSION COLUMNS WITH NULL VALUES:- COLUMNS WITH MORE THAN 47% OF DATA BEING NULL VALUES HAS TO BE DROPPED
##DROPPING THOSE COLUMNS
percentage = 47
threshold = int(((100-percentage)/100)*app_data_train.shape[0] +1)
app_dt = app_data_train.dropna(axis=1,thresh=threshold)
app_dt.head(1)
app_dt.isnull().mean()*100

### CHECKING FOR MISSING VALUES
print(app_dt.info())
###OCCUPATION_TYPE COLUMN HAS 31% MISSING VALUES.
###WE WILL IMPUTE MISSING VALUES WITH UNKNOWNS
app_dt.OCCUPATION_TYPE.isnull().mean()*100
app_dt.OCCUPATION_TYPE.value_counts(normalize=True)*100
app_dt.OCCUPATION_TYPE.fillna('Others',inplace=True)
##EXT_SOURCE 3 HAS 19% MISSING VALUES
app_dt.EXT_SOURCE_3.isnull().mean()*100
app_dt.EXT_SOURCE_3.value_counts(normalize=True)*100

sns.boxplot(app_dt.EXT_SOURCE_3)
app_dt.EXT_SOURCE_3.fillna(app_dt.EXT_SOURCE_3.median(),inplace=True) ##FILLING UP THE MISSING VALUES
app_dt.EXT_SOURCE_3.isnull().mean()*100
app_dt.EXT_SOURCE_3.value_counts(normalize=True)*100
null_cols =list(app_dt.isna().any())
##HANDLING MISSING VALUES IN COLUMNS WITH 13% MISSING VALUES
app_dt.AMT_REQ_CREDIT_BUREAU_HOUR.value_counts(normalize=True)*100
app_dt.AMT_REQ_CREDIT_BUREAU_DAY.value_counts(normalize=True)*100

##CONCLUSION:-90% OF VALUES IN AMT_REQ_CREDIT_BUREAU_DAY, _WEEK, _HOUR, _MON, _QRT ARE 0.0
##HENCE IMPUTING THESE COLUMNS WITH MODE
COLUMNS = ['AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']
for col in COLUMNS:
    app_dt[col].fillna(app_dt[col].mode()[0], inplace=True)
print(app_dt.isnull().mean()*100)

##HANDLING MORE MISSING VALUES FOR LESS THAN 1% COLUMNS
NULL_cols = list(app_dt.columns[app_dt.isna().any()])
#(app_dt.NAME_TYPE_SUITE.value_counts(normalize=True)*100)
print(app_dt.EXT_SOURCE_2.value_counts(normalize=True)*100)
##IMPUTING NUMERICAL COLUMNS
app_dt.NAME_TYPE_SUITE.fillna(app_dt.NAME_TYPE_SUITE.mode()[0],inplace=True)
app_dt.CNT_FAM_MEMBERS.fillna(app_dt.CNT_FAM_MEMBERS.mode()[0],inplace=True)
app_dt.EXT_SOURCE_2.fillna(app_dt.EXT_SOURCE_2.median(),inplace=True)
app_dt.AMT_ANNUITY.fillna(app_dt.AMT_ANNUITY.median(),inplace=True)
app_dt.AMT_GOODS_PRICE.fillna(app_dt.AMT_GOODS_PRICE.median(),inplace=True)
app_dt.DEF_60_CNT_SOCIAL_CIRCLE.fillna(app_dt.DEF_60_CNT_SOCIAL_CIRCLE.median(),inplace=True)
app_dt.DEF_30_CNT_SOCIAL_CIRCLE.fillna(app_dt.DEF_30_CNT_SOCIAL_CIRCLE.median(),inplace=True)
app_dt.OBS_60_CNT_SOCIAL_CIRCLE.fillna(app_dt.OBS_60_CNT_SOCIAL_CIRCLE.median(),inplace=True)
app_dt.OBS_30_CNT_SOCIAL_CIRCLE.fillna(app_dt.OBS_30_CNT_SOCIAL_CIRCLE.median(),inplace=True)
app_dt.DAYS_LAST_PHONE_CHANGE.fillna(app_dt.DAYS_LAST_PHONE_CHANGE.median(),inplace=True)
##NOW NULL_COLS ARE 0
####CONVERSION OF  NEGATIVE VALUES in days variables
app_dt.DAYS_BIRTH = app_dt.DAYS_BIRTH.apply(lambda x: abs(x))
app_dt.DAYS_EMPLOYED = app_dt.DAYS_EMPLOYED.apply(lambda x: abs(x))
app_dt.DAYS_REGISTRATION = app_dt.DAYS_REGISTRATION.apply(lambda x: abs(x))
app_dt.DAYS_LAST_PHONE_CHANGE = app_dt.DAYS_LAST_PHONE_CHANGE.apply(lambda x: abs(x))
app_dt.DAYS_ID_PUBLISH = app_dt.DAYS_ID_PUBLISH.apply(lambda x: abs(x))

#####BINNING OF CONTINUOUS VALUES
##STANDARDIZING DAYS COLUMNS IN YEARS FOR EASY BINNING
app_dt['YEARS_BIRTH'] = app_dt.DAYS_BIRTH.apply(lambda x: int(x//365))
app_dt['YEARS_EMPLOYED'] = app_dt.DAYS_EMPLOYED.apply(lambda x: int(x//365))
app_dt['YEARS_REGISTRATION'] = app_dt.DAYS_REGISTRATION.apply(lambda x: int(x//365))
app_dt['YEARS_ID_PUBLISH'] = app_dt.DAYS_ID_PUBLISH.apply(lambda x: int(x//365))
app_dt['YEARS_LAST_PHONE_CHANGE'] = app_dt.DAYS_LAST_PHONE_CHANGE.apply(lambda x: int(x//365))

###BINNING AMT CREDIT
app_dt.AMT_CREDIT.value_counts(normalize=True)*100
app_dt.AMT_CREDIT.describe()

app_dt['AMT_CREDIT_Category'] = pd.cut(app_dt.AMT_CREDIT,[0,200000,400000,600000,800000,1000000],
                                       labels= ['Very low Credit','Low Credit','Medium Credit','High Credit','Very High Credit'])
app_dt.AMT_CREDIT_Category.value_counts(normalize=True)*100
app_dt['AMT_CREDIT_Category'].value_counts(normalize=True).plot.bar()
##CONCLUSION:-CREDIT OF THE LOAN AMOUNT LOW(2L-4L) OR VERY HIGH(8L)


####BINNING YEARS BIRTH COLUMNS
app_dt['AGE_Category'] = pd.cut(app_dt.YEARS_BIRTH, [0,25,45,65,85],
                                labels=['Below 25','25-45','45-65','65-85'])
app_dt.AGE_Category.value_counts(normalize=True)*100
app_dt['AGE_Category'].value_counts(normalize=True).plot.pie(autopct='%1.2f%%')
plt.show()
##CONCLUSION:-MAJORITY OF APPLICANTS AGE IS INBETWEEN 25 TO 45

####DATA IMBALANCE CHECK
print(app_dt.head())

###DIVIDING APPLICATION DATASET WITH TARGET VARIABLE AS 1 AND 0
tar_0 = app_dt[app_dt.TARGET == 0]
tar_1 = app_dt[app_dt.TARGET == 1]

print(app_dt.TARGET.value_counts(normalize=True)*100)

##CONCLUSION:- 1 out 9/10  applicants are defaulters


######UNIVARIATE ANALYSIS
## NOW WE WILL SEPARATE OUT WITH CATEGORICAL AND NUMERICAL COLUMNS
cat_cols = list(app_dt.columns[app_dt.dtypes == np.object_])
num_cols = list(app_dt.columns[app_dt.dtypes == np.int64]) + list(app_dt.columns[app_dt.dtypes == np.float64])
##NOW PLOTTING ALL CAT COLS
for col in cat_cols:
    print(app_dt[col].value_counts(normalize=True))
    plt.figure(figsize=[5,5])
    app_dt[col].value_counts(normalize=True).plot.pie(labeldistance=None,autopct='%1.2f%%')
    plt.legend()
    plt.show()


##PLOT ON NUMERICAL COLUMNS
##CATEGORIZING COLUMNS WITH AND WITHOUT FLAGS
num_cols_withoutflag = []
num_cols_withflag = []
for cols in num_cols:
    if col.startswith("FLAG"):
        num_cols_withflag.append(col)
    else:
        num_cols_withoutflag.append(col)

for col in num_cols_withoutflag:
    print(app_dt[col].describe())
    plt.figure(figsize = [8,5] )
    sns.boxplot(data= app_dt,x=col)
    plt.show()
    print("------------")

###UNIVARIATE ANALYSIS ON COLUMNS WITH TARGET 0 AND 1
for col in cat_cols:
    print(f"Plot on {col} for Target 0 and 1")
    plt.figure(figsize=[10,7])
    plt.subplot(1,2,1)
    tar_0[col].value_counts(normalize=True).plot.bar()
    plt.Title("Target 0")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.Title("Target 1")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()
    print("\n\n-----------------\n\n")

######ANALYSIS ON AMT_GOODS_PRICE ON TARGET 0 AND 1
plt.figure(figsize=(10,6))
sns.distplot(tar_0['AMT_GOODS_PRICE'],label='tar_0',hist=False)
sns.distplot(tar_1['AMT_GOODS_PRICE'],label='tar_1',hist=False)
plt.legend()
plt.show()

###CONCLUSION:-THE PRICE OF THE GOODS FOR WHICH LOAN IS GIVEN HAS SAME VARIATION FOR TARGET 0 AND 1

#####BIVARIATE AND MULTIVARIATE ANALYSIS

##BIVARIATE ANALYSIS BETWEEN WEEKDAY_APPR_PROCESS_START VS HOUR_APPR_PROCESS_START
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.boxplot(x='WEEKDAY_APPR_PROCESS_START', Y='HOUR_APPR_PROCESS_START',data=tar_0)
plt.subplot(1,2,2)
sns.boxplot(x='WEEKDAY_APPR_PROCESS_START', Y='HOUR_APPR_PROCESS_START',data=tar_1)
plt.show()

#####BIVARIATE ANALYSIS BETWEEN AGE_CATEGORY VS AMT_CREDIT
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.boxplot(x='AGE_Category', Y='AMT_CREDIT',data=tar_0)
plt.subplot(1,2,2)
sns.boxplot(x='AGE_Category', Y='AMT_CREDIT',data=tar_1)
plt.show()


####PAIR PLOT OF AMOUNT COLUMNS FOR TARGET 0
sns.pairplot(tar_0[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']])
plt.show()

####PAIR PLOT OF AMOUNT COLUMNS FOR TARGET 1
sns.pairplot(tar_1[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']])
plt.show()

#####CO-RELATION BETWEEN NUMERICAL COLUMNS
corr_data = app_dt[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','YEARS_BIRTH','YEARS_EMPLOYED','YEARS_REGISTRATION','YEARS_ID_PUBLISH','YEARS_LAST_PHONE_CHANGE']]
print(corr_data.head(1))

plt.figure(figsize=(10,10))
sns.heatmap(corr_data.corr(),annot=True,cmap='RdYlGn')
plt.show()

####SPLIT THE NUMERICAL VARIABLES BASEWD ON TARGET 0 AND 1 TO FIND THE CORRELATION
corr_data_1 = tar_1[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','YEARS_BIRTH','YEARS_EMPLOYED','YEARS_REGISTRATION','YEARS_ID_PUBLISH','YEARS_LAST_PHONE_CHANGE']]
corr_data_0 = tar_0[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','YEARS_BIRTH','YEARS_EMPLOYED','YEARS_REGISTRATION','YEARS_ID_PUBLISH','YEARS_LAST_PHONE_CHANGE']]

###READ PREVIOUS APPLICATION CSV
#GET INFO AND SHAPE ON DATASET
papp_data = pd.read_csv("previous_application.csv")

####DATA QUALLITY CHECK
#CHECK FOR %NULL VALUES IN DATASET
Percentage = 49
threshold_p = int(((100-Percentage)/100)*papp_data.shape[0] + 1)
papp_dt = papp_data.dropna(axis=1, thresh=threshold_p)

##IMPUTE MISSING VALUES
#CHECK DTYPE OF MISSING VALUES IN APPLICATION DATASET BEFORE IMPUTING VALUES

for col in papp_dt.columns:
    if papp_dt[col].dtypes == np.int64 or papp_dt[col].dtypes == np.float64:
        papp_dt[col] = papp_dt[col].apply(lambda x: abs(x))

##VALIDATE IF ANY NULL VALUES PRESENT IN DATASET
null_Cols = list(papp_dt.columns[papp_dt.isna().any()])
len(null_Cols)

###BINNING OF CONTINUOUS VARIABLES
#BINNING AMT_CREDIT COLUMN
papp_dt['AMT_CREDIT_Category'] = pd.cut(papp_dt.AMT_CREDIT,[0,200000,400000,600000,800000,1000000],
                                       labels= ['Very low Credit','Low Credit','Medium Credit','High Credit','Very High Credit'])
papp_dt.AMT_CREDIT_Category.value_counts(normalize=True)*100
papp_dt['AMT_CREDIT_Category'].value_counts(normalize=True).plot.bar()
##CONCLUSION:- THE CREDIT AMOUNT OF THE LOAN FOR MOST APPLICANTS IS EITHER LOW(200000 TO 400000)
papp_dt['AMT_GOODS_PRICE_Category'].value_counts(normalize=True).plot.pie(autopct='%1.2f%%')
plt.legend()
plt.show()

###DATA IMBALANCE CHECK
#DIVIDING APPLICATION DATASET WITH NAME_CONTRACT_STATUS
approved = papp_dt[papp_dt.NAME_CONTRACT_STATUS == 'Approved']
cancelled = papp_dt[papp_dt.NAME_CONTRACT_STATUS == 'Canceled']
refused = papp_dt[papp_dt.NAME_CONTRACT_STATUS == 'Refused']
unused = papp_dt[papp_dt.NAME_CONTRACT_STATUS == 'Unused offer']
papp_dt.NAME_CONTRACT_STATUS.value_counts(normalize=True)*100
papp_dt['NAME_CONTRACT_STATUS'].value_counts(normalize=True).plot.pie(autopct='%1.2f%%')
plt.legend()
plt.show()
###CONCLUSION:- 62% OF THE APPLICANTS HAVE THE LOAN APPROVED, 17%REJECTED OR CANCELLED AND 2% ARE UNUSED

###UNIVARIATE ANALYSIS
cat_Cols = list(papp_dt.columns[papp_dt.dtypes == np.object_])
num_Cols = list(papp_dt.columns[papp_dt.dtypes == np.int64]) + list(papp_dt.columns[papp_dt.dtypes == np.float64])

##PLOTTING ON CATEGORICAL COLUMNS
for col in cat_Cols:
    print(papp_dt[col].value_counts(normalize=True)*100)
    plt.figure(figsize=[5,5])
    papp_dt[col].value_counts(normalize=True).plot.pie(labeldistance=None, auotpct='%1.2f%%')
    plt.legend()
    plt.show()
    print("---------------")

#PLOTTING ON NUMERICAL COLUMNS
for col in num_Cols:
    print(papp_dt[col].value_counts(normalize=True)*100)
    print('99th Percentile',np.percentile(papp_dt[col],99))
    print(papp_dt[col].describe())
    plt.figure(figsize=[10,6])
    sns.boxplot(data=papp_dt,x=col)
    plt.show()
    print("----------------")

####BIVARIATE AND MULTIVARIATE ANALYSIS
##BIVARIANT ANALYSIS BETWEEN WEEKDAY_APPR_PROCESS_START VS AMT_APPLICATION
plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START',y='AMT_APPLICATION',data=cancelled)
plt.title("Plot for cancelled")
plt.show()

plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START',y='AMT_APPLICATION',data=approved)
plt.title("Plot for approved")
plt.show()

plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START',y='AMT_APPLICATION',data=unused)
plt.title("Plot for unused offer")
plt.show()

plt.figure(figsize=[10,5])
sns.barplot(x='WEEKDAY_APPR_PROCESS_START',y='AMT_APPLICATION',data=refused)
plt.title("Plot for refused")
plt.show()

##BIVARIENT ANALYSIS BETWEEN AMT_ANNUITY VS AM_GOODS_PRICE
plt.figure(figsize=(15,10))
plt.subplot(1,4,1)
plt.title("Approved")
sns.scatterplot(x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=approved)
plt.figure(figsize=(15,10))
plt.subplot(1,4,1)
plt.title("Cancelled")
sns.scatterplot(x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=cancelled)
plt.figure(figsize=(15,10))
plt.subplot(1,4,1)
plt.title("Refused")
sns.scatterplot(x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=refused)
plt.figure(figsize=(15,10))
plt.subplot(1,4,1)
plt.title("Unused offer")
sns.scatterplot(x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=unused)
plt.show()

##  CONCLUSION:- 1)FOR LOAN STATUS AS APPROVED,REFUSED,CANCELLED AMOUNT OF ANNUITY INCREASES WITH GOODS PRICE
#2)FOR LOAN STATUS AS REFUSED IT HAS NO LINEAR RELATIONSHIP

###CORRELATION BETWEEN NUMERICAL COLUMNS
corr_approved = approved[['DAYS_DECISION','AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_GOODS_PRICE','CNT_PAYMENT']]
corr_refused = refused[['DAYS_DECISION','AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_CREDIT','AMT_GOODS_PRICE','CNT_PAYMENT']]
corr_cancelled = cancelled[['DAYS_DECISION','AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_CREDIT','AMT_GOODS_PRICE','CNT_PAYMENT']]
corr_unused = unused[['DAYS_DECISION','AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_CREDIT','AMT_GOODS_PRICE','CNT_PAYMENT']]

###CORRELATION FOR NUMERICAL COLUMNS FOR APPROVED
plt.figure(figsize=[10,10])
sns.heatmap(corr_approved.corr(),annot=True,cmap='Blues')
plt.title("Heat Map plot for Approved")
plt.show()

###CORRELATION FOR NUMERICAL COLUMNS FOR CANCELLED
plt.figure(figsize=[10,10])
sns.heatmap(corr_cancelled.corr(),annot=True,cmap='Blues')
plt.title("Heat Map plot for Cancelled")
plt.show()
###CORRELATION FOR NUMERICAL COLUMNS FOR REFUSED
plt.figure(figsize=[10,10])
sns.heatmap(corr_refused.corr(),annot=True,cmap='Blues')
plt.title("Heat Map plot for Refused")
plt.show()
###CORRELATION FOR NUMERICAL COLUMNS FOR UNUSED OFFER
plt.figure(figsize=[10,10])
sns.heatmap(corr_unused.corr(),annot=True,cmap='Blues')
plt.title("Heat Map plot for Unused offer")
plt.show()
#CONCLUSION:-FOR APPROVED CATEGORY
#1)AMT_APPLICATION HAS HIGHER CORRELATION WITH AMT_CREDIT AND AMT_GOODS_PRICE,AMT_ANNUITY
#2)DAYS_DECISION HAS NEGATIVE CORRELATION WITH AMT_GOODS_PRICE,AMT_CREDIT,AMT_APPLICATION,CNT_PAYMENT,AMT_ANNUITY


####MERGE THE APPLICATION AND PREVIOUS APPLICATION DATAFRAMES
merge_df = app_dt.merge(papp_dt, on =['SK_ID_CURR'],how='left')
merge_df.head(1)
merge_df.info()

###FILTERING REQUIRED COLUMNS FOR OUR ANALYSIS
for col in merge_df.columns:
    if col.startswith('FLAG'):
        merge_df.drop(columns=col,axis=1,inplace=True)


merge_df.shape()
res1 = pd.pivot_table(data=merge_df,
               index=['NAME_INCOME_TYPE', 'NAME_CLIENT_TYPE'],
               columns=['NAME_CONTRACT_STATUS'],
               values='TARGET',
               aggfunc='mean')
plt.figure(figsize=[10,10])
sns.heatmap(res1,annot=True,cmap='BuPu')
plt.show()

##CONCLUSION:
#1)APPLICANTS WITH INCOME TYPE MATERNITY LEAVE AND CLIENT TYPE NEW ARE HAVING MORE CHANCES OF GETTING THE LOAN APPORVED
#2)APPLICANTS WITH INCOME TYPE MATERNITY LEAVE ,UNEMPLOYED AND CLIENT TYPE REPEATER ARE HAVING MORE CHANCES OF GETTING THE LOAN CANCELLED
#3)APPLICANTS WITH INCOME TYPE MATERNITY LEAVE, UNEMPLOYED AND CLIENT TYPE REPEATER ARE HAVING MORE CHANCES OF GETTING THE LOAN REFUSED
#4)APPLICANTS WITH INCOME TYPE MATERNITY LEAVE AND CLIENT TYPE REPEATER,WORKING AND CLIENT TYPE NEW ARE NOT ABLE TO UTILIZE THE BANK'S OFFER

res2 = pd.pivot_table(data=merge_df, index=['CODE_GENDER','NAME_SELLER_INDUSTRY'],columns=['TARGET'],values='AMT_GOODS_PRICE',aggfunc='sum')
plt.figure(figsize=[10,10])
sns.heatmap(res2,annot=True,cmap='BuPu')
plt.show()
