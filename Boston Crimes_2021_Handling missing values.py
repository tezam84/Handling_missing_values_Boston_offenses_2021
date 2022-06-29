#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info">
# <b>Import librairies
# </div>

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# <div class="alert alert-block alert-info">
# <b>Load dataset 
# </div>

# In[8]:


df=pd.read_csv('Crime incident Boston.csv')


# In[9]:


df.shape


# <font color='green'>There are 17 columns and 71,721 observations</font>

# <div class="alert alert-block alert-info">
# <b>We can observe the dataset using the head()function, which returns the first ten records from the dataset
# </div>

# In[10]:


df.head(10)


# <font color='green'>NaN means Not a number or missing values <br> We can see that the columns OFFENSE_CODE_GROUP and UCR_PART have missing values</font>

# <div class="alert alert-block alert-info">
# <b>The info() method shows some of the characteristics of the data 
# </div>

# In[11]:


df.info()


# <font color='green'>We can see that we have mainly object variables columns and missing values</font>

# <div class="alert alert-block alert-info">
# <b></b> Data Cleaning

# #### Removing duplicates and finding missing values are important, otherwise our models can lead us to incorrect conclusions

# In[12]:


duplicate_Values=df.duplicated()
print(duplicate_Values.sum())
df[duplicate_Values]


# <font color='green'>There are no duplicate variables</font>

# #### Let explore the missing values

# In[13]:


print(df.isnull().sum())


# In[14]:


# Count missing values in the dataset
print(df.isnull().values.sum())


# <font color='green'>We see there are 145,114 of total missing values. The variables concerned are OFFENSE_CODE_GROUP, DISTRICT, UCR_PART and STREET</font>

# #### Let's go deeper in the analysis of the missing values. <br>The missingno Library provides a series of visualisations to understand the presence and distribution of missing data within a pandas dataframe. <br>There are four types of plots for visualising data completeness: the barplot, the matrix plot and the dendrogram plot.

# In[15]:


import missingno as msno 


# In[16]:


msno.matrix(df)


# #### The picture shows the amount and positions of missing values. The idea is to capture not only missing values but also data sparsity.

# <font color='green'> The biggest variables concerned by missing values are OFFENSE_CODE_GROUP and UCR_PART, then DISTRICT and STREET which confirm our previous analysis</font>

# #### The sparkline to the right highlights the rows in the dataset with the highest of 15 and lowest nullity of 13

# In[17]:


msno.bar(df)


# In[18]:


msno.dendrogram(df)


# #### This is a dendrogram, which uses a hierarchical clustering algorithm to bin variables against one another by their nullity correlation measured in terms of binary distance

# #### If columns are grouped together at level zero, it shows a strong correlation between them due to the presence or absence of null values. If they are grouped much further from zero, then the correlation is much less likely.

# <font color='green'>The variables OFFENSE_CODE, INCIDENT_NUMBER,OFFENSE_DESCRIPTION, REPORTING_AREA, SHOOTING, OCCURED_ON_DATE, YEAR, MONTH, DAY_OF_THE_WEEK, HOUR and Lat have a strong correlation. </font>

# In[19]:


# get the number of missing data points per column
missing_values_count = df.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]


# In[20]:


# how many total missing values do we have?
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100


# <font color='green'> We have approximately 12% of missing values</font> 

# #### Let's remove all columns with at least one missing value

# In[21]:


columns_with_na_dropped = df.dropna(axis=1)
columns_with_na_dropped.head()


# <font color='green'> To solve our problem , I remove the following columns we do not need, along with rows with missing values</font> 

# In[22]:


df.drop(['OFFENSE_CODE_GROUP','OFFENSE_CODE','DISTRICT', 'UCR_PART', 'STREET','OCCURRED_ON_DATE','INCIDENT_NUMBER', 'Long'],axis=1)


# #### If we specify the parameter axis=1, it will delete all the columns with at least one missing element. <br>If we specify the parameter axis=0, it will delete all the rows with at least one missing element. This deletion is the default behavior.

# In[23]:


df.UCR_PART.fillna('none', inplace=True)
df['UCR_PART'].unique()


# In[24]:


df.OFFENSE_CODE_GROUP.fillna('none', inplace=True)
df['OFFENSE_CODE_GROUP'].unique()


# In[25]:


df.info()


# <font color='green'> Our dataframe is free from missing values <br> MISSION ACCOMPLISHED</font> 
