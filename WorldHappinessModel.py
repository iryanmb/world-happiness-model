#!/usr/bin/env python
# coding: utf-8

# # Lab 8: Define and Solve an ML Problem of Your Choosing

# In[7]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns


# In this lab assignment, you will follow the machine learning life cycle and implement a model to solve a machine learning problem of your choosing. You will select a data set and choose a predictive problem that the data set supports.  You will then inspect the data with your problem in mind and begin to formulate a  project plan. You will then implement the machine learning project plan. 
# 
# You will complete the following tasks:
# 
# 1. Build Your DataFrame
# 2. Define Your ML Problem
# 3. Perform exploratory data analysis to understand your data.
# 4. Define Your Project Plan
# 5. Implement Your Project Plan:
#     * Prepare your data for your model.
#     * Fit your model to the training data and evaluate your model.
#     * Improve your model's performance.

# ## Part 1: Build Your DataFrame
# 
# You will have the option to choose one of four data sets that you have worked with in this program:
# 
# * The "census" data set that contains Census information from 1994: `censusData.csv`
# * Airbnb NYC "listings" data set: `airbnbListingsData.csv`
# * World Happiness Report (WHR) data set: `WHR2018Chapter2OnlineData.csv`
# * Book Review data set: `bookReviewsData.csv`
# 
# Note that these are variations of the data sets that you have worked with in this program. For example, some do not include some of the preprocessing necessary for specific models. 
# 
# #### Load a Data Set and Save it as a Pandas DataFrame
# 
# The code cell below contains filenames (path + filename) for each of the four data sets available to you.
# 
# <b>Task:</b> In the code cell below, use the same method you have been using to load the data using `pd.read_csv()` and save it to DataFrame `df`. 
# 
# You can load each file as a new DataFrame to inspect the data before choosing your data set.

# In[8]:


# File names of the four data sets
adultDataSet_filename = os.path.join(os.getcwd(), "data", "censusData.csv")
airbnbDataSet_filename = os.path.join(os.getcwd(), "data", "airbnbListingsData.csv")
WHRDataSet_filename = os.path.join(os.getcwd(), "data", "WHR2018Chapter2OnlineData.csv")
bookReviewDataSet_filename = os.path.join(os.getcwd(), "data", "bookReviewsData.csv")


df = pd.read_csv('data/WHR2018Chapter2OnlineData.csv')

df.head(10)


# ## Part 2: Define Your ML Problem
# 
# Next you will formulate your ML Problem. In the markdown cell below, answer the following questions:
# 
# 1. List the data set you have chosen.
# 2. What will you be predicting? What is the label?
# 3. Is this a supervised or unsupervised learning problem? Is this a clustering, classification or regression problem? Is it a binary classificaiton or multi-class classifiction problem?
# 4. What are your features? (note: this list may change after your explore your data)
# 5. Explain why this is an important problem. In other words, how would a company create value with a model that predicts this label?

# #### 1) I chose the 2018 World Happiness Report dataset.
# 2) I will be predicting a country's happiness index. Specifically the label 'Life Ladder'.
# 3) This is a supervised learning problem where the output (y) variable is a continuous value. Thus a regression problem.
# 4) We can find the features by dropping the label, using df.drop('Life Ladder', axis=1). Revealing a total of 18 columns, or 18 features.
# 5) This problem is important because we can outline which features contribute the most to a country's happiness. This data analysis can inform policy decisions by making targeted interventions in areas to improve a country's overall happiness. Commercially, organizations centered around well-being, social development, and community can use such models as well to drive their country's happiness.

# #### Part 3: Understand Your Data
# 
# The next step is to perform exploratory data analysis. Inspect and analyze your data set with your machine learning problem in mind. Consider the following as you inspect your data:
# 
# 1. What data preparation techniques would you like to use? These data preparation techniques may include:
# 
#     * addressing missingness, such as replacing missing values with means
#     * finding and replacing outliers
#     * renaming features and labels
#     * finding and replacing outliers
#     * performing feature engineering techniques such as one-hot encoding on categorical features
#     * selecting appropriate features and removing irrelevant features
#     * performing specific data cleaning and preprocessing techniques for an NLP problem
#     * addressing class imbalance in your data sample to promote fair AI
#     
# 
# 2. What machine learning model (or models) you would like to use that is suitable for your predictive problem and data?
#     * Are there other data preparation techniques that you will need to apply to build a balanced modeling data set for your problem and model? For example, will you need to scale your data?
#  
#  
# 3. How will you evaluate and improve the model's performance?
#     * Are there specific evaluation metrics and methods that are appropriate for your model?
#     
# 
# Think of the different techniques you have used to inspect and analyze your data in this course. These include using Pandas to apply data filters, using the Pandas `describe()` method to get insight into key statistics for each column, using the Pandas `dtypes` property to inspect the data type of each column, and using Matplotlib and Seaborn to detect outliers and visualize relationships between features and labels. If you are working on a classification problem, use techniques you have learned to determine if there is class imbalance.
# 
# <b>Task</b>: Use the techniques you have learned in this course to inspect and analyze your data. You can import additional packages that you have used in this course that you will need to perform this task.
# 
# <b>Note</b>: You can add code cells if needed by going to the <b>Insert</b> menu and clicking on <b>Insert Cell Below</b> in the drop-drown menu.

# In[9]:


df.describe()


# In[10]:


df.dtypes # Data Types ('object' data type will not add any predictive power, so I will drop that feature)


# In[11]:


df.isnull().mean().sort_values(ascending=False) #Missing Values (0.6267 is a LOT—this feature HAS to get dropped. 0.2286 may get dropepd too)


# In[12]:


df.corr()['Life Ladder'].sort_values(ascending=False) #Correlations btwn Label & Features (values near 0 don't correlate much with label)


# In[13]:


#We need to find outliers, so I am going to write a function that calculates the IQR and finds outliers accordingly
outlier_dict = {}
def find_outliers(data):
    for col in df.select_dtypes('float64').columns: #loop through each column of floats
        Q1 = df[col].quantile(0.25) #first quartile
        Q3 = df[col].quantile(0.75) #third quartile
        IQR = Q3 - Q1 #interquartile range
        lower_bound = Q1 - (1.5 * IQR) #Lower Bound
        upper_bound = Q3 + (1.5 * IQR) #Upper Bound
        outliers = data[(data[col] < lower_bound)|(data[col] > upper_bound)] #Anything less than lower bound, or more than upperbound, is outleir
        outlier_dict[col] = len(outliers) #Set key,value pair for column and their # of outliers

    def get_outlier_count(pair): #Function to return the value in the dictionary—the # of outliers
        return pair[1]

    return sorted(outlier_dict.items(), key=get_outlier_count, reverse=True) #return a sorted tuple of key,value pairs of columns and their # of outleirs

find_outliers(df) #This dictionary outputs each feature, and their number of outliers


# In[14]:


#I created a DataFrame so I can visualize each feature and their missing values and outliers
data_summary = pd.DataFrame({
    'Data Points': df.count(),
    'Missing Values': df.isnull().sum(),
    'Pct of Mising Values': df.isnull().mean().sort_values(ascending=False),
    'Outliers': pd.Series(dict(find_outliers(df))),
    'Corr w/ Label': df.corr()['Life Ladder'],
    
})

data_summary


# In[15]:


#'Perceptions of corruption' has 130 (!!!) outliers. I'm going to visualize so I know if I need to imputate this column

plt.figure(figsize=(12,2))
sns.boxplot(x=df['Perceptions of corruption'])

#This feature has a lot of outliers, but I want to keep them. I don't want to penalize a country for having a low perception of corruption.
#So I will train the model with a Tree-Based model because they handle outliers and missing data well.


# In[16]:


plt.figure(figsize=(10,2))
sns.boxplot(x=df['gini of household income reported in Gallup, by wp5-year'])


# In[17]:


df['gini of household income reported in Gallup, by wp5-year'].skew()


# ## Part 4: Define Your Project Plan
# 
# Now that you understand your data, in the markdown cell below, define your plan to implement the remaining phases of the machine learning life cycle (data preparation, modeling, evaluation) to solve your ML problem. Answer the following questions:
# 
# * Do you have a new feature list? If so, what are the features that you chose to keep and remove after inspecting the data? 
# * Explain different data preparation techniques that you will use to prepare your data for modeling.
# * What is your model (or models)?
# * Describe your plan to train your model, analyze its performance and then improve the model. That is, describe your model building, validation and selection plan to produce a model that generalizes well to new data. 

# 1) I have a new feature list. I removed the 'country' feature because it was the only string in a dataframe of numbers, and it doesn't add any predictive value; it's more of an index. I removed 'year' for a similar reason, because it reflects time and so does not add value when predicting. I removed 'GINI index (World Bank estimate)' because it is missing 62.6761% of data points. I dropped 'Confidence in national government' because it has essentially no correlation with the label (-0.086), and is missing 161 values. It's a weak predictor. Im dropping 'Standard deviation of ladder by country-year' because it's weak correlation (-0.154). I am also dropping 'GINI index (World Bank estimate), average 2000-15', because it's correlation with the label is only -0.172745, it's missing 176 values and has 25 outliers. It's also a redundant feature because we already have 'gini of household income reported in Gallup, by wp5-year'.
# 2) When it comes to data preparation, i immediately noticed that 'gini of household income reported in Gallup, by wp5-year' has 357 outliers, missing  22.9% of values. I used df['gini...'].skew() and found a value of 0.876, then plotted the feature on a boxplot to visualize the skew and concluded it was a right skew. And so I am going to imputate with the median. There are only 13 outliers, so they won't skew the mean too much, and so i could theoretically imputate with the mean, but a 0.876 skew is very high and so i will play it safe and use the median. I chose to imputate instead of deleting the feature because it has a good correlation with the label (-0.294). Moreover, 'Perception of corruption' has 130 outliers which is a red flag, but it's correlation with the label is -0.425 so I was going to keep the feature no matter what. I considered imputating the outliers, but I felt as though that was valubale data. If a country believes their perception of corruption is low, they should not be punished. And so when i train the model, I will use a Tree-Based model because they handle missing values and outliers well.
# 3) I will use tree-based models because they handle outliers and missing data values well. Otherwise, the model will be prone to overfitting because of the high amount of outliers in 'Perception of corruption.'
# 4) I will split the data into training, validation, and testing (60/20/20). I will first train a basic decision tree regressor, then advance to a more powerful ensemble method. Random Forest Regressor, then Gradient Boosting Regressor. They are good at handling to outliers. I will evaluate the model using Root Mean Squared Eroor (RMSE) and R^2 Score. I will then tune hyperparameter's using grid search. I will use cross-validation (10 folds) to ensure the model performs well across different subsets of data. That way, I avoid overfitting. After comparing the model performance across the difference tree-based candidates, I will select the model that performs the best.

# ## Part 5: Implement Your Project Plan
# 
# <b>Task:</b> In the code cell below, import additional packages that you have used in this course that you will need to implement your project plan.

# In[18]:


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# <b>Task:</b> Use the rest of this notebook to carry out your project plan. 
# 
# You will:
# 
# 1. Prepare your data for your model.
# 2. Fit your model to the training data and evaluate your model.
# 3. Improve your model's performance by performing model selection and/or feature selection techniques to find best model for your problem.
# 
# Add code cells below and populate the notebook with commentary, code, analyses, results, and figures as you see fit. 

# In[34]:


#Drop Features (Low correlation with label, no predictive value, etc.)
drop_cols = ['country',
             'year',
             'GINI index (World Bank estimate)',
             'Standard deviation of ladder by country-year',
             'GINI index (World Bank estimate), average 2000-15',
            'Confidence in national government']

df_cleaned = df.drop(columns=drop_cols)

df_cleaned.columns


# In[36]:


#Rename features so they're easy to manipulate
df_cleaned.rename(columns={'gini of household income reported in Gallup, by wp5-year':'gini_house_income',
                          'Perceptions of corruption':'corruption',
                           'Confidence in national government': 'confidence',
                            'Healthy life expectancy at birth': 'life_expectancy',
                           'Standard deviation/Mean of ladder by country-year': 'ladder_std_mean',
                           'Log GDP per capita':'log_gdp',
                           'Freedom to make life choices':'freedom',
                           'Positive affect':'pos_affect',
                           'Negative affect':'neg_affect',
                           'Democratic Quality':'democracy',
                           'Delivery Quality':'delivery',
                           'Social support':'social_support',
                          'Life Ladder':'life_ladder'},
                            inplace=True)
df_cleaned.columns


# In[42]:


#Impuate Values with median
for col in df_cleaned.columns:
    if df_cleaned[col].isnull().sum() > 0:
        # Use median for right-skewed or outlier-sensitive features
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

df_cleaned.isnull().sum()


# In[45]:


#Define Features & Label
X = df_cleaned.drop(columns='life_ladder')
y = df_cleaned['life_ladder']

#Train/Test/Split 60/20/20
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.4,random_state=1234)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp, test_size=0.5,random_state=1234)


# In[51]:


#Train Decision Tree
dt = DecisionTreeRegressor(random_state=1234)
dt.fit(X_train,y_train)

y_pred_val = dt.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2 = r2_score(y_val, y_pred_val)

print(f"Decision Tree RMSE: {rmse:.4f}") #.4f is so we round to four decimal places
print(f"Decision Tree R²: {r2:.4f}")


# In[57]:


#Train More Powerful Models
# Random Forest
rf = RandomForestRegressor(random_state=1234)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_val)

print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_val, rf_preds)):.4f}")
print(f"Random Forest R²: {r2_score(y_val, rf_preds):.4f}")

# Gradient Boosting
gb = GradientBoostingRegressor(random_state=1234)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_val)

print(f"Gradient Boosting RMSE: {np.sqrt(mean_squared_error(y_val, gb_preds)):.4f}")
print(f"Gradient Boosting R²: {r2_score(y_val, gb_preds):.4f}")


# In[62]:


#From our 3 models, Random Forest performs the best because it's RMSE is closest to 0, and it's R^2 score is closest to 1.
#A R^2 Score too close to 1 can lead to overfitting, and same with RMSE too close to 0, so we must be mindful.
#But overall, our model is performing well! Let's try Random Forest on our test data


# In[64]:


final_model = rf
test_preds = final_model.predict(X_test)

rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))
r2_test = r2_score(y_test, test_preds)

print(f"Test RMSE: {rmse_test:.4f}")
print(f"Test R²: {r2_test:.4f}")


# TakeawaOur machine learning model, trained using Random Forest Regression, effectively predicts national happiness levels with high accuracy. With a test R² of 0.894 and RMSE of 0.3578, it captures the majority of variability in happiness outcomes, suggesting that features such as GDP, freedom, social support, and others are strong predictors of national well-being.”

# In[ ]:




