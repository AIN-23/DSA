#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importing all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


## Loading the dataset

data=pd.read_csv(r"C:\Users\Workstation X\Desktop\ICT\Exit_Exam\train_exit.csv")


# # Getting general information on the dataset

# In[3]:


## To display first 6 rows

data.head(6)


# In[4]:


## To print the count of rows and columns

data.shape


# In[5]:


# There are 13994 rows and 30 columns


# In[6]:


## To print all basic information of this dataset

data.info()


# In[7]:


## To display the datatypes of this dataset

data.dtypes


# In[8]:


## To print the names of columns

data.columns


# In[9]:


## To display the no:of different parameters in a particular column

data['selection'].value_counts()


# In[10]:


## To display the unique categories in a particular column

data['gender'].unique()


# In[11]:


## To display the no:of unique categories in a particular column

data['gender'].nunique()


# In[12]:


## To get a statistical information on numeric columns

data.describe()


# In[13]:


## To get description on categorical columns

data.describe(include='O')


# In[14]:


## To get the correlation

data.corr()


# In[62]:


## Get the correlation matrix using heatmap

sns.heatmap(data.corr(),annot=True, cmap='rocket_r')
plt.xticks(rotation=45)
sns.set(rc={'figure.figsize':(15,15)})


# # EDA

# In[64]:


## Display the count plot of male and female.

sns.countplot(x='gender', data=data)
plt.figure(figsize=(5,5))


# In[17]:


# This plot shows that the count of male and female is quite the same.


# In[18]:


## Display the count plot of male and female with regards to 'dedication_level' .

sns.countplot(x='gender', data=data, hue ='dedication_level')


# In[19]:


## Display the count plot of male and female with regards to 'strong_foot'.

sns.countplot(x='gender', data=data, hue ='strong_foot')


# In[20]:


## Display the count plot of male and female with regards to 'coaching'.

sns.countplot(x='gender', data=data, hue ='coaching')


# In[21]:


## To display the histogram of all numeric columns

data.hist(figsize=(20,15))


# In[65]:


## To display the scatterplot of 'matched played' vs 'trophies won'

sns.scatterplot(x='matches_played', y='trophies_won', data=data)
plt.figure(figsize=(5,5))


# In[23]:


# The above plot shows that no direct relation can be found out from this scatter plot


# # Pre-processing steps

# In[24]:


## To check if there are any null values

data.isnull().sum()


# In[25]:


## To display columns with missing values - just checking with another method
columns_with_missing_values = data.columns[data.isna().any()].tolist()

print("Columns with missing values:")
for column in columns_with_missing_values:
    print(column)


# In[26]:


# The above 2 codes confirms that there are 13 columns with missing/null values


# In[27]:


## Deleting the 19 rows from column 'gender' which has missing values

data.dropna(subset=['gender'], inplace=True)


# In[28]:


data.shape


# In[29]:


## Converting the object type column 'weight' to int type

data['weight'] = pd.to_numeric(data['weight'].str.replace('lbs', ''), errors='coerce')


# In[30]:


data.head()


# In[31]:


## Filling the missing rows of numeric columns with mean values of the respective column

data['weight'].fillna(data['weight'].mean(),inplace=True)
data['ball_controlling_skills'].fillna(data['ball_controlling_skills'].mean(),inplace=True)
data['jumping_skills'].fillna(data['jumping_skills'].mean(),inplace=True)
data['penalties_conversion_rate'].fillna(data['penalties_conversion_rate'].mean(),inplace=True)
data['mental_strength'].fillna(data['mental_strength'].mean(),inplace=True)
data['shot_accuracy'].fillna(data['shot_accuracy'].mean(),inplace=True)
data['behaviour_rating'].fillna(data['behaviour_rating'].mean(),inplace=True)
data['matches_played'].fillna(data['matches_played'].mean(),inplace=True)
data['fitness_rating'].fillna(data['fitness_rating'].mean(),inplace=True)
data['years_of_experience'].fillna(data['years_of_experience'].mean(),inplace=True)


## Filling the missing rows of object type columns with mode values of the respective column

data['strong_foot'].fillna(data['strong_foot'].mode()[0],inplace=True)
data['coaching'].fillna(data['coaching'].mode()[0],inplace=True)


# In[32]:


## To confirm if there are any null values

data.isnull().sum()


# In[33]:


## Box plot for numeric columns

# Storing the numeric columns in a variable
outlier_features = data.select_dtypes(include=['float64', 'int64']).columns


# In[34]:


outlier_features


# In[35]:


plt.figure(figsize=(20,20))
sns.boxplot(data[list(outlier_features)],palette='plasma')
plt.title("Features with Outliers")
plt.xticks(rotation=45)


# In[36]:


data['no_of_disqualifications'].nunique()


# In[37]:


data['no_of_disqualifications'].value_counts()


# In[38]:


# Replace -999 with NaN to exclude them from the mean calculation
data['no_of_disqualifications'] = data['no_of_disqualifications'].replace(-999, np.nan)

# Calculate the mean excluding NaN values
mean_value = data['no_of_disqualifications'].mean()

# Replace -999 with the calculated mean
data['no_of_disqualifications'].fillna(mean_value, inplace=True)


# In[39]:


## Treating the outliers with mean values

# Calculate IQR
Q1=data[outlier_features].quantile(0.25)
Q3=data[outlier_features].quantile(0.75)
IQR=Q3-Q1

# Identify outliers using the IQR method
outliers = ((data[outlier_features] < (Q1 - 1.5 * IQR)) | (data[outlier_features] > (Q3 + 1.5 * IQR)))

# Replace outliers with the mean values of the respective columns
data[outlier_features] = data[outlier_features].mask(outliers, data[outlier_features].mean(),axis=1)


# Display box plots to check for outliers
plt.figure(figsize=(20, 10))
sns.boxplot(data=data[outlier_features])
plt.title('Box Plot of Columns after Outlier Replacement')
plt.xlabel('Columns')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()


# In[40]:


data.head()


# In[41]:


## Deleting 3 columns as it won't impact the dataset.

data.drop('player_id', axis=1, inplace=True)
data.drop('name', axis=1, inplace=True)
data.drop('country', axis=1, inplace=True)

data.head()


# # Encoding the data

# In[42]:


from sklearn.preprocessing import LabelEncoder

# List of categorical columns
categorical_columns = ['gender', 'strong_foot', 'dedication_level', 'coaching']

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for column in categorical_columns:
    if column in data.columns:
        data[column] = label_encoder.fit_transform(data[column])

# Display the DataFrame after label encoding
data.head()


# In[43]:


## Converting the column height to int type

# Convert the 'height' column to inches
data['height'] = data['height'].apply(lambda x: int(x.split('\'')[0]) * 12 + int(x.split('\'')[1].replace('"', '')))

data.head(4)


# # Scaling the dataset

# In[44]:


from sklearn.preprocessing import Normalizer

# Get the column name for the future use
feature_columns = data.iloc[:,:-1]
target_column=data['selection']

# Initiatiate and fit scaler
scaler = Normalizer()
data_scaled = scaler.fit_transform(feature_columns)

# Embedding the scaled data into the dataset
data_scaled_new = pd.DataFrame(data_scaled,columns=feature_columns.columns)

data_scaled_new.head()


# # Perform Classification

# In[45]:


# Split the data into features and target

X=data_scaled_new
y=target_column


# In[46]:


# Split the data into training and test data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=0)


# # Logistic Regression

# In[48]:


# Create a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_lr = model.predict(X_test)

# Calculate the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Accuracy of the Logistic Regression model: {accuracy_lr * 100:.2f}%')


# # Support Vector Classifier

# In[49]:


from sklearn.svm import SVC

# Create and train a Support Vector Classifier (SVC)
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)


# Make predictions on the test set
svc_predictions = svc_classifier.predict(X_test)


# Evaluate the accuracy of the classifier
from sklearn.metrics import accuracy_score
svc_accuracy = accuracy_score(y_test, svc_predictions)*100
print("Support Vector Classifier Accuracy:" + str(round(svc_accuracy,2))+"%")


# # KNN Classification

# In[50]:


# Fitting clasifier to the Training set
from sklearn.neighbors import KNeighborsClassifier

# Instantiate learning model (k = 20)
classifier = KNeighborsClassifier(n_neighbors=20)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier.predict(X_test)

# Evaluate the model

from sklearn.metrics import accuracy_score
accuracy_knn=accuracy_score(y_test,y_pred_knn)*100
print("Accuracy of our model is equal to " + str(round(accuracy_knn,2))+"%")


# # Random Forest

# In[52]:


# import Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
# instantiate the classifier
rfc = RandomForestClassifier(random_state=42)
# fit the model
rfc.fit(X_train, y_train)

# Predict the Test set results
y_pred_rfc = rfc.predict(X_test)


## Check accuracy score
from sklearn.metrics import accuracy_score

acc_rfc = accuracy_score(y_test,y_pred_rfc)
print('Model accuracy score: ',str(round(acc_rfc*100,2))+"%")


# # Decision Tree Classifier

# In[53]:


## Build Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", max_depth=9)


# Train Decision Tree Classifer
clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred_dtc = clf.predict(X_test)


# Evaluating the model

from sklearn import metrics
acc_dtc= metrics.accuracy_score(y_test, y_pred_dtc)*100
print("Accuracy is "+str(round(acc_dtc,2))+"%")


# # Ada Boost Classification

# In[54]:


from sklearn.ensemble import AdaBoostClassifier

abc_model=AdaBoostClassifier(n_estimators=50)

# Train Decision Tree Classifer
abc_model.fit(X_train,y_train)

# Predict the response for test dataset
y_pred_abc_model = abc_model.predict(X_test)


# Evaluating the model

from sklearn import metrics
acc_abc= metrics.accuracy_score(y_test, y_pred_abc_model)*100
print("Accuracy is "+str(round(acc_abc,2))+"%")


# # Naive Bayes Classification

# In[55]:


from sklearn.naive_bayes import GaussianNB

model= GaussianNB()

# Train Decision Tree Classifer
model.fit(X_train,y_train)

# Predict the response for test dataset
y_pred_model = model.predict(X_test)


# Evaluating the model

from sklearn import metrics
acc_nb= metrics.accuracy_score(y_test, y_pred_model)*100
print("Accuracy is "+str(round(acc_nb,2))+"%")


# # Comparison

# In[56]:


compare = pd.DataFrame({'Model': ['K-Neighbors','Logistic Regression','SVC','Random Forest','Decision Tree Classifier','Ada Boost Classifier','Naive Bayes'],
                        'Accuracy (in %)': [accuracy_knn, accuracy_lr*100, svc_accuracy, acc_rfc*100, acc_dtc,acc_abc,acc_nb]})

compare


# In[ ]:


## Check points
#  Checked in another notebook with normalising the data and then encoding. Provided almost similar results
#  Tried with different random_state and chose the appropriate one.


# In[ ]:


## Going with Random Forest Classification since that gave better accuracy score.


# # Fine tuning using GridSearch technique for Random Forest Classifier

# In[61]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Create a Random Forest Classifier
rfc_new = RandomForestClassifier(random_state=0)

# Define the hyperparameter distribution
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

# Perform Randomized Search
randomized_search = RandomizedSearchCV(rfc_new, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=0)
randomized_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = randomized_search.best_params_

# Train a new model with the best hyperparameters
best_rf_model = RandomForestClassifier(**best_params)
best_rf_model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = best_rf_model.score(X_test, y_test)*100
print("Accuracy:", str(round(accuracy,2))+"%")


# In[70]:


first_row_array = data_scaled_new.iloc[0].values
print(first_row_array)


# # Prediction

# In[71]:


k=[[0.08203746, 0.00356685, 0.26394661, 0.62776492, 0.27464715, 0.20687707,0.16764177, 0.21044392, 0.14624069, 0.26037977, 0.046369,0.24611238,0.24967923, 0.12840646, 0.17834231, 0.25681292, 0.08703105, 0.01070054, 0.02746472, 0.0139592,0.02889145, 0.00713369, 0,0.00713369, 0.02853477,0]]

# Make Predictions 
predictions_check=best_rf_model.predict(k)

# Print the predictions for first row
print("The predicted data is : ", predictions_check)


# In[ ]:


## The prediction is checked and is correct for first row.

