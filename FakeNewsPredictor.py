#!/usr/bin/env python
# coding: utf-8

# Load packages and libraries
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
# Checks number of times a word appears in text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
#from sklearn.model_selection import train_test_split

# Read the CSV as a Pandas dataframe with UTF-8 encoding
with open('news.csv', encoding = 'utf8') as file:
    news = pd.read_csv(file)
    # Changing text in title and text columns to be all uppercase
    news['title'] = news['title'].str.upper()
    news['text'] = news['text'].str.upper()

# Display the shape, data types, and first few records of data set
print(news.shape)
print(news.dtypes)
news.head()

# Convert data types to strings because we can't perform string operations on objects
#news[['title', 'text', 'label']] = news[['title', 'text', 'label']].astype('|S')

#news.dtypes
# Causing errors where no ASCII charachter exists for a UTF-8 character in the text. Maybe I don't need to do this?

# Drop id field as it is not useful
news.drop('id', axis = 1, inplace = True)

news.head()

# Check for any NaNs
news.isnull().values.any()

#Checking which columns have NaNs
print(news['title'].isnull().sum())
print(news['text'].isnull().sum())
print(news['label'].isnull().sum())

# There is no way to fill NaNs in this data set so I will remove all rows with NaNs
news = news.dropna()

# Check to make sure NaNs were removed
print(news.isnull().values.any())

# Check new shape of dataframe
news.shape

# Checking for duplicate articles by title
print("Duplicate titles: ", news.duplicated(subset = ['title']).sum())
# Checking for duplicate articles by text
print("Duplicate text: ", news.duplicated(subset = ['text']).sum())

# Sorting by title
news.sort_values("title", inplace = True)

# Dropping duplicate titles
news.drop_duplicates(subset = "title", keep = 'first', inplace = True)

news.shape

# Sorting by text
news.sort_values("text", inplace = True)

# Dropping duplicate text
news.drop_duplicates(subset = "text", keep = 'first', inplace = True)

news.shape

# Check the values in label field
news.label.unique()

# Replace 0 with REAL
news['label'] = news['label'].replace(['0'],'REAL')
# Replace 1 with FAKE
news['label'] = news['label'].replace(['1'],'FAKE')

news.label.unique()

# Create the training testing data sets
train = news.sample(frac = 0.7)
test = news.drop(train.index)

display(train)
display(test)

# Create training and testing numpy arrays for both the input variables and the target variables
train_X = np.asarray(train.drop('label', axis = 1))
train_y = np.asarray(train.label)

test_X = np.asarray(test.drop('label', axis = 1))
test_y = np.asarray(test.label)