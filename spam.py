#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 13:40:24 2018

@author: john
"""

import pandas as pd

##############################################################################
# Step 1.1: Understanding our dataset
##############################################################################

# Import the dataset into a pandas dataframe using the read_table method.
# Because this is a tab separated dataset we will be using '\t' as the value
# for the 'sep' argument which specifies this format.
# Also, rename the column names by specifying a list ['label, 'sms_message']
# to the 'names' argument of read_table().
spam_file = "spam_data/SMSSpamCollection"
spam_df = pd.read_table(spam_file, sep='\t', names=['label', 'sms_message'])

# Print the first five values of the dataframe with the new column names.
print spam_df.head(5)

##############################################################################
# Step 1.2: Data Preprocessing
##############################################################################

# Convert the values in the 'label' column to numerical values using
# map method as follows: {'ham':0, 'spam':1} This maps the 'ham' value to 0
# and the 'spam' value to 1.
spam_df.label = spam_df.label.map({'ham': 0, 'spam': 1})

# Also, to get an idea of the size of the dataset we are dealing with,
# print out number of rows and columns using 'shape'.
print spam_df.shape

##############################################################################
# Step 2.1: Bag of Words
#
# Use sklearn's Count Vectorizer to convert text to matrix of token counts
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
##############################################################################

##############################################################################
# Step 2.2: Implement Bag of Words from Scratch
##############################################################################

# Step 1: Convert all strings to their lower case form.
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = [s.lower() for s in documents]
print(lower_case_documents)

# Step 2: Removing all punctuations
import string
sans_punctuation_documents = []
for s in lower_case_documents:
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    sans_punctuation_documents.append(s)

# Step 3: Tokenization
# Tokenize the strings stored in 'sans_punctuation_documents' using
# the split() method. and store the final document set in
# a list called 'preprocessed_documents'
preprocessed_documents = [s.split() for s in sans_punctuation_documents]

# Step 4: Count Frequencies
from collections import Counter
frequency_list = [dict(Counter(words)) for words in preprocessed_documents]

##############################################################################
# Step 2.3: Implementing Bag of Words in scikit-learn
##############################################################################

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()


