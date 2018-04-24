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
print 'first five rows of spam dataframe:'
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
print '\nshape of spam dataframe:'
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
print '\ndocuments converted to lower case:'
print lower_case_documents

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

# Fit your document dataset to the CountVectorizer object you have created
# using fit(), and get the list of words which have been categorized
# as features using the get_feature_names() method
count_vector.fit(documents)
print '\ncount vectorizer feature names:'
print list(enumerate(count_vector.get_feature_names()))

# Create a matrix with the rows being each of the 4 documents, and
# the columns being each word. The corresponding (row, column) value is
# the frequency of occurrence of that word(in the column) in
# a particular document(in the row). You can do this using
# the transform() method and passing in the document data set as the argument.
# The transform() method returns a matrix of numpy integers, you can convert
# this to an array using toarray(). Call the array 'doc_array'
doc_array = count_vector.transform(documents).toarray()
print '\ndoc_array:'
print doc_array

# Convert the array we obtained, loaded into 'doc_array', into a dataframe and
# set the column names to the word names(which you computed earlier
# using get_feature_names(). Call the dataframe 'frequency_matrix'
frequency_array = pd.DataFrame(doc_array, columns=count_vector.get_feature_names())
print '\ndoc_array converted to dataframe (frequency_array):'
print frequency_array.head()

##############################################################################
# Step 3.1: Training and testing sets
##############################################################################
# Split the dataset into a training and testing set by using
# the train_test_split method in sklearn.
# Split the data using the following variables:
#
#   X_train is our training data for the 'sms_message' column.
#   y_train is our training data for the 'label' column
#   X_test is our testing data for the 'sms_message' column.
#   y_test is our testing data for the 'label' column Print out
#       the number of rows we have in each our training and testing data.
from sklearn.cross_validation import train_test_split

X = spam_df['sms_message']
y = spam_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print '\nNumber of rows in the total set: {}'.format(spam_df.shape[0])
print 'Number of rows in the training set: {}'.format(X_train.shape[0])
print 'Number of rows in the test set: {}'.format(X_test.shape[0])

##############################################################################
# Step 3.2: Applying Bag of Words processing to our dataset
##############################################################################
count_vector = CountVectorizer()
# Fit training set and transform
training_data = count_vector.fit_transform(X_train)
# Just transform the test set
testing_data = count_vector.transform(X_test)

##############################################################################
# Step 4.1: Bayes Theorem implementation from scratch
##############################################################################
# Given:
#   P(Pos|Disease) = "Sensitivity" or True Positive Rate
#   P(Neg|~Disease) = "Specificity" or True Negative Rate
#       NOTE: P(Pos|~Disease) = 1 - P(Neg|~Disease)
#   P(Disease) = (a given prior probability)
#
# Find the other prior probability:
#   P(Pos) = P(D)*Sensitivity + P(~D)*(1 - Specificity)
#          = P(D)*P(Pos|D) + (1 - P(D))*(1 - P(Neg|~Disease))
#          = P(D)*P(Pos|D) + P(~D)*P(Pos|~D)
#
# Find the posterior probability:
#   P(A and B) = P(A)P(B|A) = P(B)P(A|B)
#   ==> P(B|A) = P(B)P(A|B) / P(A)
#   so,
#   P(D|Pos) = P(D)*P(Pos|D) / P(Pos)
#            = P(D)*P(Pos|D) / [P(D)*P(Pos|D) + (1 - P(D)*(1 - P(Neg|~D)))]

##############################################################################
# Step 4.2: Naive Bayes implementation from scratch
##############################################################################
# (See handwritten notes or Bayesian_Inference.ipynb)

##############################################################################
# Step 5: Naive Bayes implementation using scikit-learn
##############################################################################
# for discrete features (like word counts for text classification), use the
# multinominal Naive Bayes implementation
# for continuous features, use Gaussian Naive Bayes
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)

##############################################################################
# Step 6: Evaluating our model
##############################################################################
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print '\nAccuracy score: ', accuracy_score(predictions, y_test)
print 'Precision score: ', precision_score(predictions, y_test)
print 'Recall score: ', recall_score(predictions, y_test)
print 'F1 score: ', f1_score(predictions, y_test)
