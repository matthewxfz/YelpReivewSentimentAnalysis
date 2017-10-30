#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#support spyder
#%%
import pandas as pd 
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#%%
data = pd.read_csv( "/Users/matthewxfz/Workspaces/gits/YelpReivewSentimentAnalysis/data/yelp810.csv", header=0, 
 delimiter=",", quoting=3 )

ran = np.random.rand(len(data));
msk1 = ran < 0.8
msk2 = ran > 0.2

train = data[msk1]
train = train.reset_index(drop=True)
test =  data[msk2]
test = test.reset_index(drop=True)

train.columns.values
#%%
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  

#%%
print "Cleaning and parsing the training set movie reviews...\n"
clean_train_reviews = []

num_reviews = len(train["text"])

for i in xrange( 0,  train.shape[0]):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )    
    print i;                                                                 
    clean_train_reviews.append( review_to_words(train["text"][i]))
    
#%%  Creating Features from a Bag of Words (Using scikit-learn)
print "Creating the bag of words...\n"
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab

#%%
# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag
    
#%% Random forest
print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit(train_data_features, train["label"] )


#%% Get test set
# Read the test data

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["text"])
clean_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["text"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


#%% Predict
result  = forest.predict(test_data_features)


# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["label"], "sentiment":result} )

#%%
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(test["label"], result)

#%%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


mymap = {'negative':0, 'positive':1};
y = test
y = y.replace({"label":mymap})
score =result
score[score == 'positive'] = 1
score[score == 'negative'] = 0
#%%
fpr, tpr, threshold = roc_curve(y["label"], score)

#%%
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.show()

#%%
# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["label"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "/Users/matthewxfz/Workspaces/gits/YelpReivewSentimentAnalysis/data/randomForest.csv", index=False, quoting=3 )


#%%
from sklearn.metrics import f1_score
s=  y["label"]
f1a = np.array(s, dtype = np.uint8);
f1b = np.array(score, dtype = np.uint8);

f1_score(f1a, f1b,  average='macro')