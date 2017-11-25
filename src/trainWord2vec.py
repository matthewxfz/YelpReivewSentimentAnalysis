#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:57:55 2017

@author: matthewxfz
"""

#%%

import pandas as pd
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

dirpath = "/Users/matthewxfz/Workspaces/gits/course/YelpReivewSentimentAnalysis/data/"
outputfile = dirpath + "sentence.csv"
inputfile = dirpath + "yelp810.csv"
#%%
data = pd.read_csv(inputfile, header=0, 
 delimiter=",",index_col = False, quoting=3,  encoding = 'utf8')

ran = np.random.rand(len(data));
msk1 = ran < 0.8
msk2 = ran > 0.8

train = data[msk1]
train = train.reset_index(drop=True)
test =  data[msk2]
test = test.reset_index(drop=True)

#%%
import csv
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


snowball_stemmer = SnowballStemmer("english")
st = LancasterStemmer()
#%%
def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
        
    # 4. Lemmatize
    lemmatized_words = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in meaningful_words]
    # 5. Return a list of words
    return(" ".join( lemmatized_words ))
    
def writeSenence(path, sentence):
    with open(path, 'w+') as csvfile:
          spamwriter = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
          for count, text in enumerate(sentence):
              spamwriter.writerow([str(count), text]);
              
    print("write to file %s" % (path));
  
    
def readSentence(path):
    sentence = [];
    with open(path) as csvfile:
     spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
     for row in spamreader:
         sentence.append(row[1]);

    print("Read data from %s" % (path));
    return sentence;
             
#%%
# Download the punkt tokenizer for sentence splitting
import nltk.data
#nltk.download() 

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#%%
# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist(raw_sentence, \
              remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences
#%%
sentences = []  # Initialize an empty list of sentences
import os
if(os.path.exists(outputfile)):
    print("Reading data from file...")
    sentences = readSentence(outputfile)
else:
    print("Parsing sentences from training set")
    for review in train["text"]:
        sentences += review_to_sentences(review, tokenizer)
    
    print("Parsing sentences from training set")
    for review in test["text"]:
        sentences += review_to_sentences(review, tokenizer)
        
    print("Writting data into file...")
    writeSenence(outputfile, sentences);
    
#%%
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

#%%
# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count                        
num_workers = 50       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 0.995   # Downsample setting for frequent words

#%%
# Initialize and train the model (this will take some time)
from gensim.models import word2vec, Phrases
#%%
print "Training model..."
bigram_transformer = Phrases(sentences)
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "yelpLtraining"
model.save(model_name)
