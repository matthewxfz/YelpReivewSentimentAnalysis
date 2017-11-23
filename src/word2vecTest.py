#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:27:57 2017

@author: matthewxfz
"""

#%%

import pandas as pd
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

#%%
# Read data from files 
data = pd.read_csv( "/Users/matthewxfz/Workspaces/gits/course/YelpReivewSentimentAnalysis/data/bus100use5_review.csv", header=1, 
 delimiter="\t", quoting=3,  encoding = 'utf8')

data = np.array(data)

data = data[:,0:4]

data = pd.DataFrame(data)

data.columns = ['category', 'star', 'usefull', 'text']

ran = np.random.rand(data.shape[0]);
msk1 = ran < 1
msk2 = ran > 1

train = data[msk1]
#train = train.reset_index(drop=True)
test =  data[msk2]
#test = test.reset_index(drop=True)

#%%

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

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
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)
    
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

print "Parsing sentences from training set"
for review in train["text"]:
    sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in test["text"]:
    sentences += review_to_sentences(review, tokenizer)
    
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)


# Set values for various parameters
num_features = 200    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count                        
num_workers = 50       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 0.995   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "1WSamples200Features_10minW_10numW_10Context_0.995Dwonsample"
model.save(model_name)

#%%
model.doesnt_match("I disdain this nightclub.".split())

#%%
#model2 = word2vec.Word2Vec.load("300features_40minwords_10context")

#%%
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    print "whose fault"
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000 == 0:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1
       
    return reviewFeatureVecs
#%%
# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.
clean_train_reviews = []
for review in train["text"]:
    clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features )

#%%
print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["text"]:
    clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

##%%
## Fit a random forest to the training data, using 100 trees
#from sklearn.ensemble import RandomForestClassifier
#forest = RandomForestClassifier( n_estimators = 100 )
#
#print "Fitting a random forest to labeled training data..."
#forest = forest.fit( trainDataVecs, train["label"] )
#
## Test & extract results 
#result = forest.predict( testDataVecs )
#
##%%
#from sklearn.metrics import confusion_matrix
#confusion_matrix = confusion_matrix(test["label"], result)
#
##%%
#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve
#
#
#mymap = {'negative':0, 'positive':1};
#y = test
#y = y.replace({"label":mymap})
#score =result
#score[score == 'positive'] = 1
#score[score == 'negative'] = 0
##%%
#fpr, tpr, threshold = roc_curve(y["label"], score,pos_label=1)
#
##%%
#plt.plot(fpr, tpr)
#plt.plot([0, 1], [0, 1],'r--')
#plt.show()

#%%
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.stats import sem

import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = 10

def getAllAli(rang2e):
    out = np.zeros((2,rang2e));
    for i in range(2,rang2e):
        kmeans_clustering = KMeans( n_clusters = i )
        idx = kmeans_clustering.fit_predict( word_vectors )
        silhouette_avg = metrics.silhouette_score(word_vectors, idx)
        out[0][i] = silhouette_avg
        
    return out;
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")

#%%
# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                       
wv_count0  = [model.wv.vocab[m].count for m in model.wv.index2word]      
wv_count0 = np.array(wv_count0, dtype = np.float);
wv_count = np.log(1+wv_count0/wv_count0.max());
far = wv_count.std()+wv_count.mean();
selected = wv_count<=far

#wv_count
#wv_count = 0.5 + 0.5*(wv_count/wv_count.max());


pdf2 = pd.DataFrame(data={'word':model.wv.index2word, "class":idx, "freq":wv_count});
pdf = pdf2[selected]
pdfo = pd.DataFrame(data={'word':[], "class":[], "freq":[]});

words = []

#%% most similar words
for i in range(0,num_clusters):
    pdfc = pdf[pdf['class'] == i];
    #pdfc = pdfc.sort_index
    pdfo = pdfo.append(pdfc[0:5])
    words.append(str('-'.join(pdfc[0:5]['word'])))

#%%
word_centroid_map = dict(zip( model.wv.index2word, idx))

silhouette_avg = metrics.silhouette_score(word_vectors, idx);silhouette_avg 

#%%
slis = getAllAli(40)
#%%
import matplotlib.pyplot as plt
plt.plot(slis[0][2:40])
plt.ylabel('silhouette score')
plt.show()

#%%
from sklearn.cluster import DBSCAN

n_c_dbscan = [];
sil_dbscan = [];


def getSil2():
    k=1
    while k<10:
        db = DBSCAN(eps=k/10., min_samples=10).fit(word_vectors)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_c_dbscan.append(n_clusters_)
        silhouette_avg = metrics.silhouette_score(word_vectors, labels+1)
        sil_dbscan.append(silhouette_avg)
        k+=1
        
getSil2()
#%%
import matplotlib.pyplot as plt
plt.plot(sil_dbscan)
plt.ylabel('silhouette score')
plt.show()

#%%
# Compute DBSCAN
db = DBSCAN(eps=0.3,  min_samples=10).fit(word_vectors)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
labels = labels+1
silhouette_avg = metrics.silhouette_score(word_vectors, labels)

#%%
n_clusters_
#%%
silhouette_avg
#%%
pdf3 = pd.DataFrame(data={'word':model.wv.index2word, "class":labels+1, "freq":wv_count});

words = []
for i in range(0,n_clusters_):
    pdfc = pdf3[pdf3['class'] == i];
    #pdfc = pdfc.sort_index
    pdfo = pdfo.append(pdfc[0:5])
    words.append(str('-'.join(pdfc[0:5]['word'])))
    
#%%
Y = pd.DataFrame(kmeans_clustering.fit_predict(word_vectors), columns=['cluster ID'])
Z = pd.DataFrame(kmeans_clustering.cluster_centers_[Y['cluster ID']], 
                 columns=['centroid_x', 'centroid_y'])
#result = pd.concat([Y, Z], axis=1)  

#print(result.head())
#%%
# For the first 10 clusters
wordss = []
for cluster in xrange(0,10):
    #
    # Print the cluster number  
    print("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    wordss.append(words)
    
#%%
def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

#%%
# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train["text"].size, num_clusters), \
    dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros(( test["text"].size, num_clusters), \
    dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1
    
    
#%%
# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train["text"].size, num_clusters), \
    dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros(( test["text"].size, num_clusters), \
    dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1
    
#%%
from sklearn.ensemble import RandomForestClassifier
# Fit a random forest and extract predictions 
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids,train["label"])
result = forest.predict(test_centroids)

# Write the test results 
output = pd.DataFrame(data={"label":result})
output.to_csv( "/Users/matthewxfz/Workspaces/gits/course/YelpReivewSentimentAnalysis/src/BagOfCentroids.csv", index=False, quoting=3 )

#%%



