#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 19:13:41 2017

@author: matthewxfz
"""

#%%

import csv
import pandas as pd
import numpy as np
import sys

import os 
dirpath = os.path.dirname(os.path.realpath(__file__))

#dirpath = "/Users/matthewxfz/Workspaces/gits/course/YelpReivewSentimentAnalysis/src";
sys.path.append(dirpath)
import util

#dirpath = "/Users/matthewxfz/Workspaces/gits/course/YelpReivewSentimentAnalysis/data"
outputfile = dirpath + "/wv_embendded.csv"
modelpath = dirpath + "/yelpLtraining";
outpufpng = dirpath + "/wcloud.png"
outpufpng = dirpath + "/wcloud_cluster.png"

def plot_with_labels(low_dim_embs, labels, idx, filename=outpufpng,fonts=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        colors = idx[i];
        plt.scatter(x, y)
        plt.annotate(label,
                    fontproperties=fonts,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')

    plt.savefig(filename,dpi=800)
    
def plot_with_cluster(data, idx, filename=outpufpng,fonts=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    plt.scatter(data[:,0], data[:,1], c=idx)
    plt.savefig(filename,dpi=800)
    
#%%
from gensim.models import word2vec, Phrases
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

model = word2vec.Word2Vec.load(modelpath);
#%%
import re

wv_arr = []
file = open(outputfile, 'r') 
for line in file: 
     wv = re.findall("-?[0-9]*\.[0-9]*",line)
     wv_arr.append([float(wv[0]), float(wv[1])])
     
wv_arr = np.array(wv_arr, dtype = float)

#%% cluster the model
from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or ans
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = 20;
nr = 1000;
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters, max_iter = 5, n_jobs = 10,  )
idx = kmeans_clustering.fit_predict( word_vectors )
print("start printing!")
plot_with_labels(wv_arr[0:nr,:], model.wv.index2word[0:nr], idx[0:nr]/50.0);
plot_with_cluster(wv_arr[0:nr,:], idx[0:nr]/20.0)
print("end printing!")

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."