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
sys.path.append("/Users/matthewxfz/Workspaces/gits/course/YelpReivewSentimentAnalysis/src")
import util
import os 
dirpath = os.path.dirname(os.path.realpath(__file__))

#dirpath = "/Users/matthewxfz/Workspaces/gits/course/YelpReivewSentimentAnalysis/data"
outputfile = dirpath + "/wv_emnbendded.csv"
modelpath = dirpath + "/yelpLtraining";
outpufpng = dirpath + "/wcloud.png"

def plot_with_labels(low_dim_embs, labels, filename=outpufpng,fonts=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(100, 100))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                    fontproperties=fonts,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
        print(i);
    plt.savefig(filename,dpi=800)
    
#%%
from gensim.models import word2vec, Phrases
import matplotlib.pyplot as plt

model = word2vec.Word2Vec.load(modelpath);
#%%
import re

wv_arr = []
file = open(outputfile, 'r') 
for line in file: 
     wv = re.findall("-?[0-9]*\.[0-9]*",line)
     wv_arr.append([float(wv[0]), float(wv[1])])
     
wv_arr = np.array(wv_arr, dtype = float)
#%%

print("start printing!")
plot_with_labels(wv_arr , model.wv.index2word);
print("end printing!")
