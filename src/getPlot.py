#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:59:47 2017

@author: matthewxfz
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:24:45 2017

@author: matthewxfz
"""

#%%
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

import os 
dirpath = os.path.dirname(os.path.realpath(__file__))

import pandas as pd
import numpy as np
import sys
sys.path.append(dir_path)
import tsne as t
import util
sys.setdefaultencoding("utf-8")
         
modelpath = dirpath + "yelpLtraining";

#%%
from gensim.models import word2vec, Phrases
from sklearn.manifold import TSNE
#%%s
model = word2vec.Word2Vec.load(modelpath);
#%% T-SNE visuallizing
print("start TSNE")
wv_embendded = TSNE(n_components=2).fit_transform(model.wv.syn0)

print("write TSNE result")
util.writeSenence(dirpath+"wv_embendded.csv",wv_embendded)

print("Wrote in %s", % (dirpath+"wv_embendded.csv"))