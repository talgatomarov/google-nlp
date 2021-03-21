#!/usr/bin/env python
# coding: utf-8

# # TFIDF + SVD

# In[1]:


from string import punctuation
import time
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from annoy import AnnoyIndex

nltk.download('stopwords')
nltk.download('wordnet')


# ## Loading data

# In[2]:


df = pd.read_csv('../data/articles.csv')
content = df['content']


# ## Custom tokenization

# In[3]:


lemmatizer = WordNetLemmatizer()

# I did not use this function due to time constraints
# Tokenizer built-in in TfidfVetorizer was much faster
def tokenize(text):
    tokens = np.array(word_tokenize(text.lower()))
    tokens = tokens[~np.isin(tokens, stopwords.words("english")) & ~np.isin(tokens, punctuation)]
    
    if tokens.size == 0:
        return np.array([])
    
    tokens = tokens[np.char.str_len(tokens) > 1]
    
    if tokens.size == 0:
        return np.array([])
    
    lemmatize = np.vectorize(lemmatizer.lemmatize)
    lemmas = lemmatize(tokens)        
    
    return lemmas 


# ## TFIDF

# In[4]:


vectorizer = TfidfVectorizer(max_features=30000, strip_accents='ascii', lowercase=True, stop_words='english')
tfs = vectorizer.fit_transform(content)


# ## SVD

# In[5]:


svd = TruncatedSVD(n_components = 512)
tfs_truncated = svd.fit_transform(tfs)


# ## ANNOY indexing

# In[7]:


def save_index(matrix, file_name, dim=512, n_trees=128):
    annoy_index = AnnoyIndex(dim, 'angular')
    
    if isinstance(matrix, csr_matrix):
        matrix = matrix.toarray()

    for i, v in enumerate(matrix):
        annoy_index.add_item(i, v)

    annoy_index.build(n_trees)
    annoy_index.save(file_name)


# ## Saving artifacts

# In[6]:


def save_artifacts():
    pickle.dump(vectorizer, open("../artifacts/vectorizer.p", "wb" ))
    pickle.dump(tfs, open("../artifacts/tfs.p", "wb" ))
    pickle.dump(svd, open("../artifacts/svd.p", "wb" ))
    pickle.dump(tfs_truncated, open("../artifacts/tfs_truncated.p", "wb" ))


# In[8]:


save_artifacts()
save_index(tfs_truncated, '../artifacts/index.ann')

