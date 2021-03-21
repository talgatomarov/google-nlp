#!/usr/bin/env python
# coding: utf-8

# # Comparison

# In[1]:


import timeit
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from functools import partial

from annoy import AnnoyIndex


# ## Loading artifacts

# In[2]:


vectorizer = pickle.load(open("../artifacts/vectorizer.p", "rb"))
svd = pickle.load(open("../artifacts/svd.p", "rb"))
tfs = pickle.load(open("../artifacts/tfs.p", "rb"))
tfs_truncated = pickle.load(open("../artifacts/tfs_truncated.p", "rb"))

annoy_idx = AnnoyIndex(512, 'angular')
annoy_idx.load('../artifacts/index.ann')


# In[3]:


print(f'tfidf matrix shape {tfs.shape}')
print(f'truncated tfidf matrix shape {tfs_truncated.shape}')


# In[4]:


def tfidf(query):
    query_tfs = vectorizer.transform(query)
    idxs = cosine_similarity(tfs, query_tfs).flatten().argsort()[-5:][::-1]

    return idxs


# In[5]:


def tfidf_truncated(query):
    query_tfs = vectorizer.transform(query)
    query_tfs_truncated = svd.transform(query_tfs)
    
    idxs = cosine_similarity(tfs_truncated, query_tfs_truncated).flatten().argsort()[-5:][::-1]
    
    return idxs


# In[6]:


def tfidf_truncated_annoy(query):
    query_tfs = vectorizer.transform(query)
    query_tfs_truncated = svd.transform(query_tfs)
    
    idxs = annoy_idx.get_nns_by_vector(query_tfs_truncated[0], 5)


# ## Speed comparison

# In[7]:


query = ['global warming']


# In[8]:


print(f'tfifd 100 runs: {timeit.timeit(partial(tfidf, query), number=100)} s')


# In[9]:


print(f'tfifd svd 100 runs: {timeit.timeit(partial(tfidf_truncated, query), number=100)} s')


# In[10]:


print(f'tfifd svd 100 runs: {timeit.timeit(partial(tfidf_truncated_annoy, query), number=100)} s')

