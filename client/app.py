import pickle
import streamlit as st
import pandas as pd
from annoy import AnnoyIndex
import time
import numpy as np

st.title("Local Search Engine")

# Loading dataset
df = pd.read_csv('../data/articles.csv')
content = df['content']

# Loading artifacts
vectorizer = pickle.load(open("../artifacts/vectorizer.p", "rb"))
svd = pickle.load(open("../artifacts/svd.p", "rb"))
tfs_truncated = pickle.load(open("../artifacts/tfs_truncated.p", "rb"))

annoy_idx = AnnoyIndex(512, 'angular')
annoy_idx.load('../artifacts/index.ann')


def get_idxs(query):
    t1 = time.time()
    query_tfs = vectorizer.transform(query)
    query_tfs_truncated = svd.transform(query_tfs)

    idxs = annoy_idx.get_nns_by_vector(query_tfs_truncated[0], 5)
    t2 = time.time()

    return idxs, t2-t1


# Query
raw_query = st.text_input("Query")
query = [raw_query]

if raw_query:
    idxs, delta = get_idxs(query)
    st.write(f'Query finished in {delta} seconds')

    for idx in idxs:
        st.header(df.title[idx])
        st.write(f'Article ID: {df.id[idx]}')
        st.write(df.content[idx][:500])
    # else:
    #     st.text("No results were found :(")
