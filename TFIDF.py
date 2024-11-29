import numpy as np
import pandas as pd

def compute_tf(documents):
    tf = []
    for doc in documents:
        words = doc.split()
        word_count = len(words)
        tf_dict = {}
        for word in words:
            tf_dict[word] = tf_dict.get(word, 0) + 1 / word_count
        tf.append(tf_dict)
    return tf

def compute_idf(documents):
    import math
    idf = {}
    total_docs = len(documents)
    for doc in documents:
        words = set(doc.split())
        for word in words:
            idf[word] = idf.get(word, 0) + 1
    for word, count in idf.items():
        idf[word] = math.log(total_docs / count) + 1
    return idf

def compute_tfidf(documents):
    tf = compute_tf(documents)
    idf = compute_idf(documents)
    
    tfidf = []
    for doc_tf in tf:
        tfidf_doc = {}
        for word, tf_value in doc_tf.items():
            tfidf_doc[word] = tf_value * idf[word]
        tfidf.append(tfidf_doc)
    
    return tfidf