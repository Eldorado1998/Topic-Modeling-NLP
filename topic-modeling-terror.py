#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:21:13 2021

@author: Yingzhi
"""
import pandas as pd
from NYT_functions import *

df=pd.read_csv("NYT_text_1851_1980.csv")

df_1851_1900 = df[ (1851 <= df.year) & (df.year <= 1900) ]
df_1901_1930 = df[ (1901 <= df.year) & (df.year <= 1930) ]
df_1931_1950 = df[ (1931 <= df.year) & (df.year <= 1950) ]
df_1951_1960 = df[ (1951 <= df.year) & (df.year <= 1960) ]
df_1961_1980 = df[ (1961 <= df.year) & (df.year <= 1980) ]



import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import numpy as np
from kneed import KneeLocator

def fetch_bi_grams(var):
    sentence_stream = np.array(var)
    bigram = Phrases(sentence_stream, min_count=5, threshold=10, delimiter=",")
    trigram = Phrases(bigram[sentence_stream], min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)
    trigram_phraser = Phraser(trigram)
    bi_grams = list()
    tri_grams = list()
    for sent in sentence_stream:
        bi_grams.append(bigram_phraser[sent])
        tri_grams.append(trigram_phraser[sent])
    return bi_grams, tri_grams

        
def coherence_score(data):
    data["body_clean"] = data.text.apply(
    clean_text).apply(rem_sw).apply(check_additional_sw).str.split()
    the_data=data["body_clean"]
    dictionary = Dictionary(the_data)
    id2word = corpora.Dictionary(the_data)
    
    corpus = [id2word.doc2bow(text) for text in the_data]
    c_scores = list()
    for word in range(1, 10):
        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus, num_topics=word, id2word=id2word, iterations=10, passes=5,
            random_state=123)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=the_data,
                                              dictionary=dictionary,
                                              coherence='c_v')
        c_scores.append(coherence_model_lda.get_coherence())
    
    x = range(1, 10)
    kn = KneeLocator(x, c_scores,
                     curve='concave', direction='increasing')
    opt_topics = kn.knee
    print ("Optimal topics is", kn)
    plt.plot(x, c_scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


coherence_score(df_1851_1869)



def topic_modeling(data, n):
    data["body_clean"] = data.text.apply(
        clean_text).apply(rem_sw).apply(check_additional_sw).str.split()
    the_data=data["body_clean"]
    dictionary = Dictionary(the_data)
    id2word = corpora.Dictionary(the_data)
    
    corpus = [id2word.doc2bow(text) for text in the_data]
    n_topics=n
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=n_topics, id2word=id2word, iterations=50, passes=15,
        random_state=123)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=5)
    for topic in topics:
        print(topic)

print(topic_modeling(df_1851_1900,3))
print(topic_modeling(df_1901_1930,3))
print(topic_modeling(df_1931_1950,3))
print(topic_modeling(df_1951_1960,3))
print(topic_modeling(df_1961_1980,3))

def topic_distribution(data):
    data["body_clean"] = data.text.apply(
        clean_text).apply(rem_sw).apply(check_additional_sw).str.split()
    the_data=data["body_clean"]
    dictionary = Dictionary(the_data)
    id2word = corpora.Dictionary(the_data)
    corpus = [id2word.doc2bow(text) for text in the_data]
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=3, id2word=id2word, iterations=50, passes=15,
        random_state=123,minimum_probability=0.0)
    output=list(ldamodel[corpus])[0]
    return(output)

topic_distribution(df_1961_1980)