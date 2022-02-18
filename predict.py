from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import joblib
import pickle
import string
import scipy
from sentence_transformers import SentenceTransformer
#from textblob import TextBlob

import numpy as np
import pandas as pd

def pre_processing(question):
    def lemmatize_with_pos_tag(sentence):
        tokenized_sentence = TextBlob(sentence)
        tag_dict = {"J": 'a', "N": 'n', "V": 'v', "R": 'r'}
        words_and_tags = [(word, tag_dict.get(pos[0], 'n')) for word, pos in tokenized_sentence.tags]
        lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
        return " ".join(lemmatized_list)
    question = question.lower()
    question = question.replace('[','')
    question = question.replace(']','')
    question.translate(str.maketrans(" ", " ", string.punctuation))
    #question = lemmatize_with_pos_tag(question)
    return question
  
def bert_disease_predict(query):
  
  model_bert = SentenceTransformer('model')
  with open('model/sentence_encoder_symp', 'rb') as f:
    sentence_embeddings = pickle.load(f)
  with open('model/symp.pkl', 'rb') as f:
    symps = pickle.load(f)
  queries = pre_processing(query)
  query_embeddings = model_bert.encode([queries])

  for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
  return results
    
