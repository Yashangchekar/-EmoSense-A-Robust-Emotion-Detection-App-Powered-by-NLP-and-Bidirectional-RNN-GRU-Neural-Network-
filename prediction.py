# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:28:50 2024

@author: yash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.tokenize import   word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense,Dropout,Embedding,LSTM,SimpleRNN,GRU
from tensorflow.keras.models import Sequential
from keras.layers import Bidirectional, GRU, Embedding, Dropout, BatchNormalization, Dense
from keras.models import Sequential
from tensorflow.keras.utils import pad_sequences
import pickle
import streamlit as st

tokenizer = Tokenizer(num_words=50000) 
def clean_token(text):
  stop=stopwords.words('english')
  punc=list(punctuation)
  bad_tokens=stop+punc
  lemma=WordNetLemmatizer()
  tokens=word_tokenize(text)
  word_tokens=[t for t in tokens if t.isalpha()]
  clean_token=[ lemma.lemmatize(t.lower())for t in word_tokens if t not in bad_tokens]
  return " ".join(clean_token)


from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model

loaded_model = load_model('C:/Users/yash/Documents/NLPclass/project/emotions_analysis/model.h5')
model = pickle.load(open('C:/Users/yash/Documents/NLPclass/project/emotions_analysis/model.sav','rb'))

input_text11="i gave up my internship with the dmrg and am feeling distraught"
transformed_text11 = clean_token(input_text11)
print("Transformed Tweet:", transformed_text11)  # Add this line to check the output
maxlen=79

sequence11 = tokenizer.texts_to_sequences([transformed_text11])
print(sequence11)

padded_sequence11 =sequence.pad_sequences(sequence11, maxlen=maxlen)
print(padded_sequence11)

    # Predict
result1 = loaded_model.predict(padded_sequence11)[0]
print(result1)

emotion_labels1 = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
predicted_emotion1 = emotion_labels1[int(result1.argmax())]
print(predicted_emotion1)