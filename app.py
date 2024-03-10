# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:30:07 2024

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

loaded_model = load_model('C:/Users/yash/Documents/NLPclass/project/emotions_analysis/model.h5')


tokenizer101 = pickle.load(open('C:/Users/yash/Documents/NLPclass/project/emotions_analysis/vectorizer.sav','rb'))
model = pickle.load(open('C:/Users/yash/Documents/NLPclass/project/emotions_analysis/model.sav','rb'))

st.title("enter the text")

input_text = st.text_area("Enter the message")
print('49',input_text)
maxlen=79


if st.button('Predict'):
    transformed_text = clean_token(input_text)
    print("Transformed Tweet:", transformed_text)  # Add this line to check the output
    sequence1 = tokenizer.texts_to_sequences([transformed_text])
    print(sequence1)

    padded_sequence =sequence.pad_sequences(sequence1, maxlen=maxlen)
    print(padded_sequence)

    # Predict
    result = model.predict(padded_sequence)[0]
    print(result)
    emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    predicted_emotion = emotion_labels[int(result.argmax())]
    print(predicted_emotion)


