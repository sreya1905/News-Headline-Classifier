#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[9]:


bbc_text = pd.read_csv(r"C:\Users\Sreya\Desktop\duk notes\semester 2\nlpir\bbc-text.txt")
bbc_text=bbc_text.rename(columns = {'text': 'News_Headline'}, inplace = False)#Renaming column


# In[10]:


bbc_text.category = bbc_text.category.map({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})


# In[11]:


X = bbc_text.News_Headline
y = bbc_text.category
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)


# In[12]:


vector = CountVectorizer(stop_words = 'english',lowercase=False)
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_transformed.toarray()
X_test_transformed = vector.transform(X_test)


# In[13]:


naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)


# In[14]:


saved_model = pickle.dumps(naivebayes)
s = pickle.loads(saved_model)


# In[15]:


st.header('News Headline Classifier')
input = st.text_area("Please enter the news headline", value="")
if st.button("Predict"):
    v=vector.transform([input]).toarray()
    st.write(str(list(s.predict(v))[0]).replace('0','TECH').replace('1','BUSINESS').replace('2','SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))


# In[ ]:




