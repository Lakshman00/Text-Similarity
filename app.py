import pandas as pd
import streamlit as st
import textdistance
import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")

df=pd.read_csv("Precily_Text_Similarity.csv")

def text_processing(sentence):
    sentence = [token.lemma_.lower()
                for token in nlp(sentence)
                if token.is_alpha and not token.is_stop]
    return sentence

def jaccard_sim(sen1,sen2):
    sentence1 = text_processing(sen1)
    sentence2 = text_processing(sen2)
    return textdistance.jaccard.normalized_similarity(sentence1, sentence2)


st.title("Text Similarity Prediction")
text1=st.text_input("Enter text1")
text2=st.text_input("Enter text2")

if(st.button("Predict Similarity")):
    pred=jaccard_sim(text1,text2)
    st.title("Similarity :"+str(round(pred,3)))


