import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download('stopwords')
ps=PorterStemmer()

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf=pickle.load(open("vectorizer.pkl","rb"))
model=pickle.load(open("model.pkl","rb"))
st.title("Email/SMS Spam Classifier")
input_sms=st.text_area("Enter The Message")
btn=st.button("Predict The SMS")
if btn:
    # preprocess
    tranform_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([tranform_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display The result
    if result == 0:
        st.header("Not Spam")
    if result == 1:
        st.header("Spam")




