import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Set NLTK data path to a common location
nltk.data.path.append("/usr/share/nltk_data")

# Download necessary NLTK data
nltk.download('punkt', download_dir="/usr/share/nltk_data")
nltk.download('stopwords', download_dir="/usr/share/nltk_data")

ps = PorterStemmer()

# Stemming
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Spam Message Verifier System")

input_msg = st.text_area("Enter your message below:")

if st.button('Click here to check'):
    # 1. Preprocess
    transformed_msg = transform_text(input_msg)
    
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_msg])
    
    # 3. Predict
    result = model.predict(vector_input)[0]
    
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
