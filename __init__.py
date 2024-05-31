import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download the punkt tokenizer
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Stemming function
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

# Streamlit app title
st.title("Spam Message Verifier System")

# Input text area
input_msg = st.text_area("Enter your message below:")

# Button to trigger spam check
if st.button('Click here to check'):
    # 1. Preprocess
    transformed_msg = transform_text(input_msg)
    
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_msg])
    
    # 3. Predict
    result = model.predict(vector_input)[0]
    
    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
