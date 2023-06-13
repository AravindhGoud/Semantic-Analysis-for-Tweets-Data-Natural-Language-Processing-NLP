import streamlit as st
import pickle
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.title('Twitter Semantic Analysis Prediction Model')

# Download NLTK stopwords
nltk.download('stopwords')

# Load the trained model and vectorizer
with open(r"C:\Users\aravi\OneDrive\Desktop\NLP Project\Deployment\trained_model_classifier.pkl", "rb") as f:
    loaded_model = pickle.load(f, encoding="utf-8")

# Set up stopwords and stemmer
stop = stopwords.words('english')
stemmer = PorterStemmer()

# Clean the text 
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+", " ", text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Remove stopwords
def remove_stopwords(sentence):
    return " ".join(x for x in sentence.split() if x not in stop)

# Apply stemming
def apply_stemming(text):
    stemmed_text = [stemmer.stem(word) for word in text.split()]
    return ' '.join(stemmed_text)

# Define the Streamlit app
def main():
    # Set the app title
    
    
    # Getting input from user
    tweet = st.text_input('Enter Your Twitter Comment')

    if st.button('Predict'):
        cleaned_input = clean_text(tweet)
        removed_stopwords = remove_stopwords(cleaned_input)
        stemmed_words = apply_stemming(removed_stopwords)
        tfidf_vectorizer = loaded_model['tfidf']
        classifier = loaded_model['classifier']
        tfidf_matrix = tfidf_vectorizer.transform([stemmed_words])
        predict = classifier.predict(tfidf_matrix)
        st.success(predict)

# Run the app
if __name__ == '__main__':
    main()
