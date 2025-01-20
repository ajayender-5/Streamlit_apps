import streamlit as st
import re
import pandas as pd
import numpy as np
import emoji
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,SnowballStemmer, LancasterStemmer,WordNetLemmatizer
from textblob import TextBlob
from nltk.tokenize import word_tokenize,sent_tokenize


st.set_page_config(page_title="Text Pre-processing", page_icon=":memo:")
st.title("Text Pre-processing")
uploaded_files = st.file_uploader("Choose a CSV file with a single text column", accept_multiple_files=False)

if uploaded_files is not None:
    uploaded_file = uploaded_files.read().decode('utf-8')  # Read CSV file as string
    st.write("Original Text:",uploaded_file)  # Display the contents of the uploaded file
    
    cleaning_steps = [
        "Convert to Lowercase",
        "Remove HTML tags(<>)",
        "Remove Urls",
        "convert emojies to text",
        "Remove Special Characters and Punctuation",
        "Tokenization",
        "Remove Stop Words",
        "Stemming",
        "Lemmatization",
        "Spelling Corrections"
    ]

    selected_step = st.sidebar.radio("Choose the Steps to clean", cleaning_steps)

    if st.sidebar.button("Submit"):
        if selected_step == "Convert to Lowercase":
            cleaned_text = uploaded_file.lower()
            st.write("Cleaned Text:", cleaned_text)
        elif selected_step == "Remove HTML tags(<>)":
            cleaned_text = re.sub(r'<.*?>', '', uploaded_file)  # Remove HTML tags using regex
            st.write("Cleaned Text:", cleaned_text)
        elif selected_step == "Remove Urls":
            cleaned_text = re.sub(r"http[s]?://.+?\S+", "", uploaded_file)
            st.write("Cleaned Text:", cleaned_text)
        elif selected_step == "convert emojies to text":
            cleaned_text = emoji.demojize(uploaded_file)
            st.write("Cleaned Text:", cleaned_text)
        elif selected_step == "Remove Special Characters and Punctuation":
            cleaned_text = re.sub("[^a-zA-Z.]", " ", uploaded_file)
            st.write("Cleaned Text:", cleaned_text)
        elif selected_step == "Tokenization":
            cleaned_text = word_tokenize(uploaded_file)
            st.write("Cleaned Text:", cleaned_text)
        elif selected_step == "Remove Stop Words":
            stop_words = set(stopwords.words("english"))
            cleaned_text = [word for word in word_tokenize(uploaded_file) if word.lower() not in stop_words]
            st.write("Cleaned Text:", cleaned_text)
        elif selected_step == "Stemming":
            stemmer = SnowballStemmer(language="english")
            cleaned_text = " ".join([stemmer.stem(word) for word in word_tokenize(uploaded_file)])
            st.write("Cleaned Text:", cleaned_text)
        elif selected_step == "Lemmatization":
            lemmatizer = WordNetLemmatizer()
            cleaned_text = " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(uploaded_file)])
            st.write("Cleaned Text:", cleaned_text)
        elif selected_step == "Spelling Corrections":
            cleaned_text = TextBlob(uploaded_file).correct().string
            st.write("Cleaned Text:", cleaned_text)
        
    def eda(data):
        results = []
        
        lower = any(row.islower() for row in data)
        if not lower:
            results.append("All rows are in uppercase or mixed case.")
        
        html = sum(1 for row in data if re.search("<.*?>", row))
        if html > 0:
            results.append("Your data contains HTML tags.")
        
        urls = sum(1 for row in data if re.search("http[s]?://.+?\S+", row))
        if urls > 0:
            results.append("Your data contains URLs.")
        
        tags = sum(1 for row in data if re.search("#\S+", row))
        if tags > 0:
            results.append("Your data contains hashtags.")
        
        mentions = sum(1 for row in data if re.search("@\S+", row))
        if mentions > 0:
            results.append("Your data contains mentions.")
        
        un_wanted = sum(1 for row in data if re.search("[]\.\*'\-#$%^&)(0-9]!@", row))
        if un_wanted > 0:
            results.append("Your data contains unwanted characters.")
        
        emojiss = sum(1 for row in data if emoji.emoji_count(row))
        if emojiss > 0:
            results.append("Your data contains emojis.")
        
        return results

    if st.sidebar.button("All"):
        def basic_pp(x, emoj="T", spc="F"):
            x = x.lower()  # converting into lowercase
            x = re.sub("<.*?>", " ", x)  # removing html tags
            x = re.sub("http[s]?://.+?\S+", " ", x)  # removing urls
            x = re.sub("#\S+", " ", x)  # removing hashtags
            x = re.sub("@\S+", " ", x)  # removing mentions
            if emoj == "T":
                x = emoji.demojize(x)  # converting emoji to text
            x = re.sub("[]\:.\*'\-#$%^&)(0-9]", " ", x)  # removing unwanted characters
            x = re.sub("[^a-zA-Z.]", " ", x)  # removing non-alphabetic characters
            if spc == "T":
                x = TextBlob(x).correct().string  # spelling check
            return x

        cleaned_text = basic_pp(uploaded_file, "T", "T")
        st.write("Cleaned Text (All Steps):", cleaned_text)

    if st.button("Checking"):
        check_results = eda(uploaded_file.split("\n"))  # Assuming each line in uploaded_file is a row of text
        for result in check_results:
            st.write(result)