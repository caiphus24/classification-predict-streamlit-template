"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import matplotlib.style as style
from nbformat import write
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
import plotly
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import warnings
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk import pos_tag
from PIL import Image
from nlppreprocess import NLP
from os import path
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string
import re
import nltk
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Libraries for data loading, data manipulation and data visulisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib as mpl


import seaborn as sns


# Overrall graph sizes
plt.rcParams['figure.dpi'] = 150


# NLP Libraries
nlp = NLP()


# Setting global constants to ensure notebook results are reproducible

PARAMETER_CONSTANT = 'Magic String'

# ignore warnings
warnings.filterwarnings('ignore')

# Style
sns.set(font_scale=1.5)
style.use('seaborn-pastel')
style.use('seaborn-poster')


# ML Libraries
nlp = NLP()

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
raw['Analysis'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1]
                           for x in raw['sentiment']]
# Function to extract hashtags from tweets
def hashtag_extract(tweet):

	hashtags = []

	for i in tweet:
		ht = re.findall(r"#(\w+)", i)
		hashtags.append(ht)

	hashtags = sum(hashtags, [])
	frequency = nltk.FreqDist(hashtags)

	hashtag_df = pd.DataFrame(
		{'hashtag': list(frequency.keys()), 'count': list(frequency.values())})
	hashtag_df = hashtag_df.nlargest(15, columns="count")

	return hashtag_df

# Extracting the hashtags from tweets in each class
positive= hashtag_extract(raw['message'][raw['Analysis'] == 'Positve'])
negative = hashtag_extract(raw['message'][raw['Analysis'] == 'Negative'])
neutral = hashtag_extract(raw['message'][raw['Analysis'] == 'Neutral'])
news = hashtag_extract(raw['message'][raw['Analysis'] == 'News'])


def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "Analysis"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            # will write the df to the page
            st.write(raw[['sentiment', 'message']])
    # Building visuals for the Analysis
    if selection == "Analysis":
        st.info("Explotary Analysis with visuals")
    # Display target distribution
        st.write(
            'To have a bettter understanding of the data, let us have a look at the distrition.')
        fig, axy = plt.subplots(ncols=2, nrows=1, figsize=(20, 10), dpi=100)
        sns.countplot(raw['Analysis'], ax=axy[0])
        dist = raw['Analysis'].value_counts()/raw.shape[0]
        labels = (raw['Analysis'].value_counts() /
                  raw.shape[0]).index  # count the column sentiment
        # display the color of the visual
        axy[1].pie(raw['Analysis'].value_counts(), labels=labels, autopct='%1.0f%%',
                   shadow=True, startangle=90, explode=(0.1, 0.1, 0.1, 0.1))
        fig.suptitle('Tweet distribution', fontsize=30)
        colors = ['lightgreen', 'blue', 'purple', 'orange']
        st.pyplot(fig)  # show the visual
        st.markdown('We can now use the above graph to determine if the our data is a balanced or imbalanced classification. We first need to understand what the two classification mean.')
        st.markdown('**Imbalanced Classication**: A classication predictive modeling problem where thedistribution of examples across the classes is not equal.')
        st.subheader(
            'The following visuals help us analyse the most active users based on the sentiment(positive, neutral, news or negative)')
        
        # extract users from the message column
        raw['users'] = [''.join(re.findall(r'@\w{,}', line))
                        if '@' in line else np.nan for line in raw.message]
        
        # count the tweet message per users and group by users
        user_counts = raw[['message', 'users']].groupby(
            'users', as_index=False).count().sort_values(by='message', ascending=False)

        # Popular Tags For Positive Sentiment
        st.markdown('Popular Top 20 Tags For Positive Sentiment')
        fig = plt.figure(figsize=(15, 10))
        sns.countplot(x="users", data=raw[raw['Analysis'] == 'Positive'],
                      order=raw[raw['Analysis'] == 'Positive'].users.value_counts().iloc[:20].index)
        plt.title('Top 20 Positive Tags', fontsize=30)
        plt.xticks(rotation=85)
        st.pyplot(fig)

        # Popular Tags for News News
        st.markdown('Popular Top 20 Tags For News Sentiment')
        fig = plt.figure(figsize=(15, 12))
        sns.countplot(x="users", data=raw[raw['Analysis'] == 'News'],
                      order=raw[raw['Analysis'] == 'News'].users.value_counts().iloc[:20].index)
        plt.title('Top 20 News Tags')
        plt.xticks(rotation=85)
        st.pyplot(fig)

        # Popular Tags for Negative News
        st.markdown('Popular Top 20 Tags For Neutral Sentiment')
        fig = plt.figure(figsize=(15, 12))
        sns.countplot(x="users", data=raw[raw['Analysis'] == 'Neutral'],
                      order=raw[raw['Analysis'] == 'Neutral'].users.value_counts().iloc[:20].index)
        plt.title('Top 20 News Tags')
        plt.xticks(rotation=85)
        st.pyplot(fig)

        # Popular Tags for Negative News
        st.markdown('Popular Top 20 Tags For Negative Sentiment')
        fig = plt.figure(figsize=(15, 12))
        sns.countplot(x="users", data=raw[raw['Analysis'] == 'Negative'],
                      order=raw[raw['Analysis'] == 'Negative'].users.value_counts().iloc[:20].index)
        plt.title('Top 20 News Tags')
        plt.xticks(rotation=85)
        st.pyplot(fig)
        st.markdown('The above graphs gives us an idea on who was mentioned or who engaged the most in the climate change, We can therefore assume that the user or users who got mentioned most in all sentiments they are the  leading leaders of the Climate chaneg senate, It would not make sense if it happens that unknown users could have so much engagements if it happens that they hold no position in the senate of the Word Climat Change.')

		# Hashtags per sentiment
        st.subheader(
            'The following visuals provides the hashtags per tweets sentiment')

        

        
    # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(
                open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
