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
import streamlit as st
import joblib, os

# Data dependencies
import pandas as pd
from PIL import Image
from os import path

import numpy as np
import pandas as pd

# Visualization  dependencies
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import matplotlib as mpl
import matplotlib.style as style
from wordcloud import WordCloud
# Overrall graph sizes
plt.rcParams['figure.dpi'] = 150


# Preprocessing and cleaning dependencies
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
import nltk
import re
import string
from nltk import pos_tag
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from nlppreprocess import NLP
nlp = NLP()

# Setting global constants to ensure notebook results are reproducible
PARAMETER_CONSTANT = 'Magic String'
# ignore warnings
warnings.filterwarnings('ignore')

# Style
sns.set(font_scale=1.5)
style.use('seaborn-pastel')
style.use('seaborn-poster')

# Model Extraction  dependencies
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline



# Igonring warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Vectorizer
news_vectorizer = open("resources/tfidfvectu.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app

# Creating readable analysis
raw['Analysis'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1]
                   for x in raw['sentiment']]

# Creating a copy of the original data for visuals without affecting the original data
new_df = raw.copy()

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
positive = hashtag_extract(raw['message'][raw['Analysis'] == 'Positive'])
negative = hashtag_extract(raw['message'][raw['Analysis'] == 'Negative'])
neutral = hashtag_extract(raw['message'][raw['Analysis'] == 'Neutral'])
news = hashtag_extract(raw['message'][raw['Analysis'] == 'News'])

# Creating a cleaning data function, class and pipelines

# cleaning data functions


def cleanup_text(input_text):
    
    #Apply some basic cleanup to the input text.
    #input takes the text as a string
    #param input_text: The input text.
    #return: The cleaned input text

    text = input_text.lower()
    # changing the input_text to lower case
    #Replacing words matching regular expressions
    #We are looking at the most regualr expressions"""

    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"can't", 'cannot', text)
    text = re.sub(r"i'm", 'i am', text)
    text = re.sub(r"ain't", 'is not', text)
    text = re.sub(r"doesn't", 'it does not', text)
    # Removing the username
    text = re.sub('@[^\s]+', '', text)
    # Removing the special characters
    text = re.sub('r<.*?>', ' ', text)
    text = re.sub("^\s+|\s+$", "", text, flags=re.UNICODE)
    # Removing all the punctuations in the text that we may not need
    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")
    text = punctuation.sub("", text.lower())

    #It is important to not remove all stop_words as they might mess up our data during analysis, we will introduce nlp to keep some of the stop words
    nlp_stopwords = NLP(replace_words=True, remove_stopwords=True,
                        remove_numbers=True, remove_punctuations=False)
    text = nlp_stopwords.process(text)
    # Tokenizing the text
    text = text.split()
    # Limmatizing the text with the help of pos tag
    # Pos tag is a speech tag
    pos = pos_tag(text)
    # Only considering words with speech tag
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word, po[0].lower()) if (po[0].lower() in [
                    'n', 'r', 'v', 'a'] and word[0] != '@') else word for word, po in pos])

    # returning the cleaned text
    return text

# function to clean text data


def remove_pattern(input_txt, pattern):
    #The  function will take the first word as an input
    #it will then create a pattern and once a match is found
    #it will replace the input with a space"""
    # find the pattern of the input, e.g, @user or http://
    r = re.findall(pattern, input_txt)

    for i in r:
        # replaces the input with a white sapce
        input_txt = re.sub(i, '', input_txt)

    return input_txt

 # Class for cleaning the data using the above functions


class clean_data(BaseEstimator, TransformerMixin):
    #This initial class helps us clean the the text
    #Uses the above functions to perform the clean of data
    #Helps us to clean data using multi functions

    def __init__(self):
        
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        # Removing the url links applying the above input
        x['message'] = np.vectorize(remove_pattern)(
            x['message'], 'https://t.co/\w+')
        # Removing the retweets(RT)
        x['message'] = np.vectorize(remove_pattern)(x['message'], "RT")
        # Removing the whitespaces and the special characters
        x['message'] = x['message'].str.replace('[^a-zA-Z#]', ' ')
        # Removes the https without url links
        x['message'] = np.vectorize(remove_pattern)(x['message'], "https")
        # Applies the above cleanup_text function to clean the remaining uncleaned data
        x['message'] = x['message'].apply(cleanup_text)
        # Takes words taht have three characters and above
        x['message'] = x['message'].apply(
            lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
        # Removes the whitesapces after splitting and joining them
        x['message'] = x['message'].str.replace(r'[^\w\s]+', '')

        # Returns the cleaned text
        return x


# Cleaning our copied data



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
        # Addjustment figues for the multiple graphs of the hashtag below
        fig, axy = plt.subplots(ncols=2, nrows=2, figsize=(16, 14), dpi=100)
        fig.subplots_adjust(wspace=0.4, hspace=0.3, left=0, right=1)
        # Plotting positive hastags
        sns.barplot(y=positive['hashtag'], x=positive['count'], palette=(
            "Blues_d"), ax=axy[0, 0])
        axy[0, 0].set_title('Frequent Positive climate change hashtags')
        # Plotting news hashtags
        sns.barplot(y=news['hashtag'], x=news['count'],
                    palette=("Blues_d"), ax=axy[0, 1])
        axy[0, 1].set_title('Frequent News climate change hashtags')
        # Plotting neutral hashtags
        sns.barplot(y=neutral['hashtag'], x=neutral['count'],
                    palette=("Blues_d"), ax=axy[1, 0])
        axy[1, 0].set_title('Frequent Neutral climate change hashtags')
        # Plotting negative hashtags
        sns.barplot(y=negative['hashtag'], x=negative['count'], palette=(
            "Blues_d"), ax=axy[1, 1])
        axy[1, 1].set_title('Frequent Negative climate change hashtags', )
        st.pyplot(fig)

        st.markdown('A hashtag—written with a "#" symbol—is used to index keywords or topics on Twitter. This function was created on Twitter,\
            and allows people to easily follow topics they are interested in. Hashtags help you to identify the most trending topics at that moment, thus helping us in analysing our data. It is believed Our most #hashtag trends will be based mostly on  Climate change. The above graphs helps us understand which trends had more engagements and if those trends where either positive, negative, news or just nuetral. This gives us a better insight of our data. ')
        st.markdown('*The above analysis were based on uncleaned data, we  are to clean the data and this time around we are to use word cloud visuals just to have a look at the most common words used in each sentiment. This might give more knowledge as to why our data is not balanced. Balancing the data is something is something we going to do later in this stage.*')
        st.subheader(
            'The following visuals provide the most frequent words used in each tweet sentiment')
        st.markdown('The visuals will be based only on the cleaned data. We can first see the frequent words for the overral sentiment before visualising for each sentiment')

        style.use('seaborn-poster')
        cl = clean_data()
        ct = cl.fit(new_df)
        bd = ct.transform(new_df)
        all_words = " ".join([sentence for sentence in bd['message']])
        wordcloud = WordCloud(
            width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

        # plot the graph
        fig = plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Most used tags of clean tweets')
        plt.axis('off')
        st.pyplot(fig)
        
        #Word graph for each sentiment
        df=bd.groupby('Analysis')
        wc1=" ".join(tweet for tweet in df.get_group('Positive').message)
        wc2=" ".join(tweet for tweet in df.get_group('News').message)
        wc3=" ".join(tweet for tweet in df.get_group('Negative').message)
        wc0=" ".join(tweet for tweet in df.get_group('Neutral').message)
        wc = WordCloud(width=400, height=400, 
               background_color='black', colormap='Dark2',
               max_font_size=150, random_state=42)

        plt.rcParams['figure.figsize'] = [35, 25]
        tweet_list = [wc1, wc2,
                    wc3, wc0]
        title = ['Most used words on Positive tweet', 'Most used words on News tweet', 
                    'Most used words on Negative tweet', 'Most used words on Neutral tweet']

        # Create subplots 
        for i in range(0, len(tweet_list)):
            wc.generate(tweet_list[1])
    
        plt.subplot(2, 2, i + 1)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(title[i])
            
        st.pyplot()

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
                open(os.path.join("resources/linearsvc.pkl"), "rb"))
            prediction = predictor.predict(vect_text)
            if prediction == 0:
                result = 'Neutral Text'
            elif prediction ==1:
                result = 'Positive Text'
            elif prediction == 2:
                result = 'News'
            else:
                result = 'Negative Text'

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as:  {}".format(result))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
