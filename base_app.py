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
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
from nlppreprocess import NLP
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk import pos_tag
import string
import re
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
import streamlit as st
import joblib
import os

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
plt.rcParams['figure.dpi'] = 180


# Preprocessing and cleaning dependencies
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


def cleanup_text(text):

    # Apply some basic cleanup to the input text.
    # input takes the text as a string
    # param input_text: The input text.
    # return: The cleaned input text

    # This function uses regular expressions to remove url's, mentions, hashtags,
    # punctuation, numbers and any extra white space from tweets after converting
    # everything to lowercase letters.

    # Input:
    # tweet: original tweet
    #datatype: 'str'
    text = text.lower()
    # Remove mentions
    text = re.sub('@[\w]*', '', text)

    # Remove url's
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)

    # Remove hashtags
    text = re.sub(r'#\w*', '', text)

    # Removes the retweets(rt)
    text = re.sub(r'rt', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Removes the https
    text = re.sub(r'https', '', text)

    # Remove punctuation
    text = re.sub(r"[,.;':@#?!\&/$]+\ *", ' ', text)

    # Remove that funny diamond
    text = re.sub(r"U+FFFD ", ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s\s+', ' ', text)

    # Remove space in front of text
    text = text.lstrip(' ')
    # Removes stopwords
    nlp_for_stopwords = NLP(replace_words=True, remove_stopwords=True,
                            remove_numbers=True, remove_punctuations=False)
    # This will remove stops words that are not necessary. The idea is to keep words like [is, not, was]
    text = nlp_for_stopwords.process(text)

    # tokenisation
    # We used the split method instead of the word_tokenise library

    text = text.split()

    # POS
    # Part of Speech tagging is essential to ensure Lemmatization perfoms well.
    pos = pos_tag(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word, po[0].lower())
                     if (po[0].lower() in ['n', 'r', 'v', 'a'] and word[0] != '@') else word for word, po in pos])

    return text


# Cleaning our copied data


def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information", "Analysis", "Prediction"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder

        # Mini Introduction
        st.markdown("Twitter data (commonly know as tweets) is a incredibly powerful source of information on an extensive list of topics. This data will be analyzed to find trends related to climate change, measure popular sentiment, obtain feedback on past desicions and also help make future desicions. \
            With this context, We built  a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data.")

        st.subheader('Brief Discussion of models used our app')

        # Random forest classifier model
        st.info('RANDOM FOREST CLASSIFIER(RF)')
        st.markdown('Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset. \
            Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.')
        st.markdown('During thr model evaluation, the random forest classifier produced a lower accurary, we cannot really rely on this model when we need to classify tweet data.')

        # Decision tree classifier
        st.info('DECISION TREE CLASSIFIER(DECISION_TREE)')
        st.markdown('It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.')
        st.markdown(
            'Decision trees easily overdrift, during evaluation the decision had a poor accuray affected by the over-drifting of the model.')

        # K-nearest Neighberhood model
        st.info('K-NEAREST NEIGHBERHOOD(KNN)')
        st.markdown('A k-nearest-neighbor is a data classification algorithm that attempts to determine what group a data point is in by looking at the data points around it. \
                    An algorithm, looking at one point on a grid, trying to determine if a point is in group A or B, looks at the states of the points that are near it.\
                    The range is arbitrarily determined, but the point is to take a sample of the data. If the majority of the points are in group A, then it is likely that the data point in question will be A rather than B, and vice versa.')
        st.markdown('During model evaluating, the KNN model did produce some good results, which i believe it will be useful for text data that will be classified using this application.')

        # Multinomial model
        st.info('MULTINOMAIL NAIVE BAYES(NB)')
        st.markdown('The Multinomial Naive Bayes algorithm is a Bayesian learning approach popular in Natural Language Processing (NLP). The program guesses the tag of a text, such as an email or a newspaper story, using the Bayes theorem. It calculates each tag likelihood for a given sample and outputs the tag with the greatest chance. \
            The Naive Bayes classifier is made up of a number of algorithms that all have one thing in common: each feature being classed is unrelated to any other feature. A feature existence or absence has no bearing on the inclusion or exclusion of another feature')
        st.markdown('During model evaluating, the Multinomial model did produce some good results, which i believe it will be useful for text data that will be classified using this application.')

        # Logistic Regression
        st.info('LOGISTIC REGRESSION(LR)')
        st.markdown('Logistic Regression is a ‘Statistical Learning’ technique categorized in ‘Supervised’ Machine Learning (ML) methods dedicated to ‘Classification’ tasks.')
        st.markdown('The logistic regression produced best accuray results with the help of hyperparameters. It is the best model to use for tweet classificaton')

        # Linear Support Vector Machine
        st.info('LINEAR SUPPORT VECTOR MACHINE')
        st.markdown('SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.')
        st.markdown('The Linear Support Vector Machine produced outstanding average accuray results with the help of hyperparameters. It is the top model to use for tweet classificaton')

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

        # cleaning the data
        new_df['message'] = new_df['message'].apply(cleanup_text)

        # plotting a word cloud for the new data
        all_words = " ".join([sentence for sentence in new_df['message']])
        wordcloud = WordCloud(
            width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

        # plot the graph
        fig = plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Frequent words used for the overrall tweets')
        plt.axis('off')
        st.pyplot(fig)

        st.markdown('The most words that will appear overral are the words that had the most hashtags, this might be that it was the most topic people engaged in. The word cloud gives you a better understand of the inside text data and helps us understand which frequent words were used in our text data.')

        # Word graph for each sentiment
        df = new_df.groupby('Analysis')
        wc1 = " ".join(tweet for tweet in df.get_group('Positive').message)
        wc2 = " ".join(tweet for tweet in df.get_group('News').message)
        wc3 = " ".join(tweet for tweet in df.get_group('Negative').message)
        wc0 = " ".join(tweet for tweet in df.get_group('Neutral').message)

        pos = WordCloud(width=600, height=400,
                        colormap='Greens', background_color='black',
                        max_font_size=180, random_state=42).generate(wc1)
        nes = WordCloud(width=600, height=400,
                        colormap='Blues', background_color='black',
                        max_font_size=180, random_state=42).generate(wc2)
        neu = WordCloud(width=600, height=400,
                        colormap='Dark2', background_color='black',
                        max_font_size=180, random_state=42).generate(wc3)
        neg = WordCloud(width=600, height=400,
                        colormap='Reds', background_color='black',
                        max_font_size=180, random_state=42, stopwords='english').generate(wc0)
        f, axarr = plt.subplots(2, 2, figsize=(35, 25))
        plt.subplots_adjust(wspace=0.02, hspace=0.1)
        axarr[0, 0].imshow(pos, interpolation="bilinear")
        axarr[0, 1].imshow(nes, interpolation="bilinear")
        axarr[1, 0].imshow(neu, interpolation="bilinear")
        axarr[1, 1].imshow(neg, interpolation="bilinear")

        # Remove the ticks on the x and y axes
        for ax in f.axes:
            plt.sca(ax)
            plt.axis('off')

        axarr[0, 0].set_title('Positive climate change\n', fontsize=35)
        axarr[0, 1].set_title('News\n', fontsize=35)
        axarr[1, 0].set_title('Neutral\n', fontsize=35)
        axarr[1, 1].set_title('Negative Climate Change\n', fontsize=35)

        st.pyplot()
    # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        # differentiating between a single text and a dataset inpit
        data_source = ['Select type of Data', 'Tweet Text', 'CSV file']

        source_selection = st.selectbox('Type of Data', data_source)

        # Fuction to return a key value
        def get_keys(val, my_dict):
            for key, value in my_dict.items():
                if val == value:
                    return key

        if source_selection == 'Tweet Text':
            st.info(
                '**Classification for single tweet text, this is limited to 140 Characters as per tweet.**')
            tweet_text = st.text_area("Enter Text Below:")
            model_name = ["LinearSVC", "NB", "RFOREST",
                          "DECISION_TREE", "LR", "KNN"]
            model_choice = st.selectbox(
                "Select a Classifier  Model", model_name)

            prediction_labels = {
                'Negative Tweet': -1, 'Neutral Tweet': 0, 'Positive Tweet': 1, 'News Tweet': 2}
            # Classifying using different models
            if st.button("Classify"):

                # Cleaning the input text
                tweet_text = cleanup_text(tweet_text)
                # Transforming user input with vectorizer
                vect_text = tweet_cv.transform([tweet_text]).toarray()
                # Load your .pkl file with the model of your choice + make predictions
                # Try loading in multiple models to give the user a choice

                # Linear SVC Model
                if model_choice == "LinearSVC":
                    predictor = joblib.load(
                        open(os.path.join("resources/linearsvc.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                # MutilNomial Naive Model
                if model_choice == "NB":
                    predictor = joblib.load(
                        open(os.path.join("resources/MultinomialNB.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                # Random Forest Model
                if model_choice == "RFOREST":
                    predictor = joblib.load(
                        open(os.path.join("resources/RandomForestClassifier.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                # Decision Tree Classifer Model
                if model_choice == "DECISION_TREE":
                    predictor = joblib.load(
                        open(os.path.join("resources/DecisionTreeClassifier.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                # Logistic Regression Model
                if model_choice == "LR":
                    predictor = joblib.load(
                        open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                # K-Neighborhood Naive Model
                if model_choice == "KNN":
                    predictor = joblib.load(
                        open(os.path.join("resources/KNeighborsClassifier.pkl.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                final_result = get_keys(prediction, prediction_labels)
                # When model has successfully run, will print prediction
                # You can use a dictionary or similar structure to make this output
                # more human interpretable.
                st.success("Text Categorized as:  {}".format(final_result))

        if source_selection == 'CSV file':
            # CSV File Prediction
            st.subheader('Tweet Classification for csv file')

            model_name = ["LinearSVC", "NB", "RFOREST",
                          "DECISION_TREE", "LR", "KNN"]

            prediction_labels = {'Negative': -1,
                                 'Neutral': 0, 'Positive': 1, 'News': 2}
            data = st.file_uploader("Select a CSV file", type="csv")
            if data is not None:
                data = pd.read_csv(data)

            uploaded_dataset = st.checkbox('Click to view uploaded file')
            if uploaded_dataset:
                st.dataframe(data.head(25))
            st.markdown(
                'Please type the name column you wish to classify as the way it is named in the file uploaded.')
            new_column = st.text_area('Enter the column name:')

            model_choice = st.selectbox(
                "Select a Classifier  Model", model_name)

            if st.button("Classify"):
                # Cleaning the input data cloumn

                dt = data[new_column].apply(cleanup_text)

                # Transforming user cleaned data with vectorizer
                vect_text = tweet_cv.transform([dt]).toarray()
                # Load your .pkl file with the model of your choice + make predictions
                # Try loading in multiple models to give the user a choice

                # Linear SVC Model
                if model_choice == "LinearSVC":
                    predictor = joblib.load(
                        open(os.path.join("resources/linearsvc.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                # MutilNomial Naive Model
                if model_choice == "NB":
                    predictor = joblib.load(
                        open(os.path.join("resources/MultinomialNB.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                # Random Forest Model
                if model_choice == "RFOREST":
                    predictor = joblib.load(
                        open(os.path.join("resources/RandomForestClassifier.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                # Decision Tree Classifer Model
                if model_choice == "DECISION_TREE":
                    predictor = joblib.load(
                        open(os.path.join("resources/DecisionTreeClassifier.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                # Logistic Regression Model
                if model_choice == "LR":
                    predictor = joblib.load(
                        open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)

                # K-Neighborhood Naive Model
                if model_choice == "KNN":
                    predictor = joblib.load(
                        open(os.path.join("resources/KNeighborsClassifier.pkl.pkl"), "rb"))
                    prediction = predictor.predict(vect_text)
                data['sentiment'] = prediction
                final_result = get_keys(prediction, prediction_labels)
                # When model has successfully run, will print prediction
                # You can use a dictionary or similar structure to make this output
                # more human interpretable.
                st.success("Text Categorized as:  {}".format(final_result))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
