import pandas as pd 
import numpy as np
import string 
import swifter
import re

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


def preprocessingTraining():
    init_stopwords = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 'aja']
    DATA_FOLDER = "ml_core/data/training/"
    RAW_DATA_PATH = "ml_core/raw_data/Raw_Dataset_Training.csv"

    TWEET_DATA = pd.read_csv(RAW_DATA_PATH, encoding = "ISO-8859-1")

    # ------ Case Folding --------
    TWEET_DATA['tweet'] = TWEET_DATA['tweet'].str.lower()

    # ------ Tokenizing ---------
    #remove number
    def remove_number(text):
        return  re.sub(r"\d+", "", text)

    TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_number)

    #remove punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans("","",string.punctuation))

    TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_punctuation)

    #remove whitespace leading & trailing
    def remove_whitespace_LT(text):
        return text.strip()

    TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_whitespace_LT)

    #remove multiple whitespace into single whitespace
    def remove_whitespace_multiple(text):
        return re.sub('\s+',' ',text)

    TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_whitespace_multiple)


    # NLTK word rokenize 
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    TWEET_DATA['tweet_tokens'] = TWEET_DATA['tweet'].apply(word_tokenize_wrapper)

    # get stopword indonesia
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend(init_stopwords)
    list_stopwords = set(list_stopwords)

    #remove stopword pada list token
    def stopwords_removal(words):
        return [word for word in words if not word in list_stopwords]

    TWEET_DATA['tweet_tokens_WSW'] = TWEET_DATA['tweet_tokens'].apply(stopwords_removal)

    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in TWEET_DATA['tweet_tokens_WSW']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '
                
    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)

    # apply stemmed term to dataframe
    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    TWEET_DATA['tweet_tokens_stemmed'] = TWEET_DATA['tweet_tokens_WSW'].swifter.apply(get_stemmed_term)

    TWEET_DATA.to_csv(DATA_FOLDER + "Preprocessed_Dataset_Training.csv")

    return 'success'