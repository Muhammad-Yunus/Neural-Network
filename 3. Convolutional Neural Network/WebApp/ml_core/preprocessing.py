from datetime import datetime
import pandas as pd 
import numpy as np
import string 
import swifter
import re
import ast

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def TextPreprocessing(documents):
    
    documents = ast.literal_eval(documents)
    TWEET_DATA = pd.DataFrame(documents, columns =['tweet'])

    # ------ Case Folding --------
    # gunakan fungsi Series.str.lower() pada Pandas
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

    # NLTK calc frequency distribution
    def freqDist_wrapper(text):
        return FreqDist(text)

    TWEET_DATA['tweet_tokens_fdist'] = TWEET_DATA['tweet_tokens'].apply(freqDist_wrapper)


    # get stopword indonesia
    list_stopwords = stopwords.words('indonesian')

    # append additional stopword
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 'aja'])

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

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    TWEET_DATA.to_csv("ml_core/data/Preprocessed_Dataset_" + dt_string + ".csv")
    
    BF = TWEET_DATA['tweet_tokens_WSW'].to_list()
    AT = TWEET_DATA['tweet_tokens_stemmed'].to_list()
    
    ResultArray = []
    for i in range(len(BF)):
        for j in range(len(BF[i])):
            list_item = [ 'Document_' + str(i), BF[i][j], AT[i][j], 1]
            if list_item in ResultArray:
                idx = ResultArray.index(list_item)
                ResultArray[idx][3] += 1 
            else :  
                ResultArray.append(list_item) 
    
    return ResultArray