from datetime import datetime
import pandas as pd 
import numpy as np
import ast
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize

from scipy import sparse
from ml_core.json_utils import readJson_config, writeJson_config

def featureEngineering():
    # ----------------- READ PREPROCESSING FILE -----------------------
    DATA_FOLDER = "ml_core/data/"
    VECT_FOLDER = "ml_core/vector/"
    FEATURES_FOLDER = "ml_core/vector/features/"

    FileName = []
    for filename in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, filename)
        if not os.path.isdir(path):
            strDatetime = filename.replace("Preprocessed_Dataset_", "").replace(".csv", "")
            FileDatetime = datetime.strptime(strDatetime, "%d%m%Y_%H%M%S")
            FileName.append([filename, FileDatetime])

    FileName = sorted(FileName, key=lambda t: t[1], reverse=True)

    TWEET_DATA = pd.read_csv(DATA_FOLDER + FileName[0][0], usecols=["tweet_tokens_stemmed"])
    TWEET_DATA.columns = ["tweet"]

    # join list of token as single document string
    def join_text_list(texts):
        texts = ast.literal_eval(texts)
        return ' '.join([text for text in texts])
    TWEET_DATA["tweet_join"] = TWEET_DATA["tweet"].apply(join_text_list)


    #------------------------- READ CONFIG ---------------------------
    ses_max_feature = readJson_config('ml_core/', 'configuration.json', 'max_features')
    max_features = int(ses_max_feature[0]) if ses_max_feature is not None else 1000

    # ------------------------- MAIN CALC ----------------------------
    # ngram_range (1, 3) to use unigram, bigram, trigram
    cvect = CountVectorizer(max_features=max_features, ngram_range=(1,1))
    counts = cvect.fit_transform(TWEET_DATA["tweet_join"])

    normalized_counts = normalize(counts, norm='l1', axis=1)

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,1), smooth_idf=False)
    tfs = tfidf.fit_transform(TWEET_DATA["tweet_join"])
    tfidf_sparse = normalized_counts.multiply(tfidf.idf_)
    
    feature_name = {}
    feature_name['feature'] = tfidf.get_feature_names()

    tfidf_mat = tfidf_sparse.toarray()

    #------------------------ SAVE -----------------------------------
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    tfidf_sparse = sparse.csr_matrix(tfidf_sparse)

    sparse.save_npz(VECT_FOLDER + "tfidf_sparse_" + dt_string + ".npz", tfidf_sparse)
    writeJson_config(FEATURES_FOLDER, ("tfidf_feature_" + dt_string + ".json"), feature_name, append=False)

    #------------------------- DATA VIEW --------------------------------
    TableRecords = []

    terms = tfidf.get_feature_names()
    TF = normalized_counts.toarray()
    IDF = tfidf.idf_
    TFIDF = tfidf_mat

    for i in range(len(TF)):
        for item in zip(terms,TF[i], IDF, TFIDF[i]):
            if item[1] != 0.0:
                Num = np.array([item[1], item[2], item[3]]).round(decimals=3)
                TableRecords.append(['Document_'+ str(i) , item[0], Num[0], Num[1], Num[2]])
            
    return TableRecords