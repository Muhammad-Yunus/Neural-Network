import pandas as pd 
import numpy as np
import ast

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize

from scipy import sparse
from ml_core.json_utils import readJson_config, writeJson_config

def featureEngineeringTraining():
    # ----------------- READ PREPROCESSING FILE -----------------------
    DATA_FOLDER = "ml_core/data/training/"
    VECT_FOLDER = "ml_core/vector/training/"
    FEATURES_FOLDER = "ml_core/vector/training/features/"

    FileName = "Preprocessed_Dataset_Training.csv"

    TWEET_DATA = pd.read_csv(DATA_FOLDER + FileName, usecols=["tweet_tokens_stemmed"])
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
    tfidf_sparse = sparse.csr_matrix(tfidf_sparse)

    sparse.save_npz(VECT_FOLDER + "tfidf_sparse_training.npz", tfidf_sparse)
    writeJson_config(FEATURES_FOLDER, ("tfidf_feature_training.json"), feature_name, append=False)

            
    return 'success'