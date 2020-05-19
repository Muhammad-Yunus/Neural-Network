from sklearn.feature_selection import mutual_info_classif
from ml_core.json_utils import readJson_config, writeJson_config
from datetime import datetime
from scipy import sparse
import pandas as pd
import numpy as np
import os
import ast

def featureSelectionTraining(training=False):

    threshold=0.01
    np.set_printoptions(suppress=True)

    # ----------------- READ PREPROCESSING FILE -----------------------
    VECT_SEL_FOLDER = "ml_core/vector_selection/training/" 
    VECT_FOLDER = "ml_core/vector/training/" 
    
    VECT_TEMPLATE = "ml_core/template/tfidf_sparse_template.npz"
    TEMPLATE_FOLDER = "ml_core/template/"
    FEATURE_CONFIG = "feature_template.json"
    
    LABEL_PATH = "ml_core/data/training/Preprocessed_Dataset_Training.csv"
    TWEET_DATA = pd.read_csv(LABEL_PATH, usecols=["label"])
    tags = TWEET_DATA.label

    # ----------------------- LOAD SPARSE MATRIX -------------------------
    FileName = "tfidf_sparse_training.npz"
    tfidf_mat = sparse.load_npz(VECT_FOLDER + FileName).toarray()
    
    json_feature = "tfidf_feature_training.json"
    features = readJson_config(VECT_FOLDER + "features/", json_feature, 'feature')[0]

    tfidf_mat_selection = None
    features_template = None
    tfidf_mat_template = None
    selected_idx = []

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    # ---------------------------- TRAINING -----------------------------
    mi = mutual_info_classif(tfidf_mat, tags)
    norm_mi = mi/np.max(mi)

    column_idx = [i for i, mi_item in enumerate(norm_mi) if mi_item < 0.01]
    tfidf_mat_selection = np.delete(tfidf_mat, column_idx ,1)

    # template data
    selected_idx = [j for j in range(len(norm_mi)) if j not in column_idx]
    selected_features = []
    for idx in selected_idx:
        selected_features.append(features[idx])
    
    tfidf_mat_template = [0.0] * len(selected_features)
    features_template = selected_features
    
    #-------------------------------- SAVE -----------------------------------
    # Save template 
    tfidf_sparse_template = sparse.csr_matrix(tfidf_mat_template)
    sparse.save_npz(VECT_TEMPLATE, tfidf_sparse_template)         

    feature_dict = {}
    feature_dict['feature'] = features_template
    writeJson_config(TEMPLATE_FOLDER, FEATURE_CONFIG, feature_dict, append=False)

    # save training data
    tfidf_sparse = sparse.csr_matrix(tfidf_mat_selection)
    sparse.save_npz(VECT_SEL_FOLDER + "tfidf_selection_sparse_training.npz", tfidf_sparse)
    writeJson_config(VECT_SEL_FOLDER + "features/","tfidf_feature_training.json", features_template, append=False)

    return 'success'