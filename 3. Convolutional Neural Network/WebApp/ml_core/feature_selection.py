from sklearn.feature_selection import mutual_info_classif
from ml_core.json_utils import readJson_config, writeJson_config
from datetime import datetime
from scipy import sparse
import numpy as np
import os
import ast

def featureSelection():

    threshold=0.01
    np.set_printoptions(suppress=True)

    # ----------------- READ PREPROCESSING FILE -----------------------
    VECT_SEL_FOLDER = "ml_core/vector_selection/"
    VECT_FOLDER = "ml_core/vector/"
    
    VECT_TEMPLATE = "ml_core/template/tfidf_sparse_template.npz"
    TEMPLATE_FOLDER = "ml_core/template/"
    FEATURE_CONFIG = "feature_template.json"

    FileName = []
    
    for filename in os.listdir(VECT_FOLDER):
        path = os.path.join(VECT_FOLDER, filename)
        if not os.path.isdir(path):
            strDatetime = filename.replace("tfidf_sparse_", "").replace(".npz", "")
            FileDatetime = datetime.strptime(strDatetime, "%d%m%Y_%H%M%S")
            FileName.append([filename, FileDatetime])

    
    # ----------------------- LOAD SPARSE MATRIX -------------------------
    FileName = sorted(FileName, key=lambda t: t[1], reverse=True)
    tfidf_mat = sparse.load_npz(VECT_FOLDER + FileName[0][0]).toarray()

    json_name = (FileName[0][0]).replace(".npz", ".json").replace("sparse", "feature")
    features = readJson_config(VECT_FOLDER + "features/", json_name, 'feature')[0]

    tfidf_mat_selection = None
    features_template = None
    tfidf_mat_template = None
    selected_idx = []

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    # ------------------------------- RUN -------------------------------
    tfidf_mat_template = []
    for i in range(len(tfidf_mat)):
        tfidf_mat_template.append(sparse.load_npz(VECT_TEMPLATE).toarray()[0])

    features_template = readJson_config(TEMPLATE_FOLDER, FEATURE_CONFIG, 'feature')
    print(features_template)

    for i in range(len(tfidf_mat)):
        for feature in features:
            if feature in features_template:
                idx_template = features_template.index(feature)
                idx = features.index(feature)
                tfidf_mat_template[i][idx_template] = tfidf_mat[i][idx]
                selected_idx.append(1)
            else :
                selected_idx.append(0)
        
    #-------------------------------- SAVE -----------------------------------
    tfidf_sparse_template = sparse.csr_matrix(tfidf_mat_template)
    sparse.save_npz(VECT_SEL_FOLDER + "tfidf_selection_sparse_" + dt_string + ".npz", tfidf_sparse_template)
    writeJson_config(VECT_SEL_FOLDER + "features/", ("tfidf_sparse_" + dt_string + ".json"), features_template, append=False)

    # ----------------------------- LOAD DATA VIEW ----------------------------
    tableRecords = []
    for i in range(len(tfidf_mat)):
        for item in zip(features, tfidf_mat[i], selected_idx):
            tfidf_round = np.array([item[1]]).round(decimals=3)
            if item[1] != 0.0:
                tableRecords.append(['Document_' + str(i), item[0], tfidf_round[0], item[2]])

    return tableRecords