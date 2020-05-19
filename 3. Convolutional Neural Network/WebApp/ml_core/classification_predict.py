# from keras.models import load_model

from scipy import sparse

from datetime import datetime
import numpy as np
import os

# def model_definition():
#     MODEL_PATH = "ml_core/model/cnn_model_training.h5"
#     model = load_model(MODEL_PATH, custom_objects={"rec": rec, "prec": prec, "f1": f1})
#     model.summary()
#     return model

def load_sparse():
    VECT_SEL_FOLDER = "ml_core/vector_selection/"

    FileName = []

    for filename in os.listdir(VECT_SEL_FOLDER):
        path = os.path.join(VECT_SEL_FOLDER, filename)
        if not os.path.isdir(path):
            strDatetime = filename.replace("tfidf_selection_sparse_", "").replace(".npz", "")
            FileDatetime = datetime.strptime(strDatetime, "%d%m%Y_%H%M%S")
            FileName.append([filename, FileDatetime])

    FileName = sorted(FileName, key=lambda t: t[1], reverse=True)

    X  = sparse.load_npz(VECT_SEL_FOLDER + FileName[0][0])

    return X