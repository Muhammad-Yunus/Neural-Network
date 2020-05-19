import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from scipy import sparse

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from keras.layers import Embedding
from keras import metrics

from ml_core.custom_metric import rec, prec, f1

from datetime import datetime
from ml_core.json_utils import writeJson_config
import pandas as pd
import pickle

def get_sparse_training():
    VECT_SEL_PATH = "ml_core/vector_selection/training/tfidf_selection_sparse_training.npz"
    LABEL_DATA_PATH = "ml_core/data/training/Preprocessed_Dataset_Training.csv" 
    
    LABEL_DATA = pd.read_csv(LABEL_DATA_PATH, encoding = "ISO-8859-1", usecols=["label"])
    tags = LABEL_DATA.label

        # ----------------------- LOAD SPARSE MATRIX -------------------------
    tfidf_mat_selection = sparse.load_npz(VECT_SEL_PATH).toarray()

    return [tfidf_mat_selection, tags]

def get_cnn_model():  
    VECT_TEMP_PATH = "ml_core/template/tfidf_sparse_template.npz"
    tfidf_mat_selection = sparse.load_npz(VECT_TEMP_PATH)

    max_len = tfidf_mat_selection.shape[1]
    model = Sequential()
    
    model.add(Embedding(1000,
                        20,
                        input_length=max_len))
    # model.add(Dropout(0.2))
    model.add(Conv1D(64,
                    3,
                    padding='valid',
                    activation='relu',
                    strides=1))
    # model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1))
    model.add(Dropout(0.2))
    model.add(Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['acc', rec, prec])
    return model

def classificationTraining(cnn_model, tfidf_mat_selection, tags):
    MODEL_FOLDER = "ml_core/model/"

    # split dataset 
    tfidf_mat_train, tfidf_mat_test, tags_train, tags_test = \
            train_test_split(tfidf_mat_selection, tags, test_size=0.25, random_state=42)



    def check_model(model,x,y): 
        #return model.fit(x,y,batch_size=63,epochs=20,verbose=1,validation_split=0.15)
        return model.fit(x, y, verbose=1, validation_split=0.2)

    # ----------------------------- START TRAINING MODEL ------------------------------
    estimator = KerasClassifier(build_fn=cnn_model, epochs=25, batch_size=6)
    
    history = check_model(estimator,tfidf_mat_train, tags_train.ravel())

    # ------------------------------ TEST REPORT --------------------------------
    tags_pred = estimator.predict(tfidf_mat_test)
    
    def print_cm(y_true, y_pred, labels_order) :
        df = pd.DataFrame(
                    confusion_matrix(y_true, y_pred, labels=labels_order), 
                    index=['target : 1', 'target : 0'], 
                    columns=['pred : 1', 'pred : 0']
                )
        df.style.set_properties(**{'text-align': 'center'})
        return df

    cm_model = print_cm(tags_pred, tags_test, [1, 0])
    report_model = classification_report(tags_pred, tags_test, output_dict=True)

    print(cm_model)
    print(report_model)
    # ------------------------------ SAVE MODEL & HISTORY ------------------------------    
    
    estimator.model.save(MODEL_FOLDER + "cnn_model_training.h5")

    pickle.dump(estimator.classes_, open(MODEL_FOLDER + 'cnn_class_training.pkl','wb'))
    # class_json = {}
    # class_json['class'] = estimator.classes_
    # writeJson_config(MODEL_FOLDER , 'cnn_class_training.json', class_json, append=False)

    def formatStr(floats):
        return ['{:.2f}'.format(x) for x in floats]

    json_hist = {}
    json_hist["acc"] = formatStr(history.history['acc'])
    json_hist["val_acc"] = formatStr(history.history['val_acc'])
    json_hist["prec"] = formatStr(history.history['prec'])
    json_hist["val_prec"] = formatStr(history.history['val_prec'])
    json_hist["rec"] = formatStr(history.history['rec'])
    json_hist["val_rec"] = formatStr(history.history['val_rec'])
    json_hist["loss"] = formatStr(history.history['loss'])
    json_hist["val_loss"] = formatStr(history.history['val_loss'])
    writeJson_config(MODEL_FOLDER + "history/" , "cnn_history_model.json", json_hist, append=False)

    json_report = {}
    json_report['confusion_matrix'] = cm_model.values.tolist()
    json_report['report'] = report_model
    writeJson_config(MODEL_FOLDER + "report/" , "cnn_report_model.json", json_report, append=False)

    return 'success'