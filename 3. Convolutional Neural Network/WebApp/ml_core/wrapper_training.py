from ml_core.preprocessing_training import preprocessingTraining
from ml_core.feature_engineering_training import featureEngineeringTraining
from ml_core.feature_selection_training import featureSelectionTraining
from ml_core.classification_training import classificationTraining, get_cnn_model, get_sparse_training

def run_ml_pipeline():
    # preprocessing
    preprocessingTraining()

    # feature engineering
    featureEngineeringTraining()

    # feature selection
    featureSelectionTraining()

    # train model & validat
    tfidf_mat_selection, tags = get_sparse_training()
    classificationTraining(get_cnn_model, tfidf_mat_selection, tags)
    return 'success'