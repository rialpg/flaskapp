import os

################ Configuration File #######################

PHOTOS_BACKUP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/photos_backup')
TEMP_FILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/temp')

# paths to saved items
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/data/dataset.npz')
MODEL_PATH_SVC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/data/svc_classifier.sav')
MODEL_PATH_KNN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/data/knn_classifier.sav')

################# Parameters to Consider #####################

TRAINING_IMAGES = 10
CONFIDENCE_THRESHOLD = 0.7
GPU = False
RESIZE_SCALE = 1.0

##############################################################