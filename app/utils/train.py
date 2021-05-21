from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import joblib
import os
import math
import app.CONFIG as CONFIG
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def show_dataset():
    if os.path.exists(CONFIG.DATASET_PATH):
        data = np.load(CONFIG.DATASET_PATH)
        x_train, y_train = data['arr_0'], data['arr_1']

        return {
            "status": True,
            "Total training images found": len(x_train),
            "Number of People Registered": len(np.unique(y_train)),
            "People Registered": list(np.unique(y_train))
        }
    else:
        return {
            "status": False
        }

def train(n_neighbors=None):
    try:
        if os.path.isfile(CONFIG.DATASET_PATH):
            data = np.load(CONFIG.DATASET_PATH)
            x_train, y_train = data['arr_0'], data['arr_1']

            n_neighbors = int(np.sqrt(len(np.unique(y_train)))) * 2 + 1

            knn_clf = neighbors.KNeighborsClassifier()
            hyperparameters = {
                "leaf_size": list(range(1, 100)),
                "n_neighbors": [n_neighbors],
                "p": [1, 2],
                "weights": ['uniform', 'distance'],
                "algorithm": ['ball_tree', 'kd_tree', 'brute']
            }
            knncv = GridSearchCV(knn_clf, hyperparameters, cv=10, n_jobs=-1)
            knncv.fit(x_train, y_train)
            joblib.dump(knncv.best_estimator_, CONFIG.MODEL_PATH_KNN)

            # training SVC classifier
            svc = SVC()
            parameters = {
                'C': [0.01, 0.1, 1, 5, 10, 100, 500, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                'probability': [True]
            }
            svccv = GridSearchCV(svc, parameters, cv=7, n_jobs=-1)
            svccv.fit(x_train, y_train)
            joblib.dump(svccv.best_estimator_, CONFIG.MODEL_PATH_SVC)

            return {
                "status": True,
                "message": "Classifiers has been trained successfully !!!"
            }
        else:
            return {
                "status": False,
                "message": "Dataset was not found..."
            }
    except Exception as e:
        return {
                "status": False,
                "message": str(e)
            }
