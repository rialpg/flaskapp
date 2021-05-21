import numpy as np
import app.CONFIG as CONFIG
import os


def delete_person(person_to_delete):
    if os.path.exists(CONFIG.DATASET_PATH):
        data = np.load(CONFIG.DATASET_PATH)
        x_train, y_train = data['arr_0'], data['arr_1']

        if person_to_delete in y_train:
            indexes = np.where(y_train == person_to_delete)
            x_train_new = np.delete(x_train, indexes, axis=0)
            y_train_new = np.delete(y_train, indexes)

            np.savez_compressed(CONFIG.DATASET_PATH, x_train_new, y_train_new)
            print("all done...")
            return {
                "status": True,
                "message": "Person has successfully deleted..."
            }
        else:
            return {
                "status": False,
                "message": "The person doesn't exists in the dataset..."
            }
    else:
        return {
                "status": False,
                "message": "No Data Exists..."
            }

