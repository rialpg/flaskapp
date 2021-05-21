import numpy as np
import cv2
import os
import shutil
import face_recognition
from imutils.video import WebcamVideoStream

import app.CONFIG as CONFIG
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Capture_Images(object):
    def __init__(self, new_name, training_images=10, gpu=False):
        self.stream = WebcamVideoStream(src=0).start()
        self.new_name = None
        self.count = 0
        self.encodings = []
        self.labels = []
        self.training_images = training_images
        self.gpu = gpu
        self.registered = False
        self.entered_name = new_name
        self.name_to_register = self._check_name()

    def __del__(self):
        self.stream.stop()

    def _check_name(self):
        if os.path.exists(CONFIG.DATASET_PATH):
            data = np.load(CONFIG.DATASET_PATH)
            registered_people = np.unique(data['arr_1'])
            if self.entered_name in registered_people:
                return None
        return self.entered_name

    def save_embeddings(self):
        # save arrays to one file in compressed format
        if os.path.exists(CONFIG.DATASET_PATH):
            # load old data
            data = np.load(CONFIG.DATASET_PATH)
            x_train_old, y_train_old = data['arr_0'], data['arr_1']
            # merge with old data
            x_train_all = np.concatenate((x_train_old, self.encodings))
            y_train_all = np.concatenate((y_train_old, self.labels))
            np.savez_compressed(CONFIG.DATASET_PATH, x_train_all, y_train_all)
        else:
            np.savez_compressed(CONFIG.DATASET_PATH,
                                self.encodings, self.labels)

    def capture_and_process_images(self):
        frame = self.stream.read()

        if self.name_to_register is not None:

            if self.gpu:
                face_bboxes = face_recognition.face_locations(
                    frame, model="cnn")
            else:
                face_bboxes = face_recognition.face_locations(frame)

            # ignore the frame with no faces detected or multiple faces detected
            if len(face_bboxes) == 1:
                if self.gpu:
                    face_enc = face_recognition.face_encodings(
                        frame, known_face_locations=face_bboxes, model="cnn")[0]
                else:
                    face_enc = face_recognition.face_encodings(
                        frame, known_face_locations=face_bboxes)[0]

                # Add face encoding for current image with corresponding label (name) to the training data
                self.encodings.append(face_enc)
                self.labels.append(self.name_to_register)

                # Crop the image frame into rectangle
                y1, x2, y2, x1 = face_bboxes[0]

                # Display the video frame, with bounded rectangle on the person's face, person name and frame count
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(self.count), (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

                if not self.registered:
                    cv2.putText(frame, "Registering new person: " + self.name_to_register,
                                (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"{self.name_to_register} has been Registered !!! ", (
                        50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                self.count += 1
            cv2.waitKey(200)

            # If image taken reach 20, stop taking video
            if self.count == self.training_images:
                self.save_embeddings()
                self.registered = True

        else:
            _str = f"{self.entered_name} is already Registered.\nTry Another name !!!"
            cv2.putText(frame, _str, (50, 80),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        data = []
        data.append(jpeg.tobytes())
        return data


def register_capture_images(camera):
    while True:
        data = camera.capture_and_process_images()
        frame = data[0]
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# this function takes images and process them
def process_existing_images(name, training_images=10, gpu=False):
    _registered_classes = []
    if os.path.exists(CONFIG.DATASET_PATH):
        # load old data
        data = np.load(CONFIG.DATASET_PATH)
        y_train = data['arr_1']
        _registered_classes = np.unique(y_train)

    if os.path.exists(CONFIG.TEMP_FILES_PATH):
        images = os.listdir(CONFIG.TEMP_FILES_PATH)
        if len(images) == 0:
            return {
                "status": False,
                "message": "No images found..."
            }
    else:
        return {
            "status": False,
            "message": "No images found..."
        }

    encodings = []
    labels = []
    counter = 0
    is_registered = False
    is_saved = False

    for image_path in os.listdir(CONFIG.TEMP_FILES_PATH):
        # if person already present in database
        if len(_registered_classes) > 0:
            if name in _registered_classes:
                for image_path in os.listdir(CONFIG.TEMP_FILES_PATH):
                    os.remove(os.path.join(CONFIG.TEMP_FILES_PATH, image_path))
                return {
                    "status": False,
                    "message": "current person has already been registered..."
                }

        # if the file is not image then proceed to the next file
        if image_path.split(".")[-1] not in ["jpg", "JPG", "png", "jpeg"]:
            continue
        
        image_path = os.path.join(CONFIG.TEMP_FILES_PATH, image_path)
        try:
            frame = cv2.imread(image_path)
            # first detecting faces in an image (finding face locations)
            if gpu:
                face_bboxes = face_recognition.face_locations(frame, model="cnn")
            else:
                face_bboxes = face_recognition.face_locations(frame)

            # if an image has exactly one person
            if len(face_bboxes) == 1:
                
                if not is_saved:
                    shutil.copy(image_path, os.path.join(CONFIG.PHOTOS_BACKUP_PATH, name + ".jpg"))
                    is_saved = True

                # find face embeddings
                if gpu:
                    face_enc = face_recognition.face_encodings(
                        frame, known_face_locations=face_bboxes, model="cnn")[0]
                else:
                    face_enc = face_recognition.face_encodings(
                        frame, known_face_locations=face_bboxes)[0]

                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                labels.append(name)
                counter += 1

                # if images has already been added, then save data, break the loop and proceed to the next person
                if counter >= training_images:
                    print('\nsaving data...')
                    save_embeddings(encodings, labels)
                    print("{} registered successfully...\n".format(name))
                    is_registered = True
                    break
        except Exception as e:
            pass
    
    # removing temp files
    for image_path in os.listdir(CONFIG.TEMP_FILES_PATH):
        os.remove(os.path.join(CONFIG.TEMP_FILES_PATH, image_path))

    if not is_registered:
        print("\nperson was not registered sucessfully due to the following reasons...")
        print("1. either the images contained multiple faces or no faces")
        print("2. the number of images were less than required number ({} default)\n".format(training_images))
        return {
            "status": False,
            "message": "Not registered due to wrong or less number of images"
        }

    return {
            "status": True,
            "message": f"{name} registered successfully !!!"
        }


def save_embeddings(encodings, labels):
    # save arrays to one file in compressed format
    if os.path.exists(CONFIG.DATASET_PATH):
        # load old data
        data = np.load(CONFIG.DATASET_PATH)
        x_train_old, y_train_old = data['arr_0'], data['arr_1']
        # merge with old data
        x_train_all = np.concatenate((x_train_old, encodings))
        y_train_all = np.concatenate((y_train_old, labels))
        np.savez_compressed(CONFIG.DATASET_PATH, x_train_all, y_train_all)
    else:
        np.savez_compressed(CONFIG.DATASET_PATH, encodings, labels)