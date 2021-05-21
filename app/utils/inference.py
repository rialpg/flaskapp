import os
import cv2
import numpy as np
import joblib
import app.CONFIG as CONFIG
import face_recognition
import shutil
from imutils.video import WebcamVideoStream

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)


class Infernce(object):
    def __init__(self, threshold=0.8, resize_scale=0.75, gpu=False):
        self.stream = WebcamVideoStream(src=0).start()
        self.threshold = threshold
        self.resize_scale = resize_scale
        self.gpu = gpu
        self._classes = None
        self.knn_clf = None
        self.svc_clf = None

        if os.path.isfile(CONFIG.MODEL_PATH_KNN) and os.path.isfile(CONFIG.MODEL_PATH_SVC):
            self.knn_clf = joblib.load(CONFIG.MODEL_PATH_KNN)
            self.svc_clf = joblib.load(CONFIG.MODEL_PATH_SVC)    
            self._classes = self.knn_clf.classes_

    def __del__(self):
        self.stream.stop()

    def inference(self):
        
        frame_orig = self.stream.read()

        if self.knn_clf is not None and self.svc_clf is not None:
            # resize the frame to increase the speed
            frame = frame_orig
            
            # height, width = frame.shape[0], frame.shape[1]
            frame = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)

            if self.gpu:
                face_bboxes = face_recognition.face_locations(frame, model="cnn")
            else:
                face_bboxes = face_recognition.face_locations(frame)

            if len(face_bboxes) > 0:
                if self.gpu:
                    face_embeddings = face_recognition.face_encodings(frame, known_face_locations=face_bboxes, model="cnn")
                else:
                    face_embeddings = face_recognition.face_encodings(frame, known_face_locations=face_bboxes)

                knn_probs = self.knn_clf.predict_proba(face_embeddings)
                knn_preds = self.knn_clf.predict(face_embeddings)

                svc_probs = self.svc_clf.predict_proba(face_embeddings)
                svc_preds = self.svc_clf.predict(face_embeddings)

                # -----------------------------------------------------------------------------------------------
                # modified part (combining two classifiers together)
                predictions = []
                # iterating through predictions, probabalities from both classifiers and face bounding boxes 
                for pred_1, prob_1, pred_2, prob_2, box in zip(knn_preds, knn_probs, svc_preds, svc_probs, face_bboxes):
                    _prob_1 = prob_1[np.where(self._classes==pred_1)]
                    _prob_2 = prob_2[np.where(self._classes==pred_2)]

                    # taking average probability
                    prob_avg = (_prob_1 + _prob_2) / 2

                    # if both classifier has different predictions then consider it as unknown OR
                    # if average probabality is less than the given threshold
                    if pred_1 != pred_2 or prob_avg < self.threshold:
                        predictions.append(("unknown", box, prob_avg))
                    
                    # else classify as a known person
                    else:
                        predictions.append((pred_1, box, prob_avg))
                # -----------------------------------------------------------------------------------------------


                for name, (top, right, bottom, left), prob in predictions:
                    prob = round(prob[0] * 100, 5)
                    top, left = int(top / self.resize_scale), int(left / self.resize_scale)
                    bottom, right = int(bottom / self.resize_scale), int(right / self.resize_scale)

                    if name == "unknown":
                        # Crop the image frame into rectangle
                        cv2.rectangle(frame_orig, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame_orig, name + ": " + str(prob) + "%", (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    else:
                        # Crop the image frame into rectangle
                        cv2.rectangle(frame_orig, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame_orig, name + ": " + str(prob) + "%", (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            _str = "No Classifier was found..."
            cv2.putText(frame_orig, _str, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)            
        # frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        ret, jpeg = cv2.imencode('.jpg', frame_orig)
        data = []
        data.append(jpeg.tobytes())
        return data


def inference_webcam(camera):
    while True:
        data = camera.inference()
        frame = data[0]
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
