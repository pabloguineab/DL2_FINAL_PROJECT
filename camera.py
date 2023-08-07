import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX
def startapplication(frame=None):
    if frame is None:
        video = cv2.VideoCapture('cars.mp4')
        ret, frame = video.read()
    else:
        ret = True

    # rest of the code remains the same

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))
    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    
    return frame, pred, prob

if __name__ == '__main__':
    startapplication()