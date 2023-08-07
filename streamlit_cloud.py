import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from keras.models import model_from_json

class AccidentDetectionModel(object):
    class_nums = ['Accident', "No Accident"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_accident(self, img):
        self.preds = self.loaded_model.predict(img)
        return AccidentDetectionModel.class_nums[np.argmax(self.preds)], self.preds

font = cv2.FONT_HERSHEY_SIMPLEX
model = AccidentDetectionModel("model.json", os.path.join(os.getcwd(), 'model_weights.h5'))

def startapplication(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))

    pred, prob = model.predict_accident(roi[np.newaxis, :, :])

    if pred == "Accident":
        prob = (round(prob[0][0]*100, 2))
        cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
        cv2.putText(frame, pred + " " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)

    return frame

def main():
    st.title('Accident Detection')
    st.write("""
    Upload a video and click on the "Execute" button to detect any accidents in the video.
    """)

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"], key="file_uploader")

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        cap.open(tfile.name)  # Open the video file

        if st.button('Execute', key="execute_button"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = startapplication(frame)
                frame_to_display = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_to_display, channels="RGB")
            cap.release()  # Release the video capture when processing is done

if __name__ == '__main__':
    main()
