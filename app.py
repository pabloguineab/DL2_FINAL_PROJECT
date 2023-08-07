import streamlit as st
import cv2
import tempfile
from camera import startapplication, font

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
            # Process the frame using the existing startapplication function
            processed_frame, pred, prob = startapplication(frame)
            if pred == "Accident":
                prob = (round(prob[0][0]*100, 2))
                cv2.rectangle(processed_frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(processed_frame, pred + " " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)
            # Convert the BGR image into RGB
            frame_to_display = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_to_display, channels="RGB")
        cap.release()  # Release the video capture when processing is done