import streamlit as st
import cv2
from pose_module import PoseDetector

# Initialize pose detector
detector = PoseDetector()

# Create camera capture object
cap = cv2.VideoCapture(0)

st.title("Live Pose Detection")

if cap.isOpened():
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        # Process the frame for pose detection
        frame = cv2.resize(frame, (1280, 720))
        frame = detector.findPose(frame)
        lmList = detector.findPosition(frame, draw=False)

        if len(lmList) != 0:
            # Handle pose detection output here

        # Show processed frame
            stframe.image(frame, channels="BGR")

    cap.release()
else:
    st.write("Unable to access the camera.")
