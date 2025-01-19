import streamlit as st
import cv2
from pose_module import PoseDetector

# Upload Video Page
st.title("Upload Video for Pose Detection")

video_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"])

if video_file is not None:
    st.video(video_file)
    # Process the uploaded video file
    video_path = "/tmp/uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    # Process the video using OpenCV for pose detection
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        frame = detector.findPose(frame)
        lmList = detector.findPosition(frame, draw=False)

        if len(lmList) != 0:
            # Handle pose detection logic here

        # Display processed video frame
            st.image(frame, channels="BGR")
    
    cap.release()
