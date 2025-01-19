import streamlit as st
import cv2
import numpy as np
import tempfile
from pose_module import PoseDetector
import winsound  # For beep sounds on Windows

# Streamlit configuration
st.set_page_config(page_title="Bicep Curl Counter", layout="wide")

st.markdown(
    """
    <style>
    body { background-color: #f4f4f4; }
    .header { text-align: center; font-size: 36px; font-weight: bold; margin-top: 20px; color: #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header">AI Bicep Curl Tracker</div>', unsafe_allow_html=True)

# Sidebar options
with st.sidebar:
    mode = st.selectbox("Choose Mode", ["Live Tracking", "Demo Video", "Upload Video"])

# Initialize Pose Detector
detector = PoseDetector()

# Initialize session state for count and direction
if "count" not in st.session_state:
    st.session_state.count = 0
if "dir" not in st.session_state:
    st.session_state.dir = 0

# Smoothing variables
smooth_bar = 650  # Initialize the smooth bar position
smooth_percentage = 0  # Initialize the smooth percentage value
SMOOTHING_FACTOR = 0.3  # Adjust smoothing factor for faster response

# Reset button functionality
if st.sidebar.button("Reset Count"):
    st.session_state.count = 0
    st.session_state.dir = 0


def process_frame(frame, detector, count, dir, smooth_bar, smooth_percentage):
    # Ensure that the frame is in BGR format
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR if necessary

    frame = detector.findPose(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        # Calculate the angle between arm joints
        angle = detector.findAngle(frame, 12, 14, 16)
        target_percentage = np.interp(angle, (210, 310), (0, 100))
        target_bar = np.interp(angle, (220, 310), (650, 100))

        # Smoothly interpolate the bar position and percentage with faster response
        smooth_bar = smooth_bar * (1 - SMOOTHING_FACTOR) + target_bar * SMOOTHING_FACTOR
        smooth_percentage = smooth_percentage * (1 - SMOOTHING_FACTOR) + target_percentage * SMOOTHING_FACTOR

        # Determine color and update count
        color = (255, 0, 255)
        if target_percentage == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        elif target_percentage == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        # Beep functionality for 30% to 40%
        if 30 <= target_percentage <= 40:
            winsound.Beep(1000, 200)  # Frequency: 1000Hz, Duration: 200ms

        # Draw progress bar
        cv2.rectangle(frame, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(frame, (1100, int(smooth_bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(frame, f'{int(smooth_percentage)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Display the counter
        cv2.rectangle(frame, (20, 550), (180, 700), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, str(int(count)), (50, 680), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

        # Draw red circles on arm points
        for i in [12, 14, 16]:  # Shoulder, elbow, wrist
            cv2.circle(frame, (lmList[i][1], lmList[i][2]), 10, (0, 0, 255), cv2.FILLED)

    return frame, count, dir, smooth_bar, smooth_percentage


if mode == "Live Tracking":
    cap = cv2.VideoCapture(0)  # Use the default webcam (index 0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Unable to access webcam.")
            break

        frame = cv2.resize(frame, (1280, 720))  # Resize for consistency
        frame, st.session_state.count, st.session_state.dir, smooth_bar, smooth_percentage = process_frame(
            frame, detector, st.session_state.count, st.session_state.dir, smooth_bar, smooth_percentage
        )
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

elif mode == "Demo Video":
    demo_video_path = "demo.mp4"  # Path to the demo video
    cap = cv2.VideoCapture(demo_video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        frame, st.session_state.count, st.session_state.dir, smooth_bar, smooth_percentage = process_frame(
            frame, detector, st.session_state.count, st.session_state.dir, smooth_bar, smooth_percentage
        )
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

elif mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            frame, st.session_state.count, st.session_state.dir, smooth_bar, smooth_percentage = process_frame(
                frame, detector, st.session_state.count, st.session_state.dir, smooth_bar, smooth_percentage
            )
            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
