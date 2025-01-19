# import cv2
# import mediapipe as mp

# class PoseDetector:
#     def __init__(self, mode=False, complexity=1, smooth=True, detectionCon=0.5, trackCon=0.5):
#         """
#         Initializes the PoseDetector with the given parameters.
#         """
#         # Ensure proper argument types
#         assert isinstance(mode, bool), "Mode should be a boolean."
#         assert isinstance(complexity, int) and 0 <= complexity <= 2, "Complexity should be an integer (0, 1, or 2)."
#         assert isinstance(smooth, bool), "Smooth should be a boolean."
#         assert isinstance(detectionCon, float) and 0 <= detectionCon <= 1, "Detection confidence should be a float between 0 and 1."
#         assert isinstance(trackCon, float) and 0 <= trackCon <= 1, "Tracking confidence should be a float between 0 and 1."
        
#         self.mode = mode
#         self.complexity = complexity
#         self.smooth = smooth
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon

#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(
#             static_image_mode=self.mode,
#             model_complexity=self.complexity,
#             smooth_landmarks=self.smooth,
#             min_detection_confidence=self.detectionCon,
#             min_tracking_confidence=self.trackCon,
#         )
#         self.mpDraw = mp.solutions.drawing_utils

#     def findPose(self, img, draw=True):
#         """
#         Detects the pose in an image and optionally draws landmarks.
#         """
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)
#         if self.results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
#         return img

#     def findPosition(self, img, draw=True):
#         """
#         Finds the position of landmarks and returns a list of them.
#         """
#         lmList = []
#         if self.results.pose_landmarks:
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#         return lmList


import cv2
import mediapipe as mp
import math

class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = bool(mode)
        self.upBody = bool(upBody)
        self.smooth = bool(smooth)
        self.detectionCon = float(detectionCon)
        self.trackCon = float(trackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode, 
            model_complexity=1,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)  # Red points
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        x3, y3 = self.lmList[p3][1:3]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - 
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 165, 255), 14)  # Orange line, thickness 6
            cv2.line(img, (x3, y3), (x2, y2), (0, 165, 255), 14)  # Orange line, thickness 6
            cv2.circle(img, (x1, y1), 20, (0, 0, 255), cv2.FILLED)  # Red point
            cv2.circle(img, (x2, y2), 20, (0, 0, 255), cv2.FILLED)  # Red point
            cv2.circle(img, (x3, y3), 20, (0, 0, 255), cv2.FILLED)  # Red point
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)  # Text color red
        return angle