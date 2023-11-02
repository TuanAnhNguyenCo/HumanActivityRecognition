import os
import cv2
import numpy as np
import math
import mediapipe as mp
from matplotlib import pyplot as plt


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class HandDetector:
    def __init__(self, mode=VisionRunningMode.IMAGE, maxHands=1, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img):
        """
        class video_idx frame_idx and one video has 64 frames
        """
        # img = cv2.imread(img_url)
        img_bone = np.ones((img.shape[0], img.shape[1], 3)) * 255

        # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_RGB = img
        self.results = self.hands.process(img_RGB)
        all_hands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                # lmList
                mylmList = []
                for id, lm in enumerate(handLms.landmark):
                    # mylmList.extend([lm.x, lm.y, lm.z])
                    mylmList.append([lm.x, lm.y, lm.z])
            return mylmList

        return None
