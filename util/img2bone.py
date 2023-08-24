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
        
    def findHands(self, img_url, draw = True):
        img = cv2.imread(img_url)
        img_bone = np.ones((img.shape[0], img.shape[1], 3)) * 255
        
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)
        all_hands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                zList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
                    zList.append(pz)
                ## bbox
                x_min, x_max = min(xList), max(xList)
                y_min, y_max = min(yList), max(yList)
                boxW, boxH = x_max - x_min, y_max - y_min
                bbox = x_min, y_min, boxW, boxH

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["xList"] = xList
                myHand["yList"] = yList
                myHand["zList"] = zList
                all_hands.append(myHand)
                # draw
                if draw:
                    self.mpDraw.draw_landmarks(img_bone, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
       
       
        # # Vẽ các keypoint lên hình ảnh
        # for point in mylmList:
        #     x, y,z = point
        #     cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Vẽ một hình tròn màu đỏ tại tọa độ (x, y)

        # cv2.imwrite("demo.jpg",img)
        if draw:
            return all_hands, img_bone
        else:
            return all_hands