from ast import Return
from unittest import result
import cv2
import mediapipe as mp
class fingerDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.75, trackCon=0.75):
        self.mp_Hands = mp.solutions.hands # иницилизация hands
        self.hands = self.mp_Hands.Hands(mode, maxHands, complexity, detectionCon, trackCon) # характирисетики для распознования hands
        self.mpDraw = mp.solutions.drawing_utils # иницилизация утилит для рисования hands
        self.fingertips = [4, 8, 12, 16, 20] # кончики fingers

    def findHands(self, img, draw=True):
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(RGB_image)
        if draw:
            for handlms in self.result.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpDraw.HAND_CONNECTIONS)

        return img