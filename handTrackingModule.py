import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detection=0.5, track=0.5):
        self.mp_Hands = mp.solutions.hands # хотим распозновать hands
        self.hands = self.mp_Hands.Hands(mode, maxHands, modelComplexity, detection, track) # характирисетики для распознования hands
        self.mpDraw = mp.solutions.drawing_utils # иницилизация утилит для рисования hands

    def findHands(self, img):
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(RGB_image)

    def findPositionPoints(self):
        pass
