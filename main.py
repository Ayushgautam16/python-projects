import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np
cap = cv2.VideoCapture(0)  # Change to '0' to use the default webcam
cap.set(3, 1280)  # Width of the frame
cap.set(4, 720)   # Height of the frame
detector = HandDetector(detectionCon=0.8)  # Hand detection confidence
colorR = (255, 0, 255)  # Color for rectangle (Pink)
