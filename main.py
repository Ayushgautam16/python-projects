import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)  # Change to '0' to use the default webcam
cap.set(3, 1280)  # Width of the frame
cap.set(4, 720)   # Height of the frame
detector = HandDetector(detectionCon=0.8)  # Hand detection confidence
colorR = (255, 0, 255)  # Color for rectangle (Pink)

# Define the DragRect class
class DragRect:
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # Check if the index finger tip is inside the rectangle
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

# Create a list of draggable rectangles
rectList = [DragRect([x * 250 + 150, 150]) for x in range(5)]

# Main loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally
    hands, img = detector.findHands(img)  # Find hands in the image

    if hands:
        lmList = hands[0]['lmList']  # Get list of landmarks for the first hand
        # Extract the (x, y) coordinates of the index and middle finger tips
        index_finger_tip = lmList[8][:2]  # Index finger tip landmark
        middle_finger_tip = lmList[12][:2]  # Middle finger tip landmark

        l, _, _ = detector.findDistance(index_finger_tip, middle_finger_tip, img)  # Distance between index and middle finger

        if l < 30:  # If distance is small enough, consider as click
            cursor = index_finger_tip  # Use the index finger tip for dragging
            for rect in rectList:
                rect.update(cursor)

    # Draw transparent rectangles
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5  # Transparency factor
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)  # Display the output image
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
