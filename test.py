import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import threading  
import pyttsx3  

engine = pyttsx3.init()
engine.setProperty('rate', 150)  

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2) 
classifier = Classifier("detection/model/keras_model.h5", "detection/model/labels.txt")

labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, "A", "B", "C", "D", "E", "hi"]
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img, draw=False)  

    for i, hand in enumerate(hands):  
        x, y, w, h = hand["bbox"]
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)

            try:
                imgWhite[:, wGap:wGap + wCal] = imgResize
            except ValueError:
                continue  

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)

            try:
                imgWhite[hGap:hGap + hCal, :] = imgResize
            except ValueError:
                continue  

        prediction, index = classifier.getPrediction(imgWhite)
        predicted_text = str(labels[index])  
        print(f"Hand {i+1} Predicted Sign:", predicted_text)

    
        speak(predicted_text)

        cv2.imshow(f"Hand_{i+1}_Crop", imgCrop)
        cv2.imshow(f"Hand_{i+1}_White", imgWhite)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()