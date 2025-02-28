import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow warnings
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
detector = HandDetector(maxHands=2, detectionCon=0.8)  
classifier = Classifier("detection/model/keras_model.h5", "detection/model/labels.txt")

labels = ["A", "B", "C", "D", "E", "F", "H", "Hi", "Love", "I", "O", "1", "2", "3", "M"]
offset = 20
imgSize = 300
confidence_threshold = 0.7 

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1) 
    hands, img = detector.findHands(img, draw=True)

    if hands:   
        x_min, y_min, x_max, y_max = float("inf"), float("inf"), 0, 0

        for hand in hands:
            x, y, w, h = hand["bbox"]
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)

        x1, y1 = max(0, x_min - offset), max(0, y_min - offset)
        x2, y2 = min(img.shape[1], x_max + offset), min(img.shape[0], y_max + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = (y2 - y1) / (x2 - x1)
        if aspectRatio > 1:
            k = imgSize / (y2 - y1)
            wCal = math.ceil(k * (x2 - x1))
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / (x2 - x1)
            hCal = math.ceil(k * (y2 - y1))
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
        imgWhite = cv2.cvtColor(imgBlur, cv2.COLOR_GRAY2BGR)

        prediction, index = classifier.getPrediction(imgWhite)
        confidence = max(prediction)

        if confidence > confidence_threshold:
            predicted_text = labels[index]
            print(f"Predicted Sign: {predicted_text} (Confidence: {confidence:.2f})")
            speak(predicted_text)
        else:
            predicted_text = "Uncertain"
            print(f"Prediction confidence too low: {confidence:.2f}")

        cv2.imshow("Processed Sign", imgWhite)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

