import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import imageio
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300
dataset_path = "Data/Thankyou"

gesture_name = input("Enter gesture name: ")  
gesture_folder = os.path.join(dataset_path, gesture_name)
os.makedirs(gesture_folder, exist_ok=True)

frames = []
sequence_length = 30  

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    for i, hand in enumerate(hands):
        x, y, w, h = hand["bbox"]

        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        cv2.imshow(f"Hand_{i+1}_White", imgWhite)

        key = cv2.waitKey(1)

        if key == ord("s"):
            for j in range(sequence_length):
                frames.append(imgWhite)
                time.sleep(0.05)  
            
            gif_path = os.path.join(gesture_folder, f"{gesture_name}.gif")
            imageio.mimsave(gif_path, frames, duration=0.05)
            print(f"Saved GIF for {gesture_name} at {gif_path}")

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
