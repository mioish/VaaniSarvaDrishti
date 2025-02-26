import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300
folder = "Data/Love"
counter = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    if len(hands) == 1:
       
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        
    elif len(hands) == 2:
    
        x1, y1, w1, h1 = hands[0]["bbox"]
        x2, y2, w2, h2 = hands[1]["bbox"]
        
        x = min(x1, x2)
        y = min(y1, y2)
        w = max(x1 + w1, x2 + w2) - x
        h = max(y1 + h1, y2 + h2) - y

    else:
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord("q"):
            break
        continue  

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
            print("Shape mismatch in width resize")

    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)

        try:
            imgWhite[hGap:hGap + hCal, :] = imgResize
        except ValueError:
            print("Shape mismatch in height resize")


    cv2.imshow("Hand_Crop", imgCrop)
    cv2.imshow("Hand_White", imgWhite)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Hand_Img_{time.time()}.png', imgWhite)
        print(f"Saved Hand Image: {counter}")

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
