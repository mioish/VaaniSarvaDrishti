import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

handDetector = HandDetector(maxHands=2)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

offset = 20
imgSize = 300
folder = "Data/secret"
counter = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    hands, img = handDetector.findHands(img, draw=True)  

    results = face_mesh.process(imgRGB)

    face_center = None  

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, draw_spec, draw_spec)
 
            face_x = [int(pt.x * img.shape[1]) for pt in face_landmarks.landmark]
            face_y = [int(pt.y * img.shape[0]) for pt in face_landmarks.landmark]
            
            x_min, x_max = min(face_x), max(face_x)
            y_min, y_max = min(face_y), max(face_y)
            

            face_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            cv2.circle(img, face_center, 5, (0, 0, 255), -1)  
         
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    for i, hand in enumerate(hands):  
        x, y, w, h = hand["bbox"]
        
        hand_center = (x + w // 2, y + h // 2)
        cv2.circle(img, hand_center, 5, (0, 255, 255), -1) 

        if face_center:
            distance = math.sqrt((hand_center[0] - face_center[0]) ** 2 + (hand_center[1] - face_center[1]) ** 2)
            
            if distance > 150:  
                continue

        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize
        except Exception as e:
            print(f"Resize Error: {e}")
            continue

        cv2.imshow(f"Hand_{i+1}_Crop", imgCrop)
        #cv2.imshow(f"Hand_{i+1}_White", imgWhite)

        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            cv2.imwrite(f'{folder}/Hand_{i+1}_Img_{time.time()}.png', imgWhite)
            print(f"Saved Hand {i+1}: {counter}")

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
