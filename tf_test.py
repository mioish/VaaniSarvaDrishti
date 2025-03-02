from flask import Flask, jsonify, request
import cv2
import numpy as np
import threading
import pyttsx3
import time
from cvzone.HandTrackingModule import HandDetector
from tensorflow.lite.python.interpreter import Interpreter
from collections import deque

app = Flask(__name__)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

# Open webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.6)  # Lower confidence to detect shaky hands

# Load TensorFlow Lite Model
interpreter = Interpreter(model_path="c:\\flutter\\VaaniSarvaDrishti-main\\model\\model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "Hi", "I", "K", "L", "Love", "M", "O", "P", "S", "T", "V", "W", "Z"]

# Parameters
offset = 30  # Increased offset for tremors
imgSize = input_shape[1]
base_confidence_threshold = 0.6  # Lowered threshold for Parkinson’s patients

# Tremor stabilization using history
prediction_history = deque(maxlen=7)  # Longer history window
confidence_history = deque(maxlen=7)

last_prediction = None
last_time = time.time()

def process_frame():
    global last_prediction, last_time
    success, img = cap.read()
    if not success:
        return None

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=True)

    if hands:
        x_min, y_min, x_max, y_max = float("inf"), float("inf"), 0, 0

        for hand in hands:
            x, y, w, h = hand["bbox"]
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)

        # Ensure bounding box is within image dimensions
        x1, y1 = max(0, x_min - offset), max(0, y_min - offset)
        x2, y2 = min(img.shape[1], x_max + offset), min(img.shape[0], y_max + offset)

        if x1 >= x2 or y1 >= y2:
            print("Invalid crop dimensions, skipping frame")
            return None

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            print("Warning: Empty cropped image, skipping frame")
            return None

        imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
        imgEqualized = cv2.equalizeHist(imgBlur)
        imgWhite = cv2.cvtColor(imgEqualized, cv2.COLOR_GRAY2BGR)

        imgWhite = cv2.resize(imgWhite, (input_shape[1], input_shape[2]))
        imgWhite = np.expand_dims(imgWhite, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], imgWhite)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        index = np.argmax(prediction)
        confidence = prediction[index]

        prediction_history.append(index)
        confidence_history.append(confidence)

        most_common_prediction = max(set(prediction_history), key=prediction_history.count)
        avg_confidence = np.mean(confidence_history)

        # Adaptive confidence threshold for unstable movements
        confidence_threshold = base_confidence_threshold + (0.1 if len(set(prediction_history)) > 3 else 0)

        if avg_confidence > confidence_threshold:
            predicted_text = labels[most_common_prediction]

            if predicted_text != last_prediction or time.time() - last_time > 2:
                print(f"Predicted Sign: {predicted_text} (Confidence: {avg_confidence:.2f})")
                speak(predicted_text)
                last_prediction = predicted_text
                last_time = time.time()
                return predicted_text
        else:
            print(f"Low confidence: {avg_confidence:.2f}")

    return None

@app.route('/predict', methods=['GET'])
def predict():
    result = process_frame()
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)