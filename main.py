import speech_recognition as sr
import time
import string
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

arr = list("abcdefghijklmnopqrstuvwxyz")


def recognize_speech():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("\nListening... ")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio).lower()
            print(f"\nYou Said: {text}\n")

            text = text.translate(str.maketrans("", "", string.punctuation))

            if text == 'exit':
                print("Exiting... Goodbye! 👋")
                return False  
            
            else:
                show_letters(text)

        except sr.UnknownValueError:
            print("Sorry, Try again")
        except sr.RequestError:
            print("API Error! Internet Connection check karein.")
    
    return True  


def show_letters(text):
    images = []
    valid_chars = [char for char in text if char in arr]

    if not valid_chars:
        print("No Valid char found")
        return

    for char in valid_chars:
        img_path = f'convert/letters/{char}.jpg'
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
        else:
            print(f"Letter '{char.upper()}' Image not found.")

    if images:
        fig, axes = plt.subplots(1, len(images), figsize=(2*len(images), 3))
        
        if len(images) == 1:
            axes.imshow(images[0])
            axes.axis("off")
        else:
            for ax, img in zip(axes, images):
                ax.imshow(img)
                ax.axis("off")
        
        plt.show()
    else:
        print("Image not found")
'''
text = input("Enter Text: ").lower().strip()  
text = text.translate(str.maketrans("", "", string.punctuation))  
show_letters(text)  

'''
if __name__ == "__main__":
    while recognize_speech():
        pass
