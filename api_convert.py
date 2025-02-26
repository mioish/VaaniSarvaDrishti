import speech_recognition as sr

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("\nListening...")
    recognizer.adjust_for_ambient_noise(source, duration=2)  # Noise adjust karne ke liye 2 sec ka buffer
    audio = recognizer.listen(source)
    print("\nCaptured audio!")
import speech_recognition as sr

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Index {index}: {name}")
mic_index = 3  # Yeh index tumhare system ka sahi mic hoga

recognizer = sr.Recognizer()
with sr.Microphone(device_index=mic_index) as source:
    print("\nListening...")
    recognizer.adjust_for_ambient_noise(source, duration=2)
    audio = recognizer.listen(source)
    print("\nCaptured audio!")
