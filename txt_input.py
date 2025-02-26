import string
import os
import matplotlib.pyplot as plt
from PIL import Image

arr = list("abcdefghijklmnopqrstuvwxyz")

def show_letters(text):
    images = []
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    valid_chars = [char for char in text if char in arr]

    if not valid_chars:
        return

    for char in valid_chars:
        img_path = f'convert/letters/{char}.jpg'
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)

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

if __name__ == "__main__":
    while True:
        text = input("Enter Text (or type 'exit' to quit): ").strip()
        if text.lower() == "exit":
            print("Exiting...")
            break
        show_letters(text)
