from flask import Flask, request, send_file, jsonify
import string
import os
from PIL import Image

app = Flask(__name__)

arr = list("abcdefghijklmnopqrstuvwxyz")

# Function to get letter images
def get_letter_images(text):
    print(f"Received text: {text}")  # Debugging

    images = []
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))

    print(f"Filtered text: {text}")  # Debugging

    valid_chars = [char for char in text if char in arr]

    print(f"Valid characters: {valid_chars}")  # Debugging

    if not valid_chars:
        return None

    for char in valid_chars:
        img_path = os.path.join(os.getcwd(), "convert", "letters", f"{char}.jpg")
        print(f"Checking image path: {img_path}")  # Debugging

        if os.path.exists(img_path):
            images.append(img_path)
        else:
            print(f"❌ Image NOT FOUND: {img_path}")  # Debugging

    return images

# Convert text to images
@app.route('/convert', methods=['POST'])
def convert_text_to_images():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]
        images = get_letter_images(text)

        if not images:
            return jsonify({"error": "No valid characters found"}), 400

        return jsonify({"images": images})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve letter images
@app.route('/image/<letter>', methods=['GET'])
def get_image(letter):
    letter = letter.lower()
    if letter not in arr:
        return jsonify({"error": "Invalid character"}), 400

    img_path = os.path.join(os.getcwd(), "convert", "letters", f"{letter}.jpg")
    if os.path.exists(img_path):
        return send_file(img_path, mimetype="image/jpeg")

    return jsonify({"error": "Image not found"}), 404

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
