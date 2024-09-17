from flask import Flask, request, jsonify
import cv2
import easyocr
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)
reader = easyocr.Reader(['en'])

def detect_number_plate(image_bytes):
    # Convert image bytes to numpy array
    image = Image.open(BytesIO(image_bytes))
    image_np = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        plate_image = image_np[y:y+h, x:x+w]
        result = reader.readtext(plate_image)

        for detection in result:
            plates.append(detection[1])
    
    return plates

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image_bytes = file.read()
        plates = detect_number_plate(image_bytes)
        return jsonify({'plates': plates}), 200

if __name__ == '__main__':
    app.run(debug=True)
