from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Function to perform instance segmentation
def perform_instance_segmentation(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to the grayscale image
    _, instance_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply colormap to the instance image for visualization
    colored_instance_image = cv2.applyColorMap(instance_image, cv2.COLORMAP_VIRIDIS)

    return gray, colored_instance_image

# Route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and perform instance segmentation
@app.route('/segment', methods=['POST'])
def segment_image():
    # Get the image file from the POST request
    file = request.files['file']
    
    # Read the image file
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Perform instance segmentation on the uploaded image
    gray_image, colored_instance_image = perform_instance_segmentation(image)
    
    # Convert the images to base64 encoding
    _, gray_buffer = cv2.imencode('.jpg', gray_image)
    gray_image_base64 = base64.b64encode(gray_buffer).decode('utf-8')

    _, colored_buffer = cv2.imencode('.jpg', colored_instance_image)
    colored_image_base64 = base64.b64encode(colored_buffer).decode('utf-8')

    # Return the segmented images as JSON response
    return jsonify({'gray_image': gray_image_base64, 'colored_instance_image': colored_image_base64})

if __name__ == '__main__':
    app.run(debug=True)
