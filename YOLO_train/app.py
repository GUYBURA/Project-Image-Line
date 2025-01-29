from flask import Flask, request, jsonify
import torch
import os

app = Flask(__name__)

# Ensure uploads directory exists
uploads_dir = 'C:/Users/Buranon/OneDrive/Desktop/Test_Roboflow/YOLO_train/uploads'
os.makedirs(uploads_dir, exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('./yolov5', 'custom', path='C:/Users/Buranon/OneDrive/Desktop/Test_Roboflow/YOLO_train/thebest.pt', source='local')

@app.route('/')
def home():
    return "YOLOv5 API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image_path = os.path.join(uploads_dir, image.filename)
    image.save(image_path)

    # Perform object detection
    results = model(image_path)
    predictions = results.pandas().xyxy[0].to_dict(orient="records")  # Convert to JSON-compatible format

    # Delete the uploaded image
    os.remove(image_path)

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
