import cv2 
import requests
from PIL import Image, ImageDraw

# Load the image
image_path = r"C:\Users\Buranon\OneDrive\Desktop\Dataset for training\2)Night\92.JPG"
img = Image.open(image_path)

# API URL
url = "http://127.0.0.1:5000/predict"

# Send the image to the API
with open(image_path, "rb") as f:
    response = requests.post(url, files={"image": f})
    predictions = response.json()["predictions"]

# Filter predictions based on confidence threshold
confidence_threshold = 0.5
filtered_predictions = [pred for pred in predictions if pred["confidence"] > confidence_threshold]

# Sort predictions by xmin (left-to-right order)
sorted_predictions = sorted(filtered_predictions, key=lambda x: x["xmin"])

# Draw bounding boxes and collect predicted numbers
draw = ImageDraw.Draw(img)
predicted_numbers = []
for pred in sorted_predictions:
    xmin, ymin, xmax, ymax = pred["xmin"], pred["ymin"], pred["xmax"], pred["ymax"]
    name = pred["name"]  # Predicted number
    confidence = pred["confidence"]

    # Draw bounding box and text
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    draw.text((xmin, ymin - 10), f"{name} ({confidence:.2f})", fill="red")

    # Add the number to the list
    predicted_numbers.append(name)

# Combine predicted numbers into a single string
predicted_value = "".join(predicted_numbers)

# Save or show the result
img.show()
img.save("result_with_sorted_numbers.jpg")

# Print the sorted predicted value
print("Predicted Value:", predicted_value)
