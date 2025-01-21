import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# โหลดโมเดลที่บันทึกไว้
loaded_model = load_model('7Seg_model.h5')

# เส้นทางของภาพใหม่
image_path = r"C:\Users\Buranon\OneDrive\Desktop\Data\Testing image\Scaling\0\img (1).jpg"

# โหลดและประมวลผลภาพใหม่
img = load_img(image_path, target_size=(28, 28), color_mode='grayscale')  # โหลดภาพ
img_array = img_to_array(img) / 255.0  # แปลงเป็น array และ normalize
img_array = img_array.reshape(1, 28, 28, 1)  # เพิ่มมิติให้เข้ากับ input shape ของโมเดล

# ตรวจสอบข้อมูลภาพ
print(f"Image array shape: {img_array.shape}")
print(f"Min pixel value: {img_array.min()}, Max pixel value: {img_array.max()}")

# แสดงภาพที่ใช้สำหรับการพยากรณ์
plt.imshow(img_array[0].reshape(28, 28), cmap='gray')
plt.title("Input Image")
plt.show()

# พยากรณ์ด้วยโมเดลที่โหลด
prediction = loaded_model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)

# แสดงผลลัพธ์
print("Predicted probabilities:", prediction)
print(f"Predicted class from loaded model: {predicted_class[0]}")
