import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# กำหนดเส้นทางของไฟล์ CSV
train_csv_path = r'C:\Users\Buranon\OneDrive\Desktop\Data\train_split.csv'
test_csv_path = r'C:\Users\Buranon\OneDrive\Desktop\Data\test_split.csv'

# โหลดข้อมูลการฝึก
train_data = pd.read_csv(train_csv_path)
test_data = pd.read_csv(test_csv_path)

def load_and_process_images(data, base_path):
    images = []
    labels = []
    for img_path, label in zip(data['file'], data['label']):
        corrected_path = img_path.replace('./dataset/train/', '').replace('./dataset/validation/', '')
        full_path = os.path.join(base_path, corrected_path)
        # print(f"Trying to load: {full_path}")  # Debugging
        try:
            img = load_img(full_path, target_size=(28, 28), color_mode='grayscale')  # โหลดภาพและปรับขนาด
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
        except FileNotFoundError:
            print(f"File not found: {full_path}")  # Debugging
            continue
    images = np.array(images).reshape((-1, 28, 28, 1)) / 255.0  # เพิ่มมิติ (N, 28, 28, 1)
    labels = np.array(labels)  # แปลง labels เป็น numpy array
    return images, labels

# สำหรับ training data
train_images, train_labels = load_and_process_images(train_data, r'C:\Users\Buranon\OneDrive\Desktop\Data\Training image')

# สำหรับ validation/test data
test_images, test_labels = load_and_process_images(test_data, r'C:\Users\Buranon\OneDrive\Desktop\Data\Training image')

# ตรวจสอบจำนวนภาพและป้ายกำกับ
print(f"Number of training images: {len(train_images)}")
print(f"Number of training labels: {len(train_labels)}")
print(f"Number of test images: {len(test_images)}")
print(f"Number of test labels: {len(test_labels)}")

# สร้างโมเดล CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ฝึกโมเดล
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# ประเมินโมเดล
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# พยากรณ์ผลลัพธ์
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)

# สร้าง confusion matrix
cm = confusion_matrix(test_labels, y_pred_classes)

# แสดงผล confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# คำนวณ precision, recall, f1 score
report = classification_report(test_labels, y_pred_classes, target_names=[str(i) for i in range(10)])
print("Classification Report:")
print(report)

# บันทึกโมเดล
model.save('7Seg_model_noise.h5')  # บันทึกในรูปแบบ HDF5
print("Model saved successfully!")
