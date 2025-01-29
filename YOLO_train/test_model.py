from ultralytics import YOLO

# โหลดโมเดล
model = YOLO(r"C:\Users\Buranon\OneDrive\Desktop\Test Roboflow\YOLO_train\thebest.pt")

# ทดสอบกับภาพเดี่ยว
results = model.predict(source=r"C:\Users\Buranon\OneDrive\Desktop\Line Server Prepro\image\resized_image.jpg", conf=0.5, save=True)

# หรือทดสอบกับวิดีโอ
# results = model.predict(source='/path/to/video.mp4', conf=0.5, save=True)
