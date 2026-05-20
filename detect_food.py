from ultralytics import YOLO
import cv2

# YOLO 모델 불러오기
model = YOLO("yolov8n.pt")

# 이미지 예측
results = model.predict(
    source="food.jpg",
    conf=0.3,
    save=True
)

print("음식 인식 완료")