from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO("yolov8n.pt")

# 웹캠 실행
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO 예측
    results = model.predict(frame, conf=0.4)

    # 결과 이미지 생성
    annotated_frame = results[0].plot()

    # 화면 출력
    cv2.imshow("Food Detection", annotated_frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()