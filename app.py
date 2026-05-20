from flask import Flask, render_template, request
from ultralytics import YOLO
import pandas as pd
import os
import base64

from PIL import Image
from io import BytesIO

app = Flask(__name__)

# YOLO 모델
model = YOLO("yolov8n.pt")

# GI 데이터
db = pd.read_csv("gi_db.csv")

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# 메인 페이지
@app.route("/")
def home():
    return render_template("index.html")


# 예측
@app.route("/predict", methods=["POST"])
def predict():

    filepath = "static/uploads/input.png"

    # =========================
    # 카메라 촬영 처리
    # =========================
    image_data = request.form.get("image")

    if image_data and image_data.startswith("data:image"):

        image_data = image_data.split(",")[1]

        image = Image.open(BytesIO(base64.b64decode(image_data)))

        image.save(filepath)

    # =========================
    # 파일 업로드 처리
    # =========================
    elif "file" in request.files:

        file = request.files["file"]

        if file.filename != "":

            file.save(filepath)

        else:
            return "<h1>파일이 선택되지 않았습니다.</h1>"

    else:
        return "<h1>이미지가 없습니다.</h1>"

    # =========================
    # YOLO 예측
    # =========================
    results = model.predict(filepath, conf=0.3)

    detected_food = "unknown"

    names = results[0].names

    for box in results[0].boxes:

        cls = int(box.cls[0])

        detected_food = names[cls]

        break

    # =========================
    # GI 검색
    # =========================
    row = db[db["food"] == detected_food]

    if len(row) > 0:

        gi = int(row.iloc[0]["gi"])

        if gi <= 55:
            risk = "안전"

        elif gi <= 69:
            risk = "보통"

        else:
            risk = "위험"

    else:

        gi = "정보 없음"

        risk = "판별 불가"

    # =========================
    # 결과 출력
    # =========================
    return f"""
    <h1>분석 결과</h1>

    <p>음식: {detected_food}</p>

    <p>GI 지수: {gi}</p>

    <p>위험도: {risk}</p>

    <img src='/{filepath}' width='300'>

    <br><br>

    <a href="/">다시 분석</a>
    """


# 실행
if __name__ == "__main__":
    app.run(debug=True)