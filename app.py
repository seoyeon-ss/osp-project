from flask import Flask, render_template, request
from ultralytics import YOLO
import pandas as pd
import os
import base64

from PIL import Image
from io import BytesIO

app = Flask(__name__)

# YOLO 모델
model = YOLO("yolo12n.pt")

# 음식별 GI/영양소 데이터
# CSV 컬럼: food, display_name, category, calories_kcal, carbohydrate_g, fat_g,
# protein_g, sugar_g, fiber_g, gi, alternative_food
nutrition_db = pd.read_csv("gi_db.csv").fillna("")
nutrition_db["food"] = nutrition_db["food"].astype(str).str.lower()

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

LOW_GI_MAX = 55
MEDIUM_GI_MAX = 69
HIGH_GI_MIN = 70


def get_food_row(food_name):
    """YOLO가 인식한 음식 이름과 CSV의 food 값을 비교해 해당 음식 정보를 찾습니다."""
    normalized_food_name = str(food_name).lower()
    matched_rows = nutrition_db[nutrition_db["food"] == normalized_food_name]

    if len(matched_rows) == 0:
        return None

    return matched_rows.iloc[0]


def get_gi_risk(gi_value):
    """GI 지수를 낮음/보통/높음 등급으로 바꿉니다."""
    if gi_value <= LOW_GI_MAX:
        return "낮음", "혈당 상승 위험이 비교적 낮은 음식입니다."

    if gi_value <= MEDIUM_GI_MAX:
        return "보통", "섭취량을 조절하면 비교적 무난한 음식입니다."

    return "높음", "혈당을 빠르게 올릴 수 있어 대체 음식을 추천합니다."


def make_macro_ratio(food_row):
    """탄수화물/지방/단백질 비율을 계산합니다."""
    carbohydrate = float(food_row["carbohydrate_g"])
    fat = float(food_row["fat_g"])
    protein = float(food_row["protein_g"])
    total_macro = carbohydrate + fat + protein

    if total_macro == 0:
        return {
            "carbohydrate": 0,
            "fat": 0,
            "protein": 0,
        }

    return {
        "carbohydrate": round(carbohydrate / total_macro * 100, 1),
        "fat": round(fat / total_macro * 100, 1),
        "protein": round(protein / total_macro * 100, 1),
    }


def make_nutrition_info(food_row):
    """화면 출력에 필요한 영양 정보를 딕셔너리로 정리합니다."""
    return {
        "food": food_row["food"],
        "display_name": food_row["display_name"],
        "category": food_row["category"],
        "calories_kcal": float(food_row["calories_kcal"]),
        "carbohydrate_g": float(food_row["carbohydrate_g"]),
        "fat_g": float(food_row["fat_g"]),
        "protein_g": float(food_row["protein_g"]),
        "sugar_g": float(food_row["sugar_g"]),
        "fiber_g": float(food_row["fiber_g"]),
        "gi": int(food_row["gi"]),
        "macro_ratio": make_macro_ratio(food_row),
    }


def find_low_gi_recommendation(food_row):
    """GI가 높은 음식이면 CSV의 대체 음식 또는 같은 카테고리의 낮은 GI 음식을 추천합니다."""
    gi_value = int(food_row["gi"])

    if gi_value < HIGH_GI_MIN:
        return None

    alternative_food = str(food_row["alternative_food"]).strip().lower()

    if alternative_food != "":
        alternative_rows = nutrition_db[nutrition_db["food"] == alternative_food]

        if len(alternative_rows) > 0:
            return make_nutrition_info(alternative_rows.iloc[0])

    same_category_low_gi_foods = nutrition_db[
        (nutrition_db["category"] == food_row["category"])
        & (nutrition_db["gi"] <= LOW_GI_MAX)
        & (nutrition_db["food"] != food_row["food"])
    ].sort_values(by="gi")

    if len(same_category_low_gi_foods) == 0:
        return None

    return make_nutrition_info(same_category_low_gi_foods.iloc[0])


# 메인 페이지
@app.route("/")
def home():
    return render_template("index.html", result=None)


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
            return render_template(
                "index.html",
                result={"error": "파일이 선택되지 않았습니다."},
            )
    else:
        return render_template(
            "index.html",
            result={"error": "이미지가 없습니다."},
        )

    # =========================
    # YOLO Classification 예측
    # =========================
    # classification 모델은 detection 모델처럼 boxes를 만들지 않습니다.
    # 이미지 전체를 하나의 음식 class로 분류하므로 probs.top1 값을 사용합니다.
    results = model.predict(filepath)
    prediction = results[0]

    if prediction.probs is None:
        return render_template(
            "index.html",
            result={
                "image_path": filepath,
                "error": "현재 모델이 classification 결과를 반환하지 않습니다. -cls 또는 classify로 학습한 best.pt 모델을 사용하세요.",
            },
        )

    top1_index = int(prediction.probs.top1)
    detected_food = prediction.names[top1_index]

    # =========================
    # 영양/GI 검색 및 추천
    # =========================
    food_row = get_food_row(detected_food)

    if food_row is None:
        result = {
            "detected_food": detected_food,
            "image_path": filepath,
            "error": "인식된 음식의 영양 정보가 CSV에 없습니다.",
        }
    else:
        nutrition_info = make_nutrition_info(food_row)
        risk_label, risk_description = get_gi_risk(nutrition_info["gi"])
        recommendation = find_low_gi_recommendation(food_row)

        result = {
            "detected_food": detected_food,
            "image_path": filepath,
            "nutrition": nutrition_info,
            "risk_label": risk_label,
            "risk_description": risk_description,
            "recommendation": recommendation,
        }

    # =========================
    # 결과 출력
    # =========================
    return render_template("index.html", result=result)


# 실행
if __name__ == "__main__":
    app.run(debug=True)
