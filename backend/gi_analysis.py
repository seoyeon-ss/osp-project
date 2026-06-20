from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# =========================================================
# 1. GI 및 영양정보 데이터 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
GI_DB_PATH = BASE_DIR / "gi_db.csv"

# YOLO의 클래스 이름과 CSV의 food 값이 다를 때 연결합니다.
# normalize_food_name()을 거친 이름을 왼쪽에 작성합니다.
FOOD_NAME_ALIASES = {
    "white_rice": "rice",
    "steamed_rice": "rice",
    "red_apple": "apple",
    "hamburger": "burger",
    "green_salad": "salad",
}

REQUIRED_DB_COLUMNS = {
    "food",
    "display_name",
    "category",
    "calories_kcal",
    "carbohydrate_g",
    "fat_g",
    "protein_g",
    "sugar_g",
    "fiber_g",
    "gi",
    "alternative_food",
}

NUMERIC_COLUMNS = [
    "calories_kcal",
    "carbohydrate_g",
    "fat_g",
    "protein_g",
    "sugar_g",
    "fiber_g",
    "gi",
]

_food_db: pd.DataFrame | None = None


# =========================================================
# 2. CSV 데이터 조회
# =========================================================

def normalize_food_name(food_name: Any) -> str:
    """YOLO 음식명과 CSV 음식명을 비교하기 쉬운 형태로 통일합니다."""
    if food_name is None:
        return ""

    normalized = str(food_name).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def load_food_database() -> pd.DataFrame:
    """GI 및 영양정보 CSV를 읽고 필요한 자료형으로 정리합니다."""
    if not GI_DB_PATH.exists():
        raise FileNotFoundError(f"GI 데이터 파일을 찾을 수 없습니다: {GI_DB_PATH}")

    df = pd.read_csv(GI_DB_PATH, encoding="utf-8-sig")
    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = REQUIRED_DB_COLUMNS - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"GI 데이터 파일에 다음 열이 없습니다: {missing_text}")

    df = df.copy()
    df["food"] = df["food"].apply(normalize_food_name)
    df["alternative_food"] = df["alternative_food"].apply(
        lambda value: None
        if pd.isna(value) or str(value).strip() == ""
        else normalize_food_name(value)
    )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["food"].duplicated().any():
        duplicated = df.loc[df["food"].duplicated(), "food"].tolist()
        raise ValueError(f"GI 데이터 파일에 중복 음식명이 있습니다: {duplicated}")

    return df


def get_food_database() -> pd.DataFrame:
    """CSV를 최초 한 번만 읽고 이후에는 저장된 데이터프레임을 사용합니다."""
    global _food_db

    if _food_db is None:
        _food_db = load_food_database()

    return _food_db


def reload_food_database() -> None:
    """실행 중 gi_db.csv를 수정했을 때 데이터를 다시 불러옵니다."""
    global _food_db
    _food_db = load_food_database()


def to_python_value(value: Any) -> Any:
    """Pandas와 NumPy 값을 일반 Python 자료형으로 변환합니다."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    return value


def row_to_dict(row: pd.Series) -> dict[str, Any]:
    """CSV의 한 행을 HTML에서 사용하기 쉬운 딕셔너리로 변환합니다."""
    return {key: to_python_value(value) for key, value in row.to_dict().items()}


def find_food(food_name: str) -> dict[str, Any] | None:
    """YOLO가 인식한 음식명을 CSV의 food 값과 매칭합니다."""
    normalized_name = normalize_food_name(food_name)
    csv_food_name = FOOD_NAME_ALIASES.get(normalized_name, normalized_name)

    db = get_food_database()
    matched_rows = db[db["food"] == csv_food_name]

    if matched_rows.empty:
        return None

    return row_to_dict(matched_rows.iloc[0])


# =========================================================
# 3. GI 위험도, 영양비율, 대체음식 분석
# =========================================================

def get_gi_risk(gi: float) -> tuple[str, str]:
    """개별 음식의 GI 위험도를 낮음, 보통, 높음으로 분류합니다."""
    if gi <= 55:
        return "낮음", "GI 지수가 낮아 혈당 상승이 비교적 완만한 음식입니다."

    if gi <= 69:
        return "보통", "GI 지수가 중간 수준이므로 섭취량을 함께 조절하는 것이 좋습니다."

    return "높음", "GI 지수가 높아 혈당을 빠르게 올릴 수 있는 음식입니다."


def calculate_macro_ratio(food: dict[str, Any]) -> dict[str, float]:
    """
    탄수화물과 단백질은 1g당 4kcal, 지방은 1g당 9kcal로 계산하여
    총 에너지 중 각 영양소가 차지하는 비율을 구합니다.
    """
    carbohydrate_kcal = float(food.get("carbohydrate_g") or 0) * 4
    fat_kcal = float(food.get("fat_g") or 0) * 9
    protein_kcal = float(food.get("protein_g") or 0) * 4

    total_macro_kcal = carbohydrate_kcal + fat_kcal + protein_kcal

    if total_macro_kcal <= 0:
        return {"carbohydrate": 0.0, "fat": 0.0, "protein": 0.0}

    return {
        "carbohydrate": round(carbohydrate_kcal / total_macro_kcal * 100, 1),
        "fat": round(fat_kcal / total_macro_kcal * 100, 1),
        "protein": round(protein_kcal / total_macro_kcal * 100, 1),
    }


def get_recommendation(food: dict[str, Any]) -> dict[str, Any] | None:
    """GI가 70 이상인 음식에 한해 CSV의 대체음식을 조회합니다."""
    gi = float(food["gi"])

    if gi < 70:
        return None

    alternative_food = food.get("alternative_food")
    if not alternative_food:
        return None

    recommendation = find_food(str(alternative_food))
    if recommendation is None:
        return None

    recommendation["macro_ratio"] = calculate_macro_ratio(recommendation)
    return recommendation


def analyze_detected_food(detected_food: str) -> dict[str, Any]:
    """
    YOLO가 반환한 음식명을 받아 영양정보, GI 위험도,
    영양소 비율, 대체음식 추천 결과를 생성합니다.
    """
    food = find_food(detected_food)

    if food is None:
        return {
            "detected_food": detected_food,
            "error": "인식된 음식의 영양 정보가 CSV에 없습니다.",
            "error_message": (
                "YOLO 클래스 이름과 gi_db.csv의 food 값이 같은지 확인하거나 "
                "gi_analysis.py의 FOOD_NAME_ALIASES에 이름을 추가해주세요."
            ),
            "retry": True,
        }

    gi = float(food["gi"])
    risk_label, risk_description = get_gi_risk(gi)

    food["macro_ratio"] = calculate_macro_ratio(food)
    recommendation = get_recommendation(food)

    return {
        "detected_food": detected_food,
        "nutrition": food,
        "risk_label": risk_label,
        "risk_description": risk_description,
        "recommendation": recommendation,
        "retry": False,
    }
