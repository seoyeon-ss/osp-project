from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


# =========================================================
# 1. 섭취기록 파일 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "intake_history.csv"
KST = ZoneInfo("Asia/Seoul")

HISTORY_COLUMNS = [
    "recorded_at",
    "date",
    "time",
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
    "risk_label",
]

NUMERIC_COLUMNS = [
    "calories_kcal",
    "carbohydrate_g",
    "fat_g",
    "protein_g",
    "sugar_g",
    "fiber_g",
    "gi",
]


# =========================================================
# 2. 섭취기록 저장
# =========================================================

def save_intake_history(food: dict[str, Any], risk_label: str) -> None:
    """분석이 성공한 음식 정보를 intake_history.csv에 한 줄씩 저장합니다."""
    now = datetime.now(KST)

    record = {
        "recorded_at": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "food": food["food"],
        "display_name": food["display_name"],
        "category": food["category"],
        "calories_kcal": food["calories_kcal"],
        "carbohydrate_g": food["carbohydrate_g"],
        "fat_g": food["fat_g"],
        "protein_g": food["protein_g"],
        "sugar_g": food["sugar_g"],
        "fiber_g": food["fiber_g"],
        "gi": food["gi"],
        "risk_label": risk_label,
    }

    history_df = pd.DataFrame([record], columns=HISTORY_COLUMNS)
    file_exists = HISTORY_PATH.exists() and HISTORY_PATH.stat().st_size > 0

    history_df.to_csv(
        HISTORY_PATH,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig",
    )


# =========================================================
# 3. 기록 조회 및 날짜별 집계
# =========================================================

def get_daily_gi_status(average_gi: float) -> tuple[str, str]:
    """날짜별 평균 GI를 양호, 주의, 관리 필요로 분류합니다."""
    if average_gi <= 55:
        return "양호", "이날 섭취한 음식의 평균 GI가 낮은 수준입니다."

    if average_gi <= 69:
        return "주의", "이날 섭취한 음식의 평균 GI가 중간 수준입니다."

    return "관리 필요", "이날 섭취한 음식의 평균 GI가 높아 저GI 음식 선택이 필요합니다."


def read_history() -> pd.DataFrame:
    """섭취기록 CSV를 읽고 계산에 필요한 숫자형 열을 정리합니다."""
    if not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    history_df = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig")

    for column in HISTORY_COLUMNS:
        if column not in history_df.columns:
            history_df[column] = None

    for column in NUMERIC_COLUMNS:
        history_df[column] = pd.to_numeric(
            history_df[column], errors="coerce"
        ).fillna(0)

    # 이전 기록에 recorded_at이 없다면 날짜와 시간을 이용해 생성합니다.
    missing_recorded_at = history_df["recorded_at"].isna() | (
        history_df["recorded_at"].astype(str).str.strip() == ""
    )
    history_df.loc[missing_recorded_at, "recorded_at"] = (
        history_df.loc[missing_recorded_at, "date"].astype(str)
        + "T"
        + history_df.loc[missing_recorded_at, "time"].astype(str)
    )

    return history_df[HISTORY_COLUMNS]


def to_python_value(value: Any) -> Any:
    """Pandas와 NumPy 값을 HTML에서 사용할 수 있는 Python 값으로 변환합니다."""
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


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Jinja에 전달하기 전에 NaN과 NumPy 자료형을 정리합니다."""
    cleaned: dict[str, Any] = {}

    for key, value in record.items():
        python_value = to_python_value(value)

        if isinstance(python_value, float) and math.isnan(python_value):
            python_value = None

        cleaned[key] = python_value

    return cleaned


def build_grouped_history() -> list[dict[str, Any]]:
    """기록을 날짜별로 묶고 평균 GI, GI 상태, 총 칼로리를 계산합니다."""
    history_df = read_history()

    if history_df.empty:
        return []

    history_df = history_df.sort_values("recorded_at", ascending=False)
    grouped_history: list[dict[str, Any]] = []

    for date, group in history_df.groupby("date", sort=False):
        average_gi = round(float(group["gi"].mean()), 1)
        total_calories = round(float(group["calories_kcal"].sum()), 1)
        gi_status, gi_message = get_daily_gi_status(average_gi)

        records = [
            sanitize_record(record)
            for record in group.to_dict("records")
        ]

        grouped_history.append(
            {
                "date": date,
                "average_gi": average_gi,
                "total_calories": total_calories,
                "gi_status": gi_status,
                "gi_message": gi_message,
                "records": records,
            }
        )

    return grouped_history
