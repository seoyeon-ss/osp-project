from __future__ import annotations

import traceback
import base64
import binascii
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request
from ultralytics import YOLO
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

import gi_analysis
import history_manager


# =========================================================
# 1. Flask, 이미지 업로드, YOLO 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "best.pt"
UPLOAD_FOLDER = PROJECT_ROOT / "static" / "uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
YOLO_CONFIDENCE = 0.25

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
app.config["MAX_FORM_MEMORY_SIZE"] = MAX_UPLOAD_SIZE
app.config["TEMPLATES_AUTO_RELOAD"] = True

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# YOLO 모델은 최초 요청 때 한 번만 불러옵니다.
_model: YOLO | None = None


# =========================================================
# 2. 업로드 이미지 저장
# =========================================================

def allowed_file(filename: str) -> bool:
    """업로드한 파일이 허용된 이미지 확장자인지 확인합니다."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_uploaded_file(uploaded_file: FileStorage) -> tuple[Path, str]:
    """파일 선택으로 전달된 이미지를 저장합니다."""
    if not uploaded_file.filename:
        raise ValueError("선택된 이미지 파일이 없습니다.")

    if not allowed_file(uploaded_file.filename):
        raise ValueError("PNG, JPG, JPEG, WEBP 형식의 이미지만 업로드할 수 있습니다.")

    safe_name = secure_filename(uploaded_file.filename)
    extension = Path(safe_name).suffix.lower() or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{extension}"

    full_path = UPLOAD_FOLDER / unique_name
    uploaded_file.save(full_path)

    return full_path, f"static/uploads/{unique_name}"


def save_camera_image(image_data: str) -> tuple[Path, str]:
    """카메라 촬영 후 전달된 Base64 이미지를 저장합니다."""
    if not image_data:
        raise ValueError("카메라 촬영 이미지가 전달되지 않았습니다.")

    try:
        header, encoded_data = image_data.split(",", 1)
    except ValueError as exc:
        raise ValueError("카메라 이미지 형식이 올바르지 않습니다.") from exc

    image_extensions = {
        "data:image/png;base64": ".png",
        "data:image/jpeg;base64": ".jpg",
        "data:image/webp;base64": ".webp",
    }
    extension = image_extensions.get(header.lower())

    if extension is None:
        raise ValueError("카메라 이미지 형식이 올바르지 않습니다.")

    try:
        binary_data = base64.b64decode(encoded_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("카메라 이미지를 해석할 수 없습니다.") from exc

    unique_name = f"{uuid.uuid4().hex}{extension}"
    full_path = UPLOAD_FOLDER / unique_name
    full_path.write_bytes(binary_data)

    return full_path, f"static/uploads/{unique_name}"


def save_request_image() -> tuple[Path, str]:
    """파일 업로드 또는 카메라 촬영 이미지 중 전달된 이미지를 저장합니다."""
    uploaded_file = request.files.get("file")

    if uploaded_file is not None and uploaded_file.filename:
        return save_uploaded_file(uploaded_file)

    camera_image = request.form.get("image", "").strip()
    if camera_image:
        return save_camera_image(camera_image)

    raise ValueError("촬영하거나 업로드한 이미지가 없습니다.")


# =========================================================
# 3. YOLO 음식 인식
# =========================================================

def get_yolo_model() -> YOLO:
    """학습된 YOLO 모델을 최초 한 번만 불러옵니다."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO 모델 파일이 없습니다. {MODEL_PATH.name}을 프로젝트 폴더에 넣어주세요."
            )

        _model = YOLO(str(MODEL_PATH))

    return _model


def tensor_to_list(value: Any) -> list[Any]:
    """Tensor 또는 NumPy 배열을 일반 리스트로 변환합니다."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()

    return list(value)


def tensor_to_float(value: Any) -> float:
    """Tensor 또는 숫자를 float 자료형으로 변환합니다."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()

    return float(value)


def get_class_name(names: Any, class_id: int) -> str:
    """YOLO 클래스 번호에 해당하는 음식 이름을 반환합니다."""
    if names is None:
        raise ValueError("YOLO 모델의 클래스 이름을 확인할 수 없습니다.")

    return str(names[class_id])


def detect_food(image_path: Path) -> tuple[str | None, float | None]:
    """YOLO 결과 중 신뢰도가 가장 높은 음식 이름을 반환합니다."""
    model = get_yolo_model()
    results = model.predict(
        source=str(image_path),
        conf=YOLO_CONFIDENCE,
        verbose=False,
    )

    if not results:
        return None, None

    result = results[0]
    names = getattr(result, "names", None) or getattr(model, "names", None)

    # Object Detection 모델 결과
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        confidences = tensor_to_list(boxes.conf)
        class_ids = tensor_to_list(boxes.cls)

        best_index = max(
            range(len(confidences)),
            key=lambda index: confidences[index],
        )

        class_id = int(class_ids[best_index])
        confidence = float(confidences[best_index])

        return get_class_name(names, class_id), round(confidence, 4)

    # Classification 모델 결과
    probabilities = getattr(result, "probs", None)
    if probabilities is not None and getattr(probabilities, "top1", None) is not None:
        class_id = int(probabilities.top1)
        confidence = tensor_to_float(probabilities.top1conf)

        return get_class_name(names, class_id), round(confidence, 4)

    return None, None


# =========================================================
# 4. Flask 페이지 및 기능 연결
# =========================================================

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    image_path_for_html: str | None = None

    try:
        # 이미지 저장 후 YOLO에서 음식 이름을 받습니다.
        saved_image_path, image_path_for_html = save_request_image()
        detected_food, confidence = detect_food(saved_image_path)

        if detected_food is None:
            result = {
                "image_path": image_path_for_html,
                "error": "음식을 인식하지 못했습니다.",
                "error_message": "음식이 잘 보이도록 다른 각도에서 다시 촬영해주세요.",
                "retry": True,
            }
            return render_template("result.html", result=result)

        # gi_analysis.py에서 영양정보, GI 위험도, 대체음식을 분석합니다.
        result = gi_analysis.analyze_detected_food(detected_food)
        result["image_path"] = image_path_for_html
        result["confidence"] = confidence

        # 분석에 성공한 음식만 history_manager.py를 통해 저장합니다.
        if result.get("nutrition"):
            history_manager.save_intake_history(
                food=result["nutrition"],
                risk_label=result["risk_label"],
            )

        return render_template("result.html", result=result)

    except (ValueError, FileNotFoundError) as exc:
        result = {
            "image_path": image_path_for_html,
            "error": "분석을 진행할 수 없습니다.",
            "error_message": str(exc),
            "retry": True,
        }
        return render_template("result.html", result=result), 400

    except Exception as exc:
        print("===== 실제 오류 발생 =====")
        print(exc)
        traceback.print_exc()

        result = {
            "image_path": image_path_for_html,
            "error": "분석 중 오류가 발생했습니다.",
            "error_message": str(exc),
            "retry": True,
        }
    return render_template("result.html", result=result), 500


@app.get("/history")
def history():
    # history_manager.py에서 날짜별 요약 결과를 받아 화면에 전달합니다.
    grouped_history = history_manager.build_grouped_history()

    return render_template(
        "history.html",
        grouped_history=grouped_history,
    )


@app.errorhandler(413)
def file_too_large(_error):
    result = {
        "error": "업로드한 이미지가 너무 큽니다.",
        "error_message": "10MB 이하의 이미지 파일을 사용해주세요.",
        "retry": True,
    }
    return render_template("result.html", result=result), 413


if __name__ == "__main__":
    # 세 파일 중 app.py만 실행합니다.
    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)
