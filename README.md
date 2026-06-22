# 음식 GI 및 영양 분석 시스템

음식 이미지를 YOLOv12 기반 분류 모델로 인식하고, 해당 음식의 영양소와
GI(Glycemic Index, 혈당지수) 정보를 제공하는 Flask 웹 애플리케이션입니다.

GI가 높은 음식은 CSV 데이터에 등록된 낮은 GI의 대체 음식을 추천하며,
사용자가 분석한 음식은 날짜별 섭취 기록으로 저장합니다.

## 주요 기능

- 실시간 카메라를 이용한 음식 촬영
- PNG, JPG, JPEG, WEBP 이미지 업로드
- 학습된 YOLOv12n 분류 모델을 이용한 음식 인식
- 칼로리, 탄수화물, 지방, 단백질, 당류, 식이섬유 정보 출력
- 탄수화물·지방·단백질의 열량 기준 비율 계산
- GI 지수에 따른 낮음·보통·높음 위험도 표시
- 고GI 음식의 대체 음식과 영양정보 추천
- 날짜별 음식 섭취 기록 저장 및 조회

## 모델 구성

### YOLOv12와 YOLOv12n

YOLOv12는 모델 구조와 세대를 의미하며, YOLOv12n은 YOLOv12 계열 중 가장
작고 빠른 Nano 모델입니다. 이 프로젝트는 YOLOv12n의 이미지 분류 모델을
음식 데이터셋으로 추가 학습했습니다.

### 프로젝트에서 사용하는 모델 파일

| 파일 | 작업 종류 | 사용 위치 |
| --- | --- | --- |
| `backend/best.pt` | 음식 15종 이미지 분류(Classification) | Flask 웹 애플리케이션의 실제 음식 분석 |
| `yolo12n.pt` | 일반 객체 탐지(Object Detection) | `food_detection`의 독립 실행 예제 |

`best.pt`는 YOLOv12n 분류 모델을 프로젝트 음식 데이터셋으로 추가 학습한
결과입니다. 웹 애플리케이션은 일반 객체 탐지 모델인 `yolo12n.pt`가 아니라
학습된 음식 분류 모델인 `backend/best.pt`를 사용합니다.

프로젝트의 YOLOv12 라이브러리는
[`sunsmarterjie/yolov12`](https://github.com/sunsmarterjie/yolov12)의
커밋 `01a22c0603e0eaa6d9bd62120a391e744d92cea2`로 고정되어 있습니다.

## 인식 가능한 음식

현재 모델은 다음 15개 클래스를 분류합니다.

1. Banana (바나나)
2. Chocolate (초콜릿)
3. Fried egg (계란후라이)
4. Gimbap (김밥)
5. Grilled salmon (구운 연어)
6. Jajangmyeon (짜장면)
7. Kimchi (김치)
8. Ramen (라면)
9. Red apple (사과)
10. Salad (샐러드)
11. Sandwich (샌드위치)
12. Sweet potato (고구마)
13. Tteokbokki (떡볶이)
14. Waffle (와플)
15. White rice (흰쌀밥)

## 동작 과정

1. 사용자가 카메라로 촬영하거나 이미지 파일을 업로드합니다.
2. Flask 서버가 이미지를 `static/uploads`에 저장합니다.
3. `best.pt` 분류 모델이 가장 가능성이 높은 음식 클래스를 반환합니다.
4. `gi_analysis.py`가 모델 클래스 이름을 정규화합니다.
5. `gi_db.csv`에서 해당 음식의 영양정보와 GI 지수를 조회합니다.
6. GI가 70 이상이면 낮은 GI의 대체 음식을 조회합니다.
7. 분석 결과를 화면에 출력하고 날짜별 섭취 기록에 저장합니다.

## 프로젝트 구조

```text
osp-project/
├── backend/
│   ├── app.py                 # Flask 서버, 이미지 저장, YOLO 추론
│   ├── best.pt                # 음식 15종 학습 분류 모델
│   ├── gi_analysis.py         # 영양소·GI 분석과 대체 음식 추천
│   ├── history_manager.py     # 날짜별 섭취 기록 관리
│   └── intake_history.csv     # 사용자의 음식 섭취 기록
├── Dataset/
│   ├── train/                 # 학습 이미지 1,940장
│   ├── val/                   # 검증 이미지 324장
│   └── test/                  # 테스트 이미지 177장
├── food_detection/
│   ├── camera_food_detect.py  # 웹캠 객체 탐지 예제
│   └── detect_food.py         # 이미지 객체 탐지 예제
├── static/
│   └── uploads/               # 촬영·업로드 이미지 저장 위치
├── templates/
│   ├── index.html             # 카메라 및 파일 입력 화면
│   ├── result.html            # 음식 분석 결과 화면
│   └── history.html           # 날짜별 섭취 기록 화면
├── gi_db.csv                  # 음식별 영양소, GI, 대체 음식 데이터
├── requirements.txt           # 라이브러리 버전과 YOLOv12 커밋
├── yolo12n.pt                 # 일반 객체 탐지 모델
├── LICENSE
└── README.md
```

데이터셋의 각 분할은 동일한 15개 음식 클래스 폴더로 구성됩니다.

## GI 데이터 구성

`gi_db.csv`는 다음 정보를 관리합니다.

| 열 | 설명 |
| --- | --- |
| `food` | 모델 클래스와 연결되는 음식 식별자 |
| `display_name` | 화면에 표시할 한글 음식명 |
| `category` | 음식 분류 |
| `calories_kcal` | 칼로리 |
| `carbohydrate_g` | 탄수화물 함량 |
| `fat_g` | 지방 함량 |
| `protein_g` | 단백질 함량 |
| `sugar_g` | 당류 함량 |
| `fiber_g` | 식이섬유 함량 |
| `gi` | 혈당지수 |
| `alternative_food` | 고GI 음식의 대체 음식 식별자 |

모델 클래스의 공백과 하이픈은 내부적으로 밑줄 형태로 정규화됩니다. 모델명과
CSV 식별자가 다른 경우 `backend/gi_analysis.py`의 `FOOD_NAME_ALIASES`에서
연결합니다.

## 실행 환경

| 구성 요소 | 버전 |
| --- | --- |
| Python | 3.10 |
| Flask | 3.1.3 |
| NumPy | 1.26.4 |
| OpenCV | 4.11.0.86 |
| Pandas | 2.3.3 |
| Pillow | 12.2.0 |
| PyTorch | 2.12.0 |
| Torchvision | 0.27.0 |
| YOLOv12 / Ultralytics | 8.3.63, 지정 커밋 사용 |

## 설치

### 1. Conda 환경 생성

```bash
conda create -n foodai python=3.10
conda activate foodai
```

### 2. 프로젝트 라이브러리 설치

```bash
cd ~/Desktop/osp-project
pip install -r requirements.txt
```

`requirements.txt`에는 정상 실행 환경의 핵심 라이브러리 버전과 프로젝트에서
사용한 YOLOv12 저장소의 커밋이 고정되어 있습니다.

Apple Silicon Mac에서는 FlashAttention을 사용할 수 없다는 안내가 출력될 수
있습니다. 이 경우 PyTorch의 `scaled_dot_product_attention`으로 대체되며
프로그램 실행에는 문제가 없습니다.

## 실행 방법

```bash
conda activate foodai
cd ~/Desktop/osp-project
python backend/app.py
```

컴퓨터 브라우저에서 다음 주소로 접속합니다.

```text
http://127.0.0.1:5050
```

같은 Wi-Fi에 연결된 휴대폰에서는 로컬 IP와 포트 `5050`을 사용합니다.

```text
http://로컬-IP:5050
```

Flask 서버는 `0.0.0.0:5050`에서 실행되므로 같은 네트워크의 기기에서 접속할
수 있습니다. 실행 중인 컴퓨터의 방화벽에서 Python 연결 허용이 필요할 수 있습니다.

## 사용 방법

### 카메라 촬영

1. 브라우저의 카메라 권한을 허용합니다.
2. 카메라 화면에 음식이 잘 보이도록 배치합니다.
3. `촬영 후 분석` 버튼을 누릅니다.
4. 결과 화면에서 음식명, 영양소, GI 정보를 확인합니다.

실시간 카메라는 브라우저 보안 정책상 HTTPS 또는 컴퓨터의 localhost 환경에서
사용할 수 있습니다. 일반 HTTP로 접속한 휴대폰에서는 실시간 카메라가 제한될
수 있으므로 이미지 파일 업로드 기능을 사용할 수 있습니다.

### 이미지 파일 업로드

1. `파일 선택` 버튼을 누릅니다.
2. PNG, JPG, JPEG 또는 WEBP 이미지를 선택합니다.
3. `파일 분석` 버튼을 누릅니다.
4. 분석 결과와 대체 음식 정보를 확인합니다.

업로드 가능한 최대 요청 크기는 10MB입니다. 선택한 파일은
`multipart/form-data` 방식으로 Flask 서버에 전달됩니다.

## 모델 학습 결과

- 학습 Epoch: 50
- 음식 클래스: 15개
- 학습 이미지: 1,940장
- 검증 이미지: 324장
- 테스트 이미지: 177장
- 검증 Top-1 Accuracy: 0.991
- 검증 Top-5 Accuracy: 1.000

학습 결과는 당시 사용한 데이터셋 분할을 기준으로 하며, 데이터셋이 변경되면
성능을 다시 검증해야 합니다.

## 제한사항

- 현재 모델은 한 이미지에서 하나의 대표 음식만 분류합니다.
- 학습하지 않은 음식도 15개 클래스 중 하나로 분류될 수 있습니다.
- 화면에 표시되는 영양소와 GI는 `gi_db.csv`의 기준값입니다.
- 실제 영양정보는 음식의 양, 재료, 조리 방법에 따라 달라질 수 있습니다.
- 본 프로젝트의 분석 결과는 의료 진단이나 전문적인 식단 처방을 대신하지 않습니다.

## License

이 프로젝트의 소스코드는 GNU Affero General Public License v3.0 이상
(`AGPL-3.0-or-later`) 조건으로 배포됩니다. 자세한 내용은 [LICENSE](LICENSE)를
확인하세요.

본 프로젝트는 AGPL-3.0으로 배포되는
[YOLOv12](https://github.com/sunsmarterjie/yolov12) 및 Ultralytics 기반 코드를
사용합니다.

단, `Dataset` 폴더의 이미지 및 데이터셋 파일, `gi_db.csv`, 학습에 사용된 외부 자료는 소스코드 라이선스에 자동으로 포함되지 않습니다.

본 프로젝트의 데이터셋에는 팀이 직접 촬영 또는 수집한 이미지와 외부 출처 이미지가 섞여 있을 수 있으며, 일부 자료는 Roboflow를 통해 전처리, 라벨링, 증강 및 내보내기되었습니다.

데이터셋 및 이미지 자료의 사용 조건은 [DATASET_LICENSE.md](./DATASET_LICENSE.md)를 참고하세요.
