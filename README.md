# 프로젝트 이름 (예: RGB + LiDAR 융합 기반 객체 인식 파이프라인)

## 🧭 프로젝트 개요
- 본 프로젝트는 **로봇 자율주행 환경에서 RGB 카메라와 LiDAR 데이터를 동시 활용**하기 위해 시작되었습니다.  
- 단일 센서에 의존할 경우 발생하는 한계(시야 제한, 조명 민감도, 거리 정보 부족 등)를 극복하고자,  
  **멀티모달 데이터(RGB + LiDAR) 기반의 융합 인지 파이프라인**을 구축했습니다.  

## 🎯 연구 목적
- 로봇 주행 데이터셋에서 **RGB 영상과 LiDAR 포인트클라우드를 동기화**하여  
  탐지/시각화 가능한 형태로 가공.  
- YOLO 기반 객체 검출 결과를 RGB에 오버레이하고, LiDAR는 top-down projection으로 변환.  
- 최종적으로 **시간축 기반의 영상(mp4)** 및 **ROS2 토픽(/image_raw, /yolo/bounding_boxes)** 형태로 제공.  

## 🛠 구현 및 기술 스택
- **ROS2 Humble**: 퍼블리셔/서브스크라이버 구조, 토픽 기반 데이터 스트리밍
- **Python + OpenCV**: RGB 영상 처리 및 저장
- **Open3D / Numpy**: LiDAR PCD 포인트클라우드 파싱 및 시각화
- **YOLOv11**: 객체 탐지 및 바운딩 박스 생성
- **Docker + VSCode Dev Containers**: 재현 가능한 실험 환경

## 🚀 주요 기능
1. **이미지 퍼블리시**  
   - 원본 mp4 또는 jpg 시퀀스를 `/image_raw` 토픽으로 송출
2. **YOLO 객체 검출**  
   - `/yolo/bounding_boxes`, `/yolo/centers` 토픽으로 결과 퍼블리시
3. **LiDAR 처리**  
   - `.pcd` 포맷의 포인트클라우드를 top-down projection으로 변환
4. **RGB+LiDAR 융합 영상 생성**  
   - 좌측: RGB + YOLO 오버레이  
   - 우측: LiDAR top-view (grid, labels, intensity colormap 지원)  
   - 최종 결과를 mp4 파일로 저장 (`output/RGB_LiDAR_side_by_side_<timestamp>.mp4`)

## 📊 결과 예시
- 아래는 동일한 프레임에 대해 RGB와 LiDAR를 융합한 예시입니다.

(여기 GIF나 캡처 삽입)

## 💡 성과 및 의의
- ROS2 기반의 **멀티센서 데이터 파이프라인**을 처음부터 설계/구현 → 실제 로봇 데이터셋에 적용 가능.  
- 단순 시각화가 아니라 **시간 축(mp4)**으로 결과를 정리해 분석과 공유가 용이.  
- 연구/산업적 확장성: 자율주행, 로봇 내비게이션, 의료/산업용 영상 융합에도 응용 가능.  

## 📂 디렉토리 구조
/workspace
 ├── src/                # ROS2 패키지 소스코드  
 │   ├── image_pub/                         # 🎥 이미지·비디오 퍼블리셔 (raw 영상 → /image_raw)
 │   │   ├── image_pub/
 │   │   │   ├── image_pub_node.py          # 메인 노드: video/camera → sensor_msgs/Image 퍼블리시
 │   │   │   ├── image_pub_rgb.py           # RGB 전용 퍼블리시 예시
 │   │   │   └── record_video_node.py       # 비디오 파일 녹화 노드
 │   │   ├── package.xml
 │   │   ├── resource/
 │   │   │   └── image_pub
 │   │   ├── setup.cfg
 │   │   ├── setup.py
 │   │   └── test/                          # 코드 규칙 검사(pep257, flake8 등)
 │   │       ├── test_copyright.py
 │   │       ├── test_flake8.py
 │   │       └── test_pep257.py
 │   │
 │   ├── yolo_subscriber_py/                # 🧠 YOLO 감지 구독 및 퍼블리시
 │   │   ├── yolo_subscriber_py/
 │   │   │   ├── yolo_subscriber_py_node.py # /image_raw 구독 → YOLO 감지 → /yolo/bounding_boxes 퍼블리시
 │   │   │   └── det_to_pc_node.py          # 감지 결과를 포인트클라우드와 결합(확장용)
 │   │   ├── package.xml
 │   │   ├── resource/
 │   │   │   └── yolo_subscriber_py
 │   │   ├── setup.cfg
 │   │   ├── setup.py
 │   │   └── test/
 │   │       ├── test_copyright.py
 │   │       ├── test_flake8.py
 │   │       └── test_pep257.py
 │   │
 │   ├── overlay_viz/                       # 🧩 감지 결과 오버레이 시각화 (YOLO + 원본 영상)
 │   │   ├── overlay_viz/
 │   │   │   └── overlay_viz_node.py        # Bounding Box 시각화 및 융합 화면 출력
 │   │   ├── package.xml
 │   │   ├── resource/
 │   │   │   └── overlay_viz
 │   │   ├── setup.cfg
 │   │   ├── setup.py
 │   │   └── test/
 │   │       ├── test_copyright.py
 │   │       ├── test_flake8.py
 │   │       └── test_pep257.py
 │   │
 │   └── my_bringup/                        # 🚀 통합 실행(launch) + RViz 설정
 │       ├── launch/
 │       │   └── all_nodes.launch.py        # image_pub + yolo_subscriber + overlay_viz 통합 실행
 │       ├── my_bringup/
 │       │   └── __init__.py
 │       ├── package.xml
 │       ├── resource/
 │       │   └── my_bringup
 │       ├── rviz/
 │       │   └── yolo_viz.rviz              # RViz 시각화 설정
 │       ├── setup.cfg
 │       ├── setup.py
 │       └── test/
 │           ├── test_copyright.py
 │           ├── test_flake8.py
 │           └── test_pep257.py
 │
 ├── scripts/            # 데이터 처리 스크립트 (make_rgb_lidar_video.py 등)  
 ├── media/              # 원본 데이터셋 (mp4, pcd 등)  
 ├── output/             # 결과 영상 저장 위치  
 └── README.md  


## 📦 src 패키지 구조

`src/` 아래에는 ROS2 패키지들이 포함되어 있으며, 각 패키지는 다음 역할을 담당합니다:

- **image_pub/**  
  - mp4 영상 또는 jpg 시퀀스를 `/image_raw` 토픽으로 퍼블리시  
  - `image_pub_node.py`, `image_pub_rgb.py`, `record_video_node.py` 포함  

- **yolo_subscriber_py/**  
  - `/image_raw`를 구독하여 YOLOv11 추론 수행  
  - 결과를 `/yolo/bounding_boxes`, `/yolo/centers` 토픽으로 퍼블리시  

- **overlay_viz/**  
  - RGB 이미지 위에 YOLO 바운딩 박스를 합성  
  - `/image_overlay` 토픽으로 결과 송출  

- **my_bringup/**  
  - 전체 노드를 한번에 실행하는 launch 파일 제공  
  - `all_nodes.launch.py` : image_pub + yolo_subscriber_py + overlay_viz + recorder 실행  

이 구조를 통해, **단일 명령으로 전체 파이프라인을 구동**할 수 있습니다.  
