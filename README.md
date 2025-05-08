# 낙상 감지 시스템 (Fall Detection System)

실시간 비디오 스트림에서 사람의 낙상과 눕기 행동을 감지하는 AI 기반 시스템입니다.

## 주요 기능

- 실시간 사람 감지 및 추적
- 포즈 추정(Pose Estimation)을 통한 골격 분석
- 행동 인식 분류(낙상, 눕기 등)
- 웹 브라우저를 통한 비디오 스트리밍
- 다양한 카메라 소스 지원

## 시스템 구성

- TinyYOLOv3: 사람 객체 감지
- FastPose: 사람 포즈 추정
- TSSTG: 시간적 공간적 그래프 기반 행동 인식
- Flask: 웹 스트리밍 서버

## 실행 방법

### 빠른 시작
제공된 배치 파일로 쉽게 실행 가능:
```
run_stream_server.bat
```

### 수동 실행
다음 명령어로 프로그램을 실행할 수 있습니다:
```
python stream_server.py -C 0 --device cuda
```

### 매개변수
- `-C, --camera`: 카메라 소스 (기본값: 0, 웹캠)
- `--device`: 연산 장치 (cuda 또는 cpu)
- `--port`: 스트리밍 서버 포트 (기본값: 5050)

## 접속 방법
프로그램 실행 후, 같은 네트워크 내의 다른 기기에서 다음 URL로 접속 가능:
```
http://[서버IP주소]:5050
```

## 시스템 요구사항
- Python 3.6 이상
- PyTorch
- OpenCV
- CUDA (GPU 가속 사용 시) 