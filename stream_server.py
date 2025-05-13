import os
import time
import cv2
import argparse
import threading
import socket
from flask import Flask, Response, Flask, render_template_string
from CameraLoader import CamLoader, CamLoader_Q
from Detection.Utils import ResizePadding
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Tracker, Detection
from ActionsEstLoader import TSSTG
import torch
import numpy as np
from fn import draw_single
import datetime

app = Flask(__name__)

# 전역 변수로 최신 프레임 저장
global_frame = None
frame_lock = threading.Lock()
resize_fn = None  # 전역 변수로 선언
args = None  # 명령줄 인수 저장을 위한 전역 변수

# 이미지 저장 설정
IMAGE_SAVE_DIR = 'fall_images'

# 초기화 함수
def init_app():
    # 이미지 저장 디렉토리 생성
    if not os.path.exists(IMAGE_SAVE_DIR):
        os.makedirs(IMAGE_SAVE_DIR)
        print(f"이미지 저장 디렉토리 생성 완료: {IMAGE_SAVE_DIR}")

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def preproc(image):
    """preprocess function for CameraLoader."""
    global resize_fn
    if resize_fn is not None:
        image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)"""
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

def process_video(camera_source, device='cuda'):
    global global_frame, frame_lock, resize_fn
    
    print(f"process_video 시작: 장치={device}, 카메라 소스={camera_source}")
    
    try:
        # 모델 로드
        inp_dets = 320  # 검출 입력 크기
        detect_model = TinyYOLOv3_onecls(inp_dets, device=device)
        
        inp_pose = (192, 128)
        pose_model = SPPE_FastPose('resnet50', inp_pose[0], inp_pose[1], device=device)
        
        # Tracker 초기화
        tracker = Tracker(max_age=30, n_init=3)
        
        # 행동 인식 모델
        action_model = TSSTG(device=device)
        
        # 전처리 함수 (전역변수에 할당)
        resize_fn = ResizePadding(inp_dets, inp_dets)
        
        print(f"카메라 초기화 중: {camera_source}")
        # 카메라 로더 설정
        if isinstance(camera_source, str) and os.path.isfile(camera_source):
            print(f"비디오 파일 로딩: {camera_source}")
            cam = CamLoader_Q(camera_source, queue_size=1000, batch_size=1, preprocess=preproc).start()
        else:
            # 문자열 '0'을 정수 0으로 변환
            if isinstance(camera_source, str) and camera_source.isdigit():
                camera_source = int(camera_source)
            print(f"카메라 로딩: {camera_source}")
            cam = CamLoader(camera_source, preprocess=preproc).start()
            print("카메라 로더 시작됨")
        
        fps_time = 0
        f = 0
        
        # 낙상 감지 상태 추적용 변수
        fall_states = {}  # track_id를 키로 하는 딕셔너리
        
        print("비디오 처리 시작")
        # 첫 프레임이 로드되었는지 확인
        if not cam.grabbed():
            print("첫 프레임을 가져올 수 없습니다. 카메라 연결을 확인하세요.")
            # 더미 프레임 생성
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_frame = cv2.putText(dummy_frame, "카메라를 찾을 수 없습니다", (50, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            with frame_lock:
                global_frame = dummy_frame
            time.sleep(5)  # 5초 대기 후 종료
            return
        
        while cam.grabbed():
            f += 1
            frame = cam.getitem()
            image = frame.copy()
            
            # 객체 감지
            detected = detect_model.detect(frame, need_resize=False, expand_bb=10)
            
            # 트래킹 예측
            tracker.predict()
            
            # 이전 트랙 정보 추가
            for track in tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det
            
            detections = []
            event = ""
            
            if detected is not None:
                # 스켈레톤 포즈 예측
                poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
                
                # 객체 생성
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                  ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]
                
                # 감지된 객체 시각화
                for bb in detected[:, 0:5]:
                    # 좌표가 유효한지 확인
                    try:
                        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    except (ValueError, TypeError) as e:
                        print(f"경계 상자 그리기 오류: {e}, 값: {bb[0:4]}")
            
            # 트래커 업데이트
            tracker.update(detections)
            
            # 현재 액티브한 트랙 ID 저장
            active_tracks = set()
            
            # 각 트랙의 행동 예측
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                active_tracks.add(track_id)
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)
                
                action = 'pending..'
                action_name = 'pending..'  # 기본값으로 변수 초기화
                clr = (0, 255, 0)
                confidence = 0
                
                # 30프레임 단위로 행동 예측
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    confidence = out[0].max() * 100
                    action = '{}: {:.2f}%'.format(action_name, confidence)
                    
                    if action_name == 'Fall Down':
                        clr = (255, 0, 0)
                        event = "Fall Down"
                        
                        # 낙상 상태 추적 시작 또는 업데이트
                        if track_id not in fall_states:
                            fall_states[track_id] = {
                                'start_time': time.time(),
                                'is_saved': False,
                                'confidence': confidence
                            }
                        else:
                            # 이미 추적 중이면 상태 업데이트
                            fall_states[track_id]['confidence'] = max(
                                fall_states[track_id]['confidence'], 
                                confidence
                            )
                            
                        # 3초 이상 낙상 상태 지속 확인 및 이미지 저장
                        fall_duration = time.time() - fall_states[track_id]['start_time']
                        
                        if fall_duration >= 3.0 and not fall_states[track_id]['is_saved']:
                            # 현재 시간으로 파일명 생성
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_filename = f"{IMAGE_SAVE_DIR}/fall_{track_id}_{timestamp}.jpg"
                            
                            # 이미지 저장
                            cv2.imwrite(img_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            
                            # 저장 상태 업데이트
                            fall_states[track_id]['is_saved'] = True
                            print(f"낙상 감지! 트랙 ID: {track_id}, 지속 시간: {fall_duration:.2f}초, 이미지 저장됨: {img_filename}")
                    
                    elif action_name == 'Lying Down':
                        clr = (255, 200, 0)
                        event = "Lying Down"
                    else:
                        # 다른 행동이면 낙상 추적 초기화
                        if track_id in fall_states:
                            del fall_states[track_id]
                
                # 시각화 - 안전하게 처리
                try:
                    if len(track.keypoints_list) > 0:
                        if action_name == 'Fall Down':
                            # Fall Down 상황에서는 빨간색 스켈레톤 사용
                            frame = draw_single(frame, track.keypoints_list[-1], skeleton_color=(0, 0, 255))
                        else:
                            # 기본 색상(노란색) 사용
                            frame = draw_single(frame, track.keypoints_list[-1])
                    
                    # 좌표가 유효한지 확인
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cx, cy = int(center[0]), int(center[1])
                    
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    frame = cv2.putText(frame, str(track_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX,
                                       0.4, (255, 0, 0), 2)
                    frame = cv2.putText(frame, action, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_COMPLEX,
                                       0.4, clr, 1)
                    
                    # 낙상 지속 시간 표시 (낙상 감지 중인 경우)
                    if track_id in fall_states and not fall_states[track_id]['is_saved']:
                        duration = time.time() - fall_states[track_id]['start_time']
                        duration_text = f"낙상 지속: {duration:.1f}초"
                        frame = cv2.putText(frame, duration_text, (x1 + 5, y1 + 35), 
                                          cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 1)
                except (ValueError, TypeError, IndexError) as e:
                    print(f"트랙 시각화 오류: {e}")
            
            # 현재 프레임에 없는 트랙 ID 제거 (더 이상 감지되지 않는 객체)
            for track_id in list(fall_states.keys()):
                if track_id not in active_tracks:
                    del fall_states[track_id]
            
            # 프레임 크기 조정 및 FPS 표시
            frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
            frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            fps_time = time.time()
            
            # RGB로 변환 (웹 스트리밍용)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 프레임 저장 (스레드 안전)
            with frame_lock:
                global_frame = frame.copy()
            
            # 프레임 레이트 제어
            time.sleep(0.01)
        
        # 리소스 정리
        print("비디오 처리 종료")
        cam.stop()
        
    except Exception as e:
        print(f"비디오 처리 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        # 오류 메시지가 담긴 프레임 생성
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        error_text = f"오류 발생: {str(e)}"
        error_frame = cv2.putText(error_frame, error_text, (20, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        with frame_lock:
            global_frame = error_frame

def generate_frames():
    """영상 스트림 생성 함수"""
    global global_frame, frame_lock
    
    while True:
        # 프레임 가져오기 (스레드 안전)
        with frame_lock:
            if global_frame is None:
                # 프레임이 없으면 빈 프레임 생성
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = cv2.putText(frame, "비디오 스트림 준비 중...", (50, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                frame = global_frame.copy()
        
        # JPEG로 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # 스트림 데이터 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 프레임 레이트 제어
        time.sleep(0.033)  # ~30 FPS

@app.route('/stream')
def video_feed():
    """비디오 스트림 라우트"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """테스트 페이지 제공"""
    global args
    # 현재 서버의 IP 주소 확인
    local_ip = get_ip_address()
    url = f"http://{local_ip}:{args.port}"
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CCTV 스트리밍 테스트</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 0; padding: 20px; }
            h1 { color: #333; }
            .container { max-width: 800px; margin: 0 auto; }
            img { max-width: 100%; border: 1px solid #ddd; }
            .url-info { background: #f5f5f5; padding: 10px; margin: 15px 0; border-radius: 5px; }
            .note { font-size: 0.9em; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CCTV 스트리밍 테스트</h1>
            <img src="/stream" alt="라이브 스트림">
            <p>시스템이 실시간으로 움직임을 감지하고 넘어짐 상태를 분석합니다.</p>
            <div class="url-info">
                <p>다른 기기에서 접속하려면 다음 URL을 사용하세요: <br>
                <strong>{{url}}</strong></p>
                <p class="note">같은 네트워크에 있어야 접속 가능합니다.</p>
            </div>
        </div>
        <script>
            // 접속 횟수 표시 (옵션)
            let visitors = localStorage.getItem('visitors') || 0;
            localStorage.setItem('visitors', ++visitors);
        </script>
    </body>
    </html>
    """, port=args.port, ip=local_ip, url=url)

# IP 주소 가져오는 함수 정의 (메인 코드 위로 이동)
def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 외부 접속 테스트 (실제로 연결되지 않음)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human Fall Detection Streaming Server')
    parser.add_argument('-C', '--camera', default='0', 
                       help='카메라 번호 또는 비디오 파일 경로')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='모델 실행 장치 (cuda 또는 cpu)')
    parser.add_argument('--port', type=int, default=5050,
                       help='스트리밍 서버 포트')
    
    # 전역 변수에 인수 저장
    args = parser.parse_args()
    
    print(f"서버 시작: 카메라={args.camera}, 장치={args.device}, 포트={args.port}")
    
    # 앱 초기화
    init_app()
    
    # 비디오 처리 스레드 시작
    video_thread = threading.Thread(target=process_video, args=(args.camera, args.device))
    video_thread.daemon = True
    video_thread.start()
    
    # 서버 접속 URL 출력 (네트워크 인터페이스에 따라 접속 방법 안내)
    local_ip = get_ip_address()
    print(f"\n다른 기기에서 이 서버에 접속하려면 다음 URL을 사용하세요:")
    print(f"http://{local_ip}:{args.port}")
    print("\n서버 종료하려면 Ctrl+C를 누르세요...\n")
    
    # 플라스크 서버 시작
    app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False) 