import os
import cv2
import time
import torch
import argparse
import numpy as np
from requests import post

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

# source = '../Data/test_video/test7.mp4'
# source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
source = '../Data/falldata/Home/Videos/video (1).avi'


# source = 2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def send_whatsapp(attachment, message, phone):
    files = {}
    if attachment and os.path.exists(attachment):
        files["attachment1"] = open(attachment, "rb")

    params = {"message": message,
              "recipient_phone": phone,
              "ignore_replies": True,
              }

    result = post("http://%s/api/whastapp_messages/" % "agricsec.meteor-comm.com", data=params, files=files)
    print(result.text)


def preproc(image):
    """preprocess function for CameraLoader.
    """
    # Ensure we are working with a copy before resize_fn
    current_image = image.copy()
    if resize_fn is not None:
        current_image = resize_fn(current_image) # Pass the copy to resize_fn
    
    # Ensure we are working with a copy before cvtColor
    # image_to_convert = current_image.copy() 
    # cvtColor는 일반적으로 새 이미지를 반환하므로, 이 복사는 필수는 아닐 수 있음
    # 하지만 안전을 위해 추가하거나, resize_fn이 새 이미지를 반환하는지 확인 필요

    # cvtColor 자체가 새 이미지 객체를 반환하므로 image_to_convert.copy()는 필요 없을 수 있습니다.
    # 그러나 resize_fn이 내부적으로 원본을 변경할 가능성을 고려하여 위에서 current_image를 사용합니다.
    final_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
    return final_image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == '__main__':
    # 시스템의 CUDA 사용 가능 여부 확인
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        # GPU 메모리 사용 최적화
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                     help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=320,
                     help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='192x128',
                     help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                     help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                     help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                     help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                     help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                     help='Device to run model on cpu or cuda.')
    par.add_argument('--optimize_fps', action='store_true', 
                     help='Optimize for higher frame rate at the expense of some accuracy.')
    args = par.parse_args()

    device = args.device
    print(f"Running on device: {device}")
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available! Falling back to CPU.")
        device = 'cpu'

    # FPS 최적화 옵션
    if args.optimize_fps:
        print("FPS 최적화 모드 활성화됨 - 정확도가 약간 낮아질 수 있습니다")
        # 검출 주기 설정 (매 프레임마다 검출하지 않음)
        detection_interval = 2
    else:
        detection_interval = 1
    
    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 300
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    print(f"Creating action model with device: {device}")
    action_model = TSSTG(device=device)
    
    print("Models loaded successfully")

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, batch_size=2, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    # frame_size = cam.frame_size
    # scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    paused = 0
    while cam.grabbed():
        if paused > 0:
            paused -= 1
        f += 1
        source_frame_rgb = cam.getitem()
        if source_frame_rgb is None:
            continue
        frame_to_draw_on = source_frame_rgb.copy()

        # Detect humans bbox in the frame with detector model.
        detected_for_pose_input = None
        if not (args.optimize_fps and f % detection_interval != 0 and len(tracker.tracks) > 0):
            detected_for_pose_input = detect_model.detect(source_frame_rgb, need_resize=False, expand_bb=10)

        tracker.predict()
        
        current_pose_input_tensor = detected_for_pose_input
        temp_track_detections = []
        for track_item_for_pose in tracker.tracks:
            device_for_tensor = args.device # main.py에서는 args.device 사용
            if current_pose_input_tensor is not None and hasattr(current_pose_input_tensor, 'device'):
                 device_for_tensor = current_pose_input_tensor.device

            track_as_detection_tensor = torch.tensor(
                [track_item_for_pose.to_tlbr().tolist() + [0.5, 1.0, 0.0]], 
                dtype=torch.float32, 
                device=device_for_tensor
            )
            temp_track_detections.append(track_as_detection_tensor)

        if temp_track_detections:
            all_track_detections_tensor = torch.cat(temp_track_detections, dim=0)
            if current_pose_input_tensor is not None:
                current_pose_input_tensor = torch.cat([current_pose_input_tensor, all_track_detections_tensor], dim=0)
            else:
                current_pose_input_tensor = all_track_detections_tensor

        detections_for_tracker_update = []
        if current_pose_input_tensor is not None:
            poses = pose_model.predict(source_frame_rgb, current_pose_input_tensor[:, 0:4], current_pose_input_tensor[:, 4])
            detections_for_tracker_update = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]
            # args.show_detected 부분은 이미 주석 처리되어 있음 (빨간색 초기 감지 박스)

        tracker.update(detections_for_tracker_update)

        # event는 루프 시작 시 또는 객체 없을 때 초기화
        if not tracker.tracks:
            frame_to_draw_on = source_frame_rgb.copy() # 객체 없으면 원본으로 초기화 (잔상 제거)
            event = "" 
        else:
            # event = "" # for 루프 내에서 action_name에 따라 설정될 것이므로 여기서 초기화 불필요
            pass # 트랙이 있으면 이전 frame_to_draw_on (복사된 원본)을 계속 사용

        # 객체가 있을 때만 트랙별 그리기를 수행
        if tracker.tracks:
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                # center = track.get_center().astype(int) # 현재 사용 안함

                action = 'pending..'
                action_name = 'pending..'
                clr = (0, 255, 0)  # RGB Green
                # event = "" # 루프 시작시 초기화
            
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, source_frame_rgb.shape[:2])
                    predicted_action_name = action_model.class_names[out[0].argmax()]
                    if predicted_action_name in action_model.class_names:
                        action_name = predicted_action_name
                    action = action_name
                
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)  # RGB Red
                    event = "Fall Down" # event는 여기서 설정됨
                elif action_name == 'Lying Down':
                    event = "Lying Down" # event는 여기서 설정됨
                # 'pending..'이 아닌 경우에만 event가 의미를 가짐
                
                if action_name == 'pending..':
                    continue # pending 상태는 그리지 않음

                if track.time_since_update == 0: # 활성 트랙만 그림
                    if args.show_skeleton and len(track.keypoints_list) > 0:
                        current_skeleton_color = (255, 0, 0) if action_name == 'Fall Down' else (0, 255, 0)
                        frame_to_draw_on = draw_single(frame_to_draw_on, track.keypoints_list[-1], skeleton_color=current_skeleton_color)
                    
                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    text_color_on_bg = (255, 255, 255)  # White text

                    cv2.rectangle(frame_to_draw_on, (bbox[0], bbox[1]), (bbox[2], bbox[3]), clr, 1)
                    text_content = action 
                    (text_w, text_h), baseline = cv2.getTextSize(text_content, font_face, font_scale, font_thickness)
                    bg_y1_rect = max(0, bbox[1] - text_h - baseline) 
                    cv2.rectangle(frame_to_draw_on, (bbox[0], bg_y1_rect), (bbox[0] + text_w, bbox[1]), clr, cv2.FILLED)
                    cv2.putText(frame_to_draw_on, text_content, (bbox[0], bbox[1] - baseline), font_face, font_scale, text_color_on_bg, font_thickness)

        # FPS 표시는 항상 수행 (객체 유무와 관계 없이)
        frame_to_draw_on = cv2.resize(frame_to_draw_on, (0, 0), fx=2., fy=2.)
        frame_to_draw_on = cv2.putText(frame_to_draw_on, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        fps_time = time.time()

        final_frame_bgr = cv2.cvtColor(frame_to_draw_on, cv2.COLOR_RGB2BGR)

        if outvid:
            writer.write(final_frame_bgr)
        
        display_height = 720 # 기본 디스플레이 높이 설정 또는 args에서 가져오기
        if hasattr(args, 'display_height') and args.display_height is not None:
            display_height = args.display_height
        imS = image_resize(final_frame_bgr, height=display_height) 
        cv2.imshow('frame', imS)
        
        if event: # event가 있을 때 (Fall Down 등)
            if paused <= 0:
                paused = 120
                image_name = "frame%d.jpg" % f
                cv2.imwrite(image_name, final_frame_bgr) # 저장 시 BGR 프레임 사용
                # send_whatsapp(image_name, event, "972543933773") # WhatsApp 전송은 주석 처리
            # else:
                # print(paused) # paused 값 출력은 주석 처리

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
