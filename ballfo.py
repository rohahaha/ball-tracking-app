import cv2
import numpy as np
import streamlit as st
from collections import deque
import plotly.graph_objs as go
import pandas as pd
from scipy.signal import savgol_filter
import colorsys
import tempfile
import os
import gdown
import traceback
import urllib.request
import time
from streamlit_plotly_events import plotly_events  # 추가된 import
from streamlit_image_coordinates import streamlit_image_coordinates

# Streamlit 페이지 설정 (반드시 다른 st 명령어보다 먼저 와야 함)
st.set_page_config(
    page_title="객체 탐지 및 속도 분석기",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'Ball Tracking and Energy Analysis Application'
    }
)

# 앱 시작 부분에 CSS 추가
st.markdown("""
    <style>
    .stVideo {
        max-width: 384px !important;
    }
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
    }
    .st-emotion-cache-1v0mbdj img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# 앱 제목
st.title('객체 탐지 및 속도 추적 프로그램_by ROHA')

# 세션 상태 초기화 - 여기에 필요한 변수들 추가
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.analysis_frames = []
    st.session_state.analysis_speeds = []
    st.session_state.analysis_images = {}
    st.session_state.analysis_positions = {}
    st.session_state.selected_frame = None
    st.session_state.video_settings = {}  # video_settings도 추가
    
# 전역 예외 처리
try:
    if not st.session_state.initialized:
        st.session_state.initialized = True
        # 여기에 초기화 코드 추가
except Exception as e:
    st.error(f"초기화 중 오류 발생: {str(e)}")

# YOLO 파일 경로 설정 (URL은 제거하고 단순화)
YOLO_DIR = "yolo"
YOLO_FILES = {
    "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
}
WEIGHTS_FILE_ID = "1XWTMChKOcrVpo-uaIldGp6bRzBfYIGqJ"
WEIGHTS_FILENAME = "yolov4.weights"

def create_yolov4_cfg():
    """YOLOv4 설정 파일 생성"""
    cfg_content = """[net]
batch=64
subdivisions=16
width=608
height=608
channels=3
momentum=0.949
decay=0.0005
angle=0
saturation=1.5
exposure=1.5
hue=.1
learning_rate=0.00261
burn_in=1000
max_batches=500500
policy=steps
steps=400000,450000
scales=.1,.1
mosaic=1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=mish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=mish

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=mish

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=mish

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=mish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=mish

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=mish

[route]
layers=-1,-7

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=mish

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=mish

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=mish

[route]
layers=-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=mish

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=mish

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=mish

[shortcut]
from=-3
activation=linear

[route]
layers=-1,-10

[maxpool]
stride=1
size=13

[route]
layers=-1,-3,-5,-6

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers=85

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-1,-3

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers=54

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers=-1,-3

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear

[yolo]
mask=0,1,2
anchors=12,16,19,36,40,28,36,75,76,55,72,146,142,110,192,243,459,401
classes=80
num=9
jitter=.3
ignore_thresh=.7
truth_thresh=1
scale_x_y=1.2
iou_thresh=0.213
nms_kind=greedynms
beta_nms=0.6
max_delta=5

[route]
layers=-4

[convolutional]
batch_normalize=1
size=3
stride=2
pad=1
filters=256
activation=leaky

[route]
layers=-1,-16

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear

[yolo]
mask=3,4,5
anchors=12,16,19,36,40,28,36,75,76,55,72,146,142,110,192,243,459,401
classes=80
num=9
jitter=.3
ignore_thresh=.7
truth_thresh=1
scale_x_y=1.1
iou_thresh=0.213
nms_kind=greedynms
beta_nms=0.6
max_delta=5

[route]
layers=-4

[convolutional]
batch_normalize=1
size=3
stride=2
pad=1
filters=512
activation=leaky

[route]
layers=-1,-37

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear

[yolo]
mask=6,7,8
anchors=12,16,19,36,40,28,36,75,76,55,72,146,142,110,192,243,459,401
classes=80
num=9
jitter=.3
ignore_thresh=.7
truth_thresh=1
scale_x_y=1.05
iou_thresh=0.213
nms_kind=greedynms
beta_nms=0.6
max_delta=5"""
    return cfg_content.strip()

def save_yolov4_cfg():
    """YOLOv4 설정 파일 저장"""
    try:
        os.makedirs(YOLO_DIR, exist_ok=True)
        cfg_path = os.path.join(YOLO_DIR, "yolov4.cfg")
        
        # 먼저 URL에서 직접 다운로드 시도
        try:
            st.info("Downloading YOLOv4 config from source...")
            response = urllib.request.urlopen(YOLO_FILES["yolov4.cfg"])
            cfg_content = response.read().decode('utf-8')
            
            with open(cfg_path, 'w', newline='\n') as f:
                f.write(cfg_content)
            
            if os.path.exists(cfg_path):
                file_size = os.path.getsize(cfg_path)
                if file_size > 10 * 1024:  # 10KB 이상
                    st.success(f"Downloaded YOLOv4 configuration file (size: {file_size/1024:.1f}KB)")
                    return True
                
        except Exception as e:
            st.warning(f"Failed to download config file: {str(e)}. Trying backup method...")
        
        # 다운로드 실패 시 백업 URL 시도
        backup_url = "https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4.cfg"
        try:
            st.info("Trying backup source for YOLOv4 config...")
            response = urllib.request.urlopen(backup_url)
            cfg_content = response.read().decode('utf-8')
            
            with open(cfg_path, 'w', newline='\n') as f:
                f.write(cfg_content)
            
            if os.path.exists(cfg_path):
                file_size = os.path.getsize(cfg_path)
                if file_size > 10 * 1024:  # 10KB 이상
                    st.success(f"Downloaded YOLOv4 configuration file from backup (size: {file_size/1024:.1f}KB)")
                    return True
                
        except Exception as e:
            st.warning(f"Failed to download from backup: {str(e)}. Using local template...")
        
        # 모든 다운로드 실패 시 로컬 템플릿 사용
        cfg_content = create_yolov4_cfg()
        with open(cfg_path, 'w', newline='\n') as f:
            f.write(cfg_content)
        
        if os.path.exists(cfg_path):
            file_size = os.path.getsize(cfg_path)
            if file_size > 10 * 1024:  # 10KB 이상
                st.success(f"Created YOLOv4 configuration file using template (size: {file_size/1024:.1f}KB)")
                return True
            else:
                st.error(f"Created file is too small: {file_size/1024:.1f}KB")
                if os.path.exists(cfg_path):
                    os.remove(cfg_path)
                return False
        
        st.error("Failed to create valid YOLOv4 configuration file")
        if os.path.exists(cfg_path):
            os.remove(cfg_path)
        return False
        
    except Exception as e:
        st.error(f"Error handling YOLOv4 configuration file: {str(e)}")
        if 'cfg_path' in locals() and os.path.exists(cfg_path):
            os.remove(cfg_path)
        return False

def download_yolo_files():
    """YOLO 모델 파일 다운로드"""
    try:
        os.makedirs(YOLO_DIR, exist_ok=True)
        
        # YOLOv4 cfg 파일 다운로드
        if not os.path.exists(os.path.join(YOLO_DIR, "yolov4.cfg")):
            st.info("Creating YOLOv4 configuration file...")
            if not save_yolov4_cfg():
                return False
        
        # coco.names 파일 다운로드
        names_path = os.path.join(YOLO_DIR, "coco.names")
        if not os.path.exists(names_path):
            st.info("Downloading coco.names...")
            try:
                urllib.request.urlretrieve(YOLO_FILES["coco.names"], names_path)
                st.success("Downloaded coco.names")
            except Exception as e:
                st.error(f"Error downloading coco.names: {str(e)}")
                return False
        
        # weights 파일 다운로드
        weights_path = os.path.join(YOLO_DIR, WEIGHTS_FILENAME)
        if not os.path.exists(weights_path):
            with st.spinner("Downloading YOLOv4 weights from Google Drive... This may take a few minutes..."):
                try:
                    output = gdown.download(
                        f"https://drive.google.com/uc?id={WEIGHTS_FILE_ID}",
                        weights_path,
                        quiet=False
                    )
                    if output is not None and os.path.exists(weights_path):
                        file_size = os.path.getsize(weights_path) / (1024 * 1024)  # MB로 변환
                        if file_size > 200:  # 최소 200MB
                            st.success(f"Successfully downloaded YOLOv4 weights! (File size: {file_size:.1f} MB)")
                        else:
                            st.error("Downloaded weights file appears to be incomplete")
                            if os.path.exists(weights_path):
                                os.remove(weights_path)
                            return False
                    else:
                        st.error("Failed to download weights file")
                        return False
                except Exception as e:
                    st.error(f"Error downloading weights file: {str(e)}")
                    if os.path.exists(weights_path):
                        os.remove(weights_path)
                    return False
        
        # 모든 파일 검증
        return verify_yolo_files()
    
    except Exception as e:
        st.error(f"Error in download process: {str(e)}")
        return False

def verify_yolo_files():
    """YOLO 파일들의 존재 여부와 무결성 확인"""
    required_files = {
        "yolov4.cfg": 10 * 1024,     # 최소 10KB
        "coco.names": 0.5 * 1024,    # 최소 0.5KB (수정됨)
        "yolov4.weights": 200 * 1024 * 1024  # 최소 200MB
    }
    
    try:
        for filename, min_size in required_files.items():
            filepath = os.path.join(YOLO_DIR, filename)
            if not os.path.exists(filepath):
                st.warning(f"Missing required file: {filename}")
                return False
            
            file_size = os.path.getsize(filepath)
            if file_size < min_size:
                st.warning(f"File {filename} appears to be incomplete (size: {file_size/1024:.1f}KB, required: {min_size/1024:.1f}KB)")
                return False
            else:
                st.success(f"Verified {filename} (size: {file_size/1024:.1f}KB)")
        
        return True
    except Exception as e:
        st.error(f"Error during file verification: {str(e)}")
        return False

def initialize_yolo():
    """YOLO 모델 초기화"""
    try:
        # 파일 존재 여부 확인
        if not verify_yolo_files():
            st.warning("YOLO 모델 파일이 없거나 불완전합니다. 다운로드가 필요합니다.")
            if st.button("YOLO 파일 다운로드"):
                if not download_yolo_files():
                    st.error("파일 다운로드에 실패했습니다.")
                    return None, None, None

        # 파일이 있으면 모델 로드 시작
        weights_path = os.path.join(YOLO_DIR, WEIGHTS_FILENAME)
        cfg_path = os.path.join(YOLO_DIR, "yolov4.cfg")
        names_path = os.path.join(YOLO_DIR, "coco.names")
        
        with st.spinner("YOLO 모델을 로드하는 중..."):
            try:
                net = cv2.dnn.readNet(weights_path, cfg_path)
                if net is None:
                    raise Exception("Failed to load YOLO network")
                
                layer_names = net.getLayerNames()
                unconnected_layers = net.getUnconnectedOutLayers()
                
                if isinstance(unconnected_layers, np.ndarray):
                    output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]
                elif isinstance(unconnected_layers, list):
                    output_layers = [layer_names[int(i[0]) - 1] if isinstance(i, (list, np.ndarray)) 
                                   else layer_names[int(i) - 1] for i in unconnected_layers]
                else:
                    output_layers = [layer_names[int(unconnected_layers) - 1]]
                
                with open(names_path, "r") as f:
                    classes = [line.strip() for line in f.readlines()]
                
                st.success("YOLO 모델 로드 완료!")
                return net, output_layers, classes
                
            except Exception as e:
                st.error(f"YOLO 모델 로드 중 오류: {str(e)}")
                return None, None, None
            
    except Exception as e:
        st.error(f"YOLO 모델 초기화 오류: {str(e)}")
        return None, None, None

def create_stable_tracker():
    """단순화된 트래커 생성"""
    try:
        class SimpleTracker:
            def __init__(self):
                self.bbox = None
                self.last_center = None
                
            def init(self, frame, bbox):
                self.bbox = tuple(map(int, bbox))
                self.last_center = (
                    int(bbox[0] + bbox[2]/2),
                    int(bbox[1] + bbox[3]/2)
                )
                return True
                
            def update(self, frame):
                if self.bbox is None:
                    return False, None
                
                # HSV 색상 기반 추적 시도
                try:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    x, y, w, h = self.bbox
                    
                    # 현재 박스 영역의 HSV 평균값 계산
                    roi = hsv[y:y+h, x:x+w]
                    avg_hue = np.mean(roi[:,:,0])
                    
                    # HSV 범위 설정
                    lower = np.array([max(0, avg_hue-10), 50, 50])
                    upper = np.array([min(180, avg_hue+10), 255, 255])
                    
                    # 마스크 생성
                    mask = cv2.inRange(hsv, lower, upper)
                    mask = cv2.erode(mask, None, iterations=2)
                    mask = cv2.dilate(mask, None, iterations=2)
                    
                    # 컨투어 찾기
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        M = cv2.moments(c)
                        
                        if M["m00"] != 0:
                            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                            self.last_center = center
                            self.bbox = (
                                int(center[0] - self.bbox[2]/2),
                                int(center[1] - self.bbox[3]/2),
                                self.bbox[2],
                                self.bbox[3]
                            )
                            return True, self.bbox
                            
                except Exception:
                    pass
                
                # 추적 실패 시 이전 위치 반환
                return True, self.bbox
        
        tracker = SimpleTracker()
        st.success("단순 트래커가 성공적으로 생성되었습니다.")
        return tracker
        
    except Exception as e:
        st.error(f"트래커 생성 실패: {str(e)}")
        return None
        
def detect_ball(frame, lower_color, upper_color, min_radius, max_radius):
    """색상 기반 공 검출"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if min_radius < radius < max_radius:
            return (int(x), int(y), int(radius)), center
    
    return None, None

def track_ball(frame, tracker, bbox, lower_color, upper_color, min_radius, max_radius):
    """공 추적 함수 - 중심점 검출 개선"""
    # 색상 기반 공 검출
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # 노이즈 제거
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    if len(contours) > 0:
        # 가장 큰 컨투어 찾기
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        # 모멘트를 이용한 중심점 계산
        M = cv2.moments(c)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            if min_radius < radius < max_radius:
                # 원 그리기
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # 바운딩 박스 업데이트
                bbox_size = int(radius * 2)
                bbox = (
                    int(center[0] - bbox_size/2),
                    int(center[1] - bbox_size/2),
                    bbox_size,
                    bbox_size
                )
                tracker.init(frame, bbox)
    
    # 트래커 업데이트
    if center is None:
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            center = (x + w//2, y + h//2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            bbox = (x, y, w, h)
    
    return frame, center, bbox

def calculate_dynamic_correction(bbox_size, pixel_distance, real_distance):
    """동적 보정 계수 계산"""
    try:
        base_correction = real_distance / pixel_distance
        
        # 공의 크기 변화에 따른 보정 (기준 크기 대비 현재 크기 비율)
        if bbox_size > 0:
            # 기준 크기는 첫 프레임의 bbox 크기
            reference_size = 50  # 기본값, 실제 첫 프레임의 크기로 조정 가능
            size_factor = reference_size / bbox_size
            # 크기 변화가 너무 극단적이지 않도록 제한
            size_factor = max(0.5, min(1.5, size_factor))
        else:
            size_factor = 1.0
            
        # 최종 보정 계수 계산
        correction_factor = base_correction * size_factor
        
        # 보정 계수가 너무 극단적이지 않도록 제한
        correction_factor = max(0.3, min(3.0, correction_factor))
        
        return correction_factor
        
    except Exception as e:
        st.warning(f"보정 계수 계산 중 오류: {str(e)}")
        return 1.0  # 오류 시 기본값 반환

def filter_speed(speed_queue, speeds):
    """개선된 속도 필터링"""
    try:
        if not speed_queue:
            return 0
            
        # 이동 평균 계산    
        avg_speed = sum(speed_queue) / len(speed_queue)
        
        # 급격한 변화 감지 및 보정
        if speeds:
            last_speed = speeds[-1]
            
            # 변화율 계산 (0에 의한 나눗셈 방지)
            change_rate = abs(avg_speed - last_speed) / (last_speed + 1e-6)
            
            # 급격한 변화 시 보정 (30% 이상 변화)
            if change_rate > 0.3:
                # 이전 속도와 현재 속도의 가중 평균
                return last_speed * 0.7 + avg_speed * 0.3
                
        return avg_speed
        
    except Exception as e:
        st.warning(f"속도 필터링 중 오류: {str(e)}")
        return speeds[-1] if speeds else 0  # 오류 시 이전 속도 반환


def calculate_frame_speed(positions_queue, fps, pixels_per_meter, bbox_size=None):
    try:
        first_frame, first_pos = positions_queue[0]
        last_frame, last_pos = positions_queue[-1]
        
        time_diff = (last_frame - first_frame) / fps
        if time_diff <= 0:
            return None
            
        # Pixel distance calculation
        pixel_distance = np.sqrt(
            (last_pos[0] - first_pos[0])**2 + 
            (last_pos[1] - first_pos[1])**2
        )
        
        # Debugging print
        print(f"Pixel Distance: {pixel_distance}, Pixels per Meter: {pixels_per_meter}")

        # Distance in meters using the adjusted pixels_per_meter
        distance_meters = pixel_distance / pixels_per_meter
        
        # Speed calculation
        speed = distance_meters / time_diff
        
        # Debugging print
        print(f"Speed (m/s): {speed}, Time Diff: {time_diff}")

        # Filtering unreasonable speeds
        return speed if speed <= 50 else None
        
    except Exception:
        print(f"Error in speed calculation: {e}")
        return None

        
def calculate_filtered_speed(speed_queue, speeds):
    """속도 필터링 및 평균 계산"""
    try:
        avg_speed = sum(speed_queue) / len(speed_queue)
        
        # 급격한 속도 변화 필터링
        if speeds and abs(avg_speed - speeds[-1]) > 10:
            return speeds[-1]
            
        return avg_speed
        
    except Exception:
        return speeds[-1] if speeds else 0

def is_significant_frame(current_speed, speeds):
    """중요 프레임 판단 (최고/최저 속도 등)"""
    if not speeds:
        return True
    
    return (
        abs(current_speed - speeds[-1]) > 2 or  # 급격한 속도 변화
        current_speed > max(speeds) or  # 최고 속도
        current_speed < min(speeds)  # 최저 속도
    )


def rgb_to_hsv(r, g, b):
    """RGB to HSV 변환"""
    r, g, b = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 179), int(s * 255), int(v * 255)

def update_charts(frames, speeds, speed_chart, frame_count, graph_color, 
                 is_final=False, frame_images=None, ball_positions=None, fps=30):
    """차트 업데이트 - 디스플레이 수정"""
    try:
        # 기본 통계 계산
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        min_speed = np.min(speeds)
        total_time = max([frame/fps for frame in frames])
        
        if is_final:
            # 통계 표시
            st.markdown("### 전체 통계")
            cols = st.columns(4)
            with cols[0]:
                st.metric("평균 속도", f"{avg_speed:.2f} m/s")
            with cols[1]:
                st.metric("최대 속도", f"{max_speed:.2f} m/s")
            with cols[2]:
                st.metric("최소 속도", f"{min_speed:.2f} m/s")
            with cols[3]:
                st.metric("총 분석 시간", f"{total_time:.2f} s")

            # 그래프와 이미지를 위한 열 생성
            graph_col, images_col = st.columns([2, 1])
            
            with graph_col:
                # 그래프 생성
                fig = go.Figure()
                
                # 메인 속도 라인
                fig.add_trace(go.Scatter(
                    x=[frame/fps for frame in frames],
                    y=speeds,
                    mode='lines+markers',
                    name='Speed (m/s)',
                    line=dict(
                        color=graph_color,
                        width=2
                    ),
                    marker=dict(
                        size=4
                    ),
                    hovertemplate='Time: %{x:.2f}s<br>Speed: %{y:.2f} m/s<extra></extra>'
                ))
                
                # 레이아웃 설정
                fig.update_layout(
                    title="Ball Speed Analysis",
                    xaxis_title="Time (s)",
                    yaxis_title="Speed (m/s)",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    showlegend=True,
                    height=500,
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='lightgrey',
                        showline=True,
                        linewidth=1,
                        linecolor='black'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='lightgrey',
                        showline=True,
                        linewidth=1,
                        linecolor='black',
                        range=[0, max_speed * 1.1]  # y축 범위 설정
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # CSV 다운로드
                df = pd.DataFrame({
                    'Time (s)': [frame/fps for frame in frames],
                    'Frame': frames,
                    'Speed (m/s)': speeds
                })
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "속도 데이터 다운로드 (CSV)",
                    csv,
                    "ball_speed_data.csv",
                    "text/csv",
                    key='download-csv-results'
                )
            
            with images_col:
                # 최고/최저 속도 프레임 찾기
                max_speed_indices = [i for i, s in enumerate(speeds) if abs(s - max_speed) < 0.01]
                min_speed_indices = [i for i, s in enumerate(speeds) if abs(s - min_speed) < 0.01]
                
                if max_speed_indices and frame_images:
                    st.markdown(f"#### 최고 속도: {max_speed:.2f} m/s")
                    for idx in max_speed_indices[:3]:  # 최대 3개까지
                        frame_num = frames[idx]
                        if frame_num in frame_images:
                            st.markdown(f"시간: {frame_num/fps:.2f}초")
                            st.image(frame_images[frame_num], channels="BGR", use_column_width=True)
                
                if min_speed_indices and frame_images:
                    st.markdown(f"#### 최저 속도: {min_speed:.2f} m/s")
                    for idx in min_speed_indices[:3]:  # 최대 3개까지
                        frame_num = frames[idx]
                        if frame_num in frame_images:
                            st.markdown(f"시간: {frame_num/fps:.2f}초")
                            st.image(frame_images[frame_num], channels="BGR", use_column_width=True)
        
        else:
            # 실시간 업데이트용 간단한 그래프
            last_100_frames = frames[-100:]
            last_100_speeds = speeds[-100:]
            
            fig = go.Figure(go.Scatter(
                x=[frame/fps for frame in last_100_frames],
                y=last_100_speeds,
                mode='lines+markers',
                line=dict(color=graph_color)
            ))
            fig.update_layout(
                title="Real-time Speed",
                xaxis_title="Time (s)",
                yaxis_title="Speed (m/s)",
                height=300
            )
            speed_chart.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"차트 업데이트 중 오류 발생: {str(e)}")
        st.error(traceback.format_exc())


def show_frame_analysis(frame_num, frames, speeds, images, positions):
    """프레임 분석 표시 - m/s 단위로 수정"""
    available_frames = sorted(list(images.keys()))
    nearest_frame = min(available_frames, key=lambda x: abs(x - frame_num))
    
    if nearest_frame in images:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### Frame {nearest_frame}")
            current_speed = speeds[frames.index(nearest_frame)]
            frame_image = images[nearest_frame]
            st.image(frame_image, channels="BGR", 
                    caption=f"Speed: {current_speed:.2f} m/s")
            
            # 속도 분석
            st.markdown("#### 속도 분석")
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            speed_diff = current_speed - avg_speed
            
            st.write(f"- 현재 속도: {current_speed:.2f} m/s")
            st.write(f"- 평균 대비: {speed_diff:+.2f} m/s")
            st.write(f"- 최대 대비: {(current_speed/max_speed*100):.1f}%")
        
        with col2:
            st.markdown("#### 공의 궤적")
            trajectory_img = images[nearest_frame].copy()
            
            for i in range(max(0, nearest_frame-5), min(nearest_frame+6, max(frames)+1)):
                if i in positions:
                    pos = positions[i]
                    color = (0, 255, 0) if i < nearest_frame else\
                           (0, 0, 255) if i > nearest_frame else\
                           (255, 0, 0)
                    cv2.circle(trajectory_img, pos, 3, color, -1)
            
            st.image(trajectory_img, channels="BGR", 
                    caption="Green: Past, Red: Current, Blue: Future")
            
            if nearest_frame in positions:
                x, y = positions[nearest_frame]
                st.write(f"- 공의 위치: ({x}, {y}) pixels")

    # CSV 다운로드 버튼
    df = pd.DataFrame({
        'Frame': frames,
        'Speed (km/h)': speeds
    })
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "속도 데이터 다운로드 (CSV)",
        csv,
        "ball_speed_data.csv",
        "text/csv",
        key='download-csv-results'
    )

def select_color_from_image(frame):
    """이미지에서 클릭으로 색상 선택"""
    if 'color_selected' not in st.session_state:
        st.session_state.color_selected = False

    st.write("아래 이미지를 마우스로 클릭하여 공의 색상을 선택하세요.")
    
    # BGR에서 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    
    # 상태 초기화
    if 'click_x' not in st.session_state:
        st.session_state.click_x = width // 2
    if 'click_y' not in st.session_state:
        st.session_state.click_y = height // 2
    if 'selected_color' not in st.session_state:
        st.session_state.selected_color = None
    if 'lower_color' not in st.session_state:
        st.session_state.lower_color = None
    if 'upper_color' not in st.session_state:
        st.session_state.upper_color = None
    if 'color_tolerance' not in st.session_state:
        st.session_state.color_tolerance = 20

    col1, col2 = st.columns([3, 1])
    
    with col1:
        from streamlit_image_coordinates import streamlit_image_coordinates
        
        frame_display = frame_rgb.copy()
        if st.session_state.selected_color is not None:
            # 이전 선택 위치 표시
            x, y = st.session_state.click_x, st.session_state.click_y
            marker_size = 10
            cv2.line(frame_display, (x - marker_size, y), (x + marker_size, y), (0, 255, 0), 2)
            cv2.line(frame_display, (x, y - marker_size), (x, y + marker_size), (0, 255, 0), 2)
        
        clicked_coords = streamlit_image_coordinates(
            frame_display,
            key="color_picker"
        )
        
        if clicked_coords:
            x, y = clicked_coords["x"], clicked_coords["y"]
            st.session_state.click_x = x
            st.session_state.click_y = y
            st.session_state.selected_color = frame[y, x]
            
            # 현재 선택 위치 표시
            frame_with_marker = frame_rgb.copy()
            marker_size = 10
            cv2.line(frame_with_marker, (x - marker_size, y), (x + marker_size, y), (0, 255, 0), 2)
            cv2.line(frame_with_marker, (x, y - marker_size), (x, y + marker_size), (0, 255, 0), 2)
            st.image(frame_with_marker, caption="클릭하여 색상 선택", use_column_width=True)

    with col2:
        if st.session_state.selected_color is not None:
            b, g, r = st.session_state.selected_color
            st.write("선택한 색상:")
            color_display = np.zeros((100, 100, 3), dtype=np.uint8)
            color_display[:] = (r, g, b)
            st.image(color_display)
            
            h, s, v = rgb_to_hsv(r, g, b)
            st.session_state.color_tolerance = st.slider(
                "색상 허용 범위", 
                0, 50, 
                st.session_state.color_tolerance
            )
            
            # HSV 색상 범위 업데이트
            st.session_state.lower_color = np.array([
                max(0, h - st.session_state.color_tolerance), 
                max(0, s - 50), 
                max(0, v - 50)
            ])
            st.session_state.upper_color = np.array([
                min(179, h + st.session_state.color_tolerance), 
                min(255, s + 50), 
                min(255, v + 50)
            ])
            
            # 마스크 미리보기
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, st.session_state.lower_color, st.session_state.upper_color)
            mask_preview = cv2.bitwise_and(frame, frame, mask=mask)
            mask_preview_rgb = cv2.cvtColor(mask_preview, cv2.COLOR_BGR2RGB)
            st.image(mask_preview_rgb, caption="마스크 미리보기")
            
            # BGR, HSV 값 표시
            st.write(f"BGR 값: ({b}, {g}, {r})")
            st.write(f"HSV 값: ({h}, {s}, {v})")
            
            # 색상 선택 확정 버튼
            if st.button("이 색상으로 선택"):
                st.session_state.color_selected = True
                return (tuple(st.session_state.selected_color), 
                        st.session_state.lower_color, 
                        st.session_state.upper_color, 
                        (st.session_state.click_x, st.session_state.click_y))

    if st.session_state.color_selected:
        return (tuple(st.session_state.selected_color), 
                st.session_state.lower_color, 
                st.session_state.upper_color, 
                (st.session_state.click_x, st.session_state.click_y))
    
    return None, None, None, None

def detect_ball_with_yolo(frame, net, output_layers, classes):
    """YOLO를 사용한 공 검출"""
    try:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        net.setInput(blob)
        outs = net.forward(output_layers)

        best_confidence = 0
        best_box = None

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # 기준값 수정 가능
                    try:
                        class_name = classes[class_id]
                        if class_name == "sports ball" and confidence > best_confidence:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            # 범위 검사 추가
                            x = max(0, min(x, width - w))
                            y = max(0, min(y, height - h))
                            w = min(w, width - x)
                            h = min(h, height - y)

                            best_box = (x, y, w, h)
                            best_confidence = confidence
                    except IndexError:
                        continue

        if best_box is not None:
            st.success(f"YOLO가 공을 감지했습니다! (신뢰도: {best_confidence:.2f})")
            return best_box
        return None  # 경고 메시지 제거
            
    except Exception as e:
        st.error(f"YOLO 감지 중 오류 발생: {str(e)}")
        return None
        

def resize_frame(frame, target_width=384):
    """영상의 종횡비를 유지하면서 안전하게 크기 조정"""
    try:
        if frame is None:
            return None
            
        height, width = frame.shape[:2]
        if width == 0 or height == 0:
            return frame
            
        # 종횡비 계산
        aspect_ratio = float(width) / float(height)
        
        # 새로운 높이 계산
        target_height = int(round(target_width / aspect_ratio))
        
        # 최소 크기 보장
        if target_height < 1:
            target_height = 1
        
        # 리사이즈 수행
        resized = cv2.resize(frame, (target_width, target_height), 
                           interpolation=cv2.INTER_LINEAR)
        
        return resized
        
    except Exception as e:
        st.error(f"프레임 리사이즈 중 오류 발생: {str(e)}")
        return frame  # 오류 발생시 원본 반환

def process_video(video_path, initial_bbox, pixels_per_meter, net, output_layers, 
                 classes, lower_color, upper_color, graph_color):
    """비디오 처리 및 분석 - 스크린샷 기능 추가"""
    try:
        MEMORY_WINDOW = 300
        frame_images = {}
        ball_positions = {}
        
        # 최고/최저 속도 프레임 저장을 위한 변수
        max_speed_frame = None
        min_speed_frame = None
        max_speed_value = float('-inf')
        min_speed_value = float('inf')
        
        video = None 
        try:
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise ValueError("비디오를 열 수 없습니다")
                
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Streamlit 레이아웃 설정
            col1, col2 = st.columns([2, 1])
            
            with col1:
                video_frame = st.empty()
            with col2:
                real_time_speed = st.empty()
                speed_chart = st.empty()

            frames = []
            speeds = []
            
            # 트래커 초기화
            tracker = create_stable_tracker()
            if tracker is None:
                raise ValueError("트래커를 생성할 수 없습니다")

            # bbox 초기화 - 중요!
            bbox = initial_bbox
            
            # 첫 프레임에서 트래커 초기화
            ret, first_frame = video.read()
            if not ret:
                raise ValueError("첫 프레임을 읽을 수 없습니다")
                
            first_frame = resize_frame(first_frame)
            if first_frame is None:
                raise ValueError("프레임 리사이즈 실패")
                
            # YOLO로 첫 프레임에서 공 감지 시도
            detected_bbox = detect_ball_with_yolo(first_frame, net, output_layers, classes)
            if detected_bbox is not None:
                bbox = detected_bbox
                st.success("YOLO가 공을 감지했습니다!")
                
            # 트래커 초기화
            tracker.init(first_frame, bbox)

            # 속도 계산을 위한 변수들
            speed_queue = deque(maxlen=5)
            positions_queue = deque(maxlen=5)
            frame_count = 0
            
            # 진행 상태 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 비디오 처리 시작 지점으로 되감기
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # 프레임 처리 루프
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                    
                try:
                    # 메모리 관리: 오래된 프레임 제거
                    if frame_count > MEMORY_WINDOW:
                        old_frame = frame_count - MEMORY_WINDOW
                        if old_frame in frame_images:
                            del frame_images[old_frame]
                        if old_frame in ball_positions:
                            del ball_positions[old_frame]
                    
                    frame = resize_frame(frame)
                    if frame is None:
                        raise ValueError("프레임 리사이즈 실패")
                        
                    # bbox가 설정되어 있는지 확인
                    if bbox is None:
                        raise ValueError("bbox가 설정되지 않았습니다")
                        
                    processed_frame, center, bbox = track_ball(
                        frame, tracker, bbox, lower_color, upper_color, 10, 50
                    )
                    
                    if center:
                        positions_queue.append((frame_count, center))
                        
                        if len(positions_queue) >= 2:
                            # 속도 계산 및 필터링
                            speed = calculate_frame_speed(
                                positions_queue, 
                                fps, 
                                pixels_per_meter,
                                bbox[2] if bbox else None
                            )
                            
                            if speed is not None:
                                speed_queue.append(speed)
                                avg_speed = filter_speed(speed_queue, speeds)
                                
                                speeds.append(avg_speed)
                                frames.append(frame_count)
                                ball_positions[frame_count] = center

                                # 최고/최저 속도 프레임 저장
                                if avg_speed > max_speed_value:
                                    max_speed_value = avg_speed
                                    max_speed_frame = processed_frame.copy()
                                    frame_images[frame_count] = processed_frame.copy()
                                
                                if avg_speed < min_speed_value:
                                    min_speed_value = avg_speed
                                    min_speed_frame = processed_frame.copy()
                                    frame_images[frame_count] = processed_frame.copy()
                                
                                # 중요 프레임 저장
                                if is_significant_frame(avg_speed, speeds):
                                    frame_images[frame_count] = processed_frame.copy()
                                
                                real_time_speed.markdown(
                                    f"### Current Speed\n{avg_speed:.2f} m/s"
                                )
                    
                    video_frame.image(processed_frame, channels="BGR", use_column_width=False)
                    
                except Exception as e:
                    st.warning(f"프레임 {frame_count} 처리 중 오류: {str(e)}")
                    # bbox 복구 시도
                    if bbox is None:
                        bbox = initial_bbox
                    continue
                
                # 진행률 업데이트
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                progress_bar.progress(progress)
                status_text.text(f"처리 중: {frame_count}/{total_frames} 프레임 ({progress}%)")

            # 프레임 처리 루프가 끝난 후 결과 표시
            if speeds:
                st.markdown("### 📊 분석 결과")
                
                # 통계 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("평균 속도", f"{np.mean(speeds):.2f} m/s")
                with col2:
                    st.metric("최고 속도", f"{max_speed_value:.2f} m/s")
                with col3:
                    st.metric("최저 속도", f"{min_speed_value:.2f} m/s")
                with col4:
                    total_time = frame_count / fps
                    st.metric("총 분석 시간", f"{total_time:.2f} s")
                
                # 속도 그래프와 스크린샷을 나란히 표시
                graph_col, images_col = st.columns([2, 1])
                
                with graph_col:
                    # 그래프 표시 로직...
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[frame/fps for frame in frames],
                        y=speeds,
                        mode='lines+markers',
                        name='Speed (m/s)',
                        line=dict(color=graph_color, width=2),
                        marker=dict(size=4),
                        hovertemplate='Time: %{x:.2f}s<br>Speed: %{y:.2f} m/s<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Ball Speed Analysis",
                        xaxis_title="Time (s)",
                        yaxis_title="Speed (m/s)",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='black'),
                        showlegend=True,
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with images_col:
                    # 최고 속도 프레임 표시
                    if max_speed_frame is not None:
                        st.markdown(f"#### 🔼 최고 속도: {max_speed_value:.2f} m/s")
                        st.image(max_speed_frame, channels="BGR", use_column_width=True)
                    
                    # 최저 속도 프레임 표시
                    if min_speed_frame is not None:
                        st.markdown(f"#### 🔽 최저 속도: {min_speed_value:.2f} m/s")
                        st.image(min_speed_frame, channels="BGR", use_column_width=True)
                
                # CSV 다운로드 버튼
                df = pd.DataFrame({
                    'Time (s)': [frame/fps for frame in frames],
                    'Frame': frames,
                    'Speed (m/s)': speeds
                })
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 속도 데이터 다운로드 (CSV)",
                    csv,
                    "ball_speed_data.csv",
                    "text/csv",
                    key='download-csv-results'
                )
            
            else:
                st.warning("속도 데이터가 기록되지 않았습니다")
                
        finally:
            if video is not None:
                video.release()
    
    except Exception as e:
        st.error(f"비디오 처리 중 심각한 오류 발생: {str(e)}")
        st.error(traceback.format_exc())
        
    finally:
        # 메모리 정리
        frame_images.clear()
        ball_positions.clear()
        
def calculate_slopes(frames, speeds, slope_range=(9.4, 9.8)):
    segments = []
    slopes = []
    remaining_indices = list(range(len(speeds)))

    while remaining_indices:
        current_min_idx = min(remaining_indices, key=lambda i: speeds[i])
        current_max_idx = max(remaining_indices, key=lambda i: speeds[i])

        if current_min_idx < current_max_idx:
            segment = remaining_indices[current_min_idx:current_max_idx + 1]
            slope = (speeds[current_max_idx] - speeds[current_min_idx]) / \
                    (frames[current_max_idx] - frames[current_min_idx])
            
            if slope_range[0] <= slope <= slope_range[1]:
                segments.append(segment)
                slopes.append(slope)
                for idx in segment:
                    remaining_indices.remove(idx)
            else:
                break
        else:
            break

    return segments, slopes



def process_uploaded_video(uploaded_file, net, output_layers, classes):
    """업로드된 비디오 처리 - 파일 객체와 경로 문자열 모두 지원"""
    try:
        if 'video_settings' not in st.session_state:
            st.session_state.video_settings = {}

        # 입력이 문자열(경로)인 경우와 파일 객체인 경우를 구분하여 처리
        if isinstance(uploaded_file, str):
            video_path = uploaded_file
        else:
            # 파일 객체인 경우 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name

        try:
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise ValueError("비디오를 열 수 없습니다.")

            ret, first_frame = video.read()
            video.release()
            
            if not ret or first_frame is None:
                raise ValueError("비디오 프레임을 읽을 수 없습니다.")
                
            # 첫 프레임 크기 조정
            first_frame = resize_frame(first_frame)
            if first_frame is None:
                raise ValueError("프레임 처리 실패")
                
            # 비디오 표시 (업로드된 파일인 경우만)
            if not isinstance(uploaded_file, str):
                st.video(video_path)
            
            height, width = first_frame.shape[:2]
            
            # 그래프 설정
            graph_color = st.radio(
                "그래프 색상:",
                ('white', 'black'),
                key='graph_color'
            )
            
            # 색상 선택
            selected_color, lower_color, upper_color, click_pos = select_color_from_image(first_frame)
            if not any([selected_color is None, lower_color is None, upper_color is None, click_pos is None]):
                # video settings 업데이트
                st.session_state.video_settings.update({
                    'selected_color': selected_color,
                    'lower_color': lower_color,
                    'upper_color': upper_color,
                    'click_pos': click_pos,
                    'graph_color': graph_color
                })
                
                if all(k in st.session_state.video_settings for k in 
                      ['selected_color', 'lower_color', 'upper_color', 'click_pos']):
                    # 거리 측정을 위한 점 선택
                    settings_col1, settings_col2 = st.columns(2)
                    
                    with settings_col1:
                        x1 = st.slider('첫 번째 점 X 좌표', 0, width, 
                            st.session_state.video_settings.get('x1', width // 4))
                        y1 = st.slider('첫 번째 점 Y 좌표', 0, height, 
                            st.session_state.video_settings.get('y1', height // 2))
                    
                    with settings_col2:
                        x2 = st.slider('두 번째 점 X 좌표', 0, width, 
                            st.session_state.video_settings.get('x2', 3 * width // 4))
                        y2 = st.slider('두 번째 점 Y 좌표', 0, height, 
                            st.session_state.video_settings.get('y2', height // 2))
                    
                    # 프레임에 정보 표시
                    frame_with_info = first_frame.copy()
                    cv2.circle(frame_with_info, (x1, y1), 5, (0, 255, 0), -1)
                    cv2.circle(frame_with_info, (x2, y2), 5, (0, 255, 0), -1)
                    cv2.line(frame_with_info, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 선택한 색상 위치 표시
                    click_x, click_y = st.session_state.video_settings['click_pos']
                    selected_color = st.session_state.video_settings['selected_color']
                    bbox_size = 50
                    bbox_x = max(0, click_x - bbox_size//2)
                    bbox_y = max(0, click_y - bbox_size//2)
                    bbox_w = min(bbox_size, width - bbox_x)
                    bbox_h = min(bbox_size, height - bbox_y)
                    
                    cv2.rectangle(frame_with_info, (bbox_x, bbox_y), 
                                (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 255), 2)
                    cv2.circle(frame_with_info, (click_x, click_y), 5, tuple(map(int, selected_color)), -1)
                    
                    st.image(frame_with_info, channels="BGR", use_column_width=False)
                    
                    # 실제 거리 입력
                    real_distance = st.number_input(
                        "선택한 두 점 사이의 실제 거리(미터)를 입력해주세요:", 
                        min_value=0.1, 
                        value=st.session_state.video_settings.get('real_distance', 1.0), 
                        step=0.1
                    )
                    
                    # 설정값 저장
                    st.session_state.video_settings.update({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'real_distance': real_distance
                    })
                    
                    # New pixels_per_meter value
                    pixels_per_meter = 48.38 # Adjusted based on the ideal calculation
                    
                    # 분석 시작 버튼
                    if st.button('영상 내 공 추적 및 분석 시작하기'):
                        initial_bbox = (bbox_x, bbox_y, bbox_w, bbox_h)
                        st.write("Processing video...")
                        try:
                            process_video(video_path, initial_bbox, pixels_per_meter, 
                                       net, output_layers, classes, 
                                       st.session_state.video_settings['lower_color'],
                                       st.session_state.video_settings['upper_color'],
                                       st.session_state.video_settings['graph_color'])
                        except Exception as e:
                            st.error(f"비디오 처리 중 오류 발생: {str(e)}")
                            st.error(traceback.format_exc())
                else:
                    st.warning("색상을 선택해주세요.")
                    
        finally:
            # 임시 파일 정리 (업로드된 파일인 경우만)
            if not isinstance(uploaded_file, str) and 'video_path' in locals():
                try:
                    os.unlink(video_path)
                except:
                    pass
                    
    except Exception as e:
        st.error(f"비디오 처리 중 오류 발생: {str(e)}")
        st.error(traceback.format_exc())


def main():
    """메인 함수 - 초기화 과정 표시 개선"""
    try:
        # 세션 상태 초기화
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.analysis_frames = []
            st.session_state.analysis_speeds = []
            st.session_state.analysis_images = {}
            st.session_state.analysis_positions = {}
            st.session_state.selected_frame = None
            st.session_state.video_settings = {}
        
        st.write("### 🎯 시작하기")
        st.write("이 앱은 비디오에서 공의 움직임을 추적하고 속도를 분석합니다.")
        
        # YOLO 디렉토리 생성
        os.makedirs(YOLO_DIR, exist_ok=True)
        
        # YOLO 파일 존재 여부 확인
        has_cfg = os.path.exists(os.path.join(YOLO_DIR, "yolov4.cfg"))
        has_weights = os.path.exists(os.path.join(YOLO_DIR, "yolov4.weights"))
        has_names = os.path.exists(os.path.join(YOLO_DIR, "coco.names"))
        
        if not all([has_cfg, has_weights, has_names]):
            st.warning("⚠️ YOLO 모델 파일이 없습니다. 아래 버튼을 클릭하여 필요한 파일을 다운로드해주세요.")
            
            if st.button("🔽 YOLO 파일 다운로드", key="download_yolo"):
                with st.spinner("YOLO 파일을 다운로드하는 중... (약 2-3분 소요)"):
                    success = download_yolo_files()
                    if success:
                        st.success("✅ YOLO 파일 다운로드 완료!")
                        st.balloons()
                        st.experimental_rerun()
                    else:
                        st.error("❌ YOLO 파일 다운로드 실패. 다시 시도해주세요.")
            return
        else:
            st.success("✅ YOLO 모델 파일이 준비되었습니다.")

        # YOLO 모델 초기화
        with st.spinner("🔄 YOLO 모델을 초기화하는 중..."):
            net, output_layers, classes = initialize_yolo()
            if not all([net, output_layers, classes]):
                st.error("❌ YOLO 모델 초기화 실패")
                return
            st.success("✅ YOLO 모델 초기화 완료!")

        # 구분선 추가
        st.markdown("---")
        
        # 비디오 업로드 섹션
        st.write("### 📹 비디오 업로드")
        st.write("분석할 비디오 파일을 선택해주세요.")
        
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "지원되는 형식: MP4, AVI, MOV", 
            type=['mp4', 'avi', 'mov'],
            key='video_upload'
        )
        
        if uploaded_file is not None:
            st.success(f"✅ 파일 '{uploaded_file.name}' 업로드 완료!")
            process_uploaded_video(uploaded_file, net, output_layers, classes)
                    
    except Exception as e:
        st.error(f"❌ 어플리케이션 실행 중 오류 발생: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
