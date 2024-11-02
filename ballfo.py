import cv2
import numpy as np
import streamlit as st
from collections import deque
import plotly.graph_objs as go
from scipy.signal import savgol_filter
import colorsys
import tempfile
import os
import gdown
import traceback
import urllib.request

# Streamlit 페이지 설정 (반드시 다른 st 명령어보다 먼저 와야 함)
st.set_page_config(
    page_title="공 추적 및 에너지 분석기",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'Ball Tracking and Energy Analysis Application'
    }
)

# 앱 제목
st.title('공 추적 및 에너지 분석기')

# 세션 상태 초기화
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

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

# YOLO 파일 다운로드 함수 수정
def download_yolo_files():
    """YOLO 모델 파일 다운로드"""
    try:
        os.makedirs(YOLO_DIR, exist_ok=True)
        
        # YOLOv4 cfg 파일 생성
        if not os.path.exists(os.path.join(YOLO_DIR, "yolov4.cfg")):
            st.info("Creating YOLOv4 configuration file...")
            if not save_yolov4_cfg():
                return False
            st.success("Created YOLOv4 configuration file")
        
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
        
        return verify_yolo_files()
    
    except Exception as e:
        st.error(f"Error in download process: {str(e)}")
        return False

def verify_yolo_files():
    """YOLO 파일들의 존재 여부와 무결성 확인"""
    required_files = {
        "yolov4.cfg": 10 * 1024,     # 최소 10KB (수정됨)
        "coco.names": 1024,          # 최소 1KB
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

def initialize_yolo():
    """YOLO 모델 초기화"""
    try:
        if not verify_yolo_files():
            st.warning("YOLO 모델 파일이 없거나 불완전합니다. 다운로드가 필요합니다.")
            if st.button("YOLO 파일 다운로드"):
                if not download_yolo_files():
                    st.error("파일 다운로드에 실패했습니다.")
                    return None, None, None
                if not verify_yolo_files():
                    st.error("다운로드된 파일이 올바르지 않습니다.")
                    return None, None, None
        
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
    """공 추적 함수"""
    color_detection, color_center = detect_ball(frame, lower_color, upper_color, min_radius, max_radius)
    
    success, box = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(v) for v in box]
        tracker_center = (x + w//2, y + h//2)
        
        if color_detection:
            color_x, color_y, color_radius = color_detection
            distance = np.sqrt((color_x - tracker_center[0])**2 + (color_y - tracker_center[1])**2)
            
            if distance > color_radius:
                new_bbox = (color_x - color_radius, color_y - color_radius, 
                          2*color_radius, 2*color_radius)
                tracker.init(frame, new_bbox)
                bbox = new_bbox
                center = color_center
            else:
                center = ((tracker_center[0] + color_center[0])//2, 
                         (tracker_center[1] + color_center[1])//2)
                bbox = (x, y, w, h)
        else:
            center = tracker_center
            bbox = (x, y, w, h)
    elif color_detection:
        color_x, color_y, color_radius = color_detection
        bbox = (color_x - color_radius, color_y - color_radius, 
               2*color_radius, 2*color_radius)
        tracker.init(frame, bbox)
        center = color_center
    else:
        center = None
        bbox = None
    
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if center:
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
    
    return frame, center, bbox

def calculate_speed(prev_pos, curr_pos, fps, pixels_per_meter):
    """속도 계산"""
    distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
    distance_meters = distance / pixels_per_meter
    speed = distance_meters * fps  # m/s
    return speed

def calculate_energy(speed, height, mass):
    """에너지 계산"""
    kinetic_energy = 0.5 * mass * (speed ** 2)
    potential_energy = mass * 9.81 * height
    mechanical_energy = kinetic_energy + potential_energy
    return kinetic_energy, potential_energy, mechanical_energy

def rgb_to_hsv(r, g, b):
    """RGB to HSV 변환"""
    r, g, b = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 179), int(s * 255), int(v * 255)

def update_charts(frames, speeds, ke, pe, me, speed_chart, energy_chart, frame_count):
    """차트 업데이트"""
    speed_fig = go.Figure(go.Scatter(x=frames[-100:], y=speeds[-100:], 
                                   mode='lines', name='Speed (km/h)'))
    speed_fig.update_layout(title="Speed over time (last 100 frames)", 
                          xaxis_title="Frame", yaxis_title="Speed (km/h)")
    speed_chart.plotly_chart(speed_fig, use_container_width=True, 
                           key=f"speed_chart_{frame_count}")
    
    energy_fig = go.Figure()
    energy_fig.add_trace(go.Scatter(x=frames[-100:], y=ke[-100:], 
                                  mode='lines', name='Kinetic Energy (J)'))
    energy_fig.add_trace(go.Scatter(x=frames[-100:], y=pe[-100:], 
                                  mode='lines', name='Potential Energy (J)'))
    energy_fig.add_trace(go.Scatter(x=frames[-100:], y=me[-100:], 
                                  mode='lines', name='Mechanical Energy (J)'))
    energy_fig.update_layout(title="Energy over time (last 100 frames)", 
                           xaxis_title="Frame", yaxis_title="Energy (J)")
    energy_chart.plotly_chart(energy_fig, use_container_width=True, 
                            key=f"energy_chart_{frame_count}")

def select_color_from_image(frame):
    """이미지에서 클릭으로 색상 선택"""
    st.write("아래 이미지를 마우스로 클릭하여 공의 색상을 선택하세요.")
    
    # BGR에서 RGB로 변환 (Streamlit 표시용)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    
    # 세션 상태 초기화
    if 'click_x' not in st.session_state:
        st.session_state.click_x = width // 2
    if 'click_y' not in st.session_state:
        st.session_state.click_y = height // 2
    if 'selected_color' not in st.session_state:
        st.session_state.selected_color = None

    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 이미지 품질 설정을 높게 설정하여 표시
        from streamlit_image_coordinates import streamlit_image_coordinates
        
        clicked_coords = streamlit_image_coordinates(
            frame_rgb,
            key="color_picker",
            width=width  # 원본 크기 유지
        )
        
        if clicked_coords:
            # 클릭 좌표 업데이트
            x, y = clicked_coords["x"], clicked_coords["y"]
            st.session_state.click_x = x
            st.session_state.click_y = y
            
            # BGR 형식으로 색상 저장
            st.session_state.selected_color = frame[y, x]
            
            # 클릭 위치 표시
            frame_with_marker = frame_rgb.copy()
            # 클릭 위치에 십자선 표시
            marker_size = 10
            marker_color = (0, 255, 0)  # RGB green
            cv2.line(frame_with_marker, (x - marker_size, y), (x + marker_size, y), marker_color, 2)
            cv2.line(frame_with_marker, (x, y - marker_size), (x, y + marker_size), marker_color, 2)
            
            # 선택 영역 확대
            zoom_size = 50
            x1, y1 = max(0, x - zoom_size), max(0, y - zoom_size)
            x2, y2 = min(width, x + zoom_size), min(height, y + zoom_size)
            zoomed_region = frame_rgb[y1:y2, x1:x2]
            
            st.image(frame_with_marker, caption="클릭하여 색상 선택", use_column_width=True)
            st.image(zoomed_region, caption="선택 영역 확대", width=200)
    
    with col2:
        if st.session_state.selected_color is not None:
            # 선택한 색상 표시
            b, g, r = st.session_state.selected_color
            st.write("선택한 색상:")
            color_display = np.zeros((100, 100, 3), dtype=np.uint8)
            color_display[:] = (r, g, b)  # RGB 형식으로 변환
            st.image(color_display)
            
            # HSV 색상 범위 설정
            h, s, v = rgb_to_hsv(r, g, b)
            color_tolerance = st.slider("색상 허용 범위", 0, 50, 20)
            
            # 마스크 미리보기
            lower_color = np.array([max(0, h - color_tolerance), max(0, s - 50), max(0, v - 50)])
            upper_color = np.array([min(179, h + color_tolerance), min(255, s + 50), min(255, v + 50)])
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_color, upper_color)
            mask_preview = cv2.bitwise_and(frame, frame, mask=mask)
            mask_preview_rgb = cv2.cvtColor(mask_preview, cv2.COLOR_BGR2RGB)
            st.image(mask_preview_rgb, caption="마스크 미리보기")
            
            # BGR 값과 HSV 값 표시
            st.write(f"BGR 값: ({b}, {g}, {r})")
            st.write(f"HSV 값: ({h}, {s}, {v})")
            
            # 선택 확인 버튼
            if st.button("이 색상으로 선택"):
                return (b, g, r), lower_color, upper_color, (st.session_state.click_x, st.session_state.click_y)
    
    return None, None, None, None

def process_uploaded_video(uploaded_file, net, output_layers, classes):
    """업로드된 비디오 처리"""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    
    video = cv2.VideoCapture(tfile.name)
    ret, first_frame = video.read()
    video.release()
    
    if ret:
        st.video(tfile.name)
        height, width = first_frame.shape[:2]
        
        # 색상 선택
        selected_color, lower_color, upper_color, click_pos = select_color_from_image(first_frame)
        
        # 수정된 검사 방식
        if selected_color is None or lower_color is None or upper_color is None or click_pos is None:
            st.warning("색상을 선택해주세요.")
            return
            
        # 거리 측정을 위한 점 선택
        st.write("알고 있는 실제 거리에 해당하는 두 점을 선택해주세요.")
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.slider('첫 번째 점 X 좌표', 0, width, width // 4, key="x1_slider")
            y1 = st.slider('첫 번째 점 Y 좌표', 0, height, height // 2, key="y1_slider")
        with col2:
            x2 = st.slider('두 번째 점 X 좌표', 0, width, 3 * width // 4, key="x2_slider")
            y2 = st.slider('두 번째 점 Y 좌표', 0, height, height // 2, key="y2_slider")
    
        # 높이 기준점 선택
        st.write("높이의 기준점(h=0)을 선택해주세요.")
        height_reference_y = st.slider('높이 기준점 Y 좌표', 0, height, height, key="height_ref_slider")
        height_reference = (0, height_reference_y)
        
        # 초기 바운딩 박스 설정 - 클릭한 위치 주변
        click_x, click_y = click_pos
        bbox_size = 50  # 초기 박스 크기
        bbox_x = max(0, click_x - bbox_size//2)
        bbox_y = max(0, click_y - bbox_size//2)
        bbox_w = min(bbox_size, width - bbox_x)
        bbox_h = min(bbox_size, height - bbox_y)
        
        initial_bbox = (bbox_x, bbox_y, bbox_w, bbox_h)
        
        # 프레임에 정보 표시
        frame_with_info = first_frame.copy()
        cv2.circle(frame_with_info, (x1, y1), 5, (0, 255, 0), -1)
        cv2.circle(frame_with_info, (x2, y2), 5, (0, 255, 0), -1)
        cv2.line(frame_with_info, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(frame_with_info, (0, height_reference_y), (width, height_reference_y), (255, 0, 0), 2)
        cv2.rectangle(frame_with_info, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 255), 2)
        cv2.circle(frame_with_info, click_pos, 5, selected_color, -1)  # 선택한 색상 위치 표시
        st.image(frame_with_info, channels="BGR", use_column_width=True)
        
        # 실제 거리 입력
        real_distance = st.number_input("선택한 두 점 사이의 실제 거리(미터)를 입력해주세요:", 
                                      min_value=0.1, value=1.0, step=0.1)
        
        # pixels_per_meter 계산
        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        pixels_per_meter = pixel_distance / real_distance
        st.write(f"계산된 pixels_per_meter: {pixels_per_meter:.2f}")
        
        # 공의 질량 입력
        mass = st.number_input("공의 질량(kg)을 입력해주세요:", 
                             min_value=0.1, value=0.1, step=0.1)
        
        if st.button('영상 내 공 추적 및 에너지 분석 시작하기'):
            st.write("Processing video...")
            try:
                process_video(tfile.name, initial_bbox, pixels_per_meter, mass, height_reference, 
                            net, output_layers, classes, lower_color, upper_color)
            except Exception as e:
                st.error(f"비디오 처리 중 오류 발생: {str(e)}")
                st.error(traceback.format_exc())
        
        try:
            os.unlink(tfile.name)
        except PermissionError:
            pass
    else:
        st.error("Failed to read the first frame of the video.")

def process_video(video_path, initial_bbox, pixels_per_meter, mass, height_reference, net, output_layers, classes, lower_color, upper_color):
    """비디오 처리 및 분석"""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 트래커 초기화
    tracker = create_stable_tracker()
    if tracker is None:
        st.error("트래커를 생성할 수 없습니다. 프로그램을 종료합니다.")
        return

    # YOLO로 첫 프레임에서 공 탐지
    ret, first_frame = video.read()
    if not ret:
        st.error("비디오 첫 프레임을 읽을 수 없습니다.")
        return

    try:
        bbox = detect_ball_with_yolo(first_frame, net, output_layers, classes)
        if bbox is None:
            st.warning("YOLO로 공을 감지할 수 없어 수동 설정된 위치를 사용합니다.")
            bbox = initial_bbox
        
        # bbox가 tuple이 아닌 경우 변환
        if not isinstance(bbox, tuple):
            bbox = tuple(bbox)
        
        st.info(f"초기 바운딩 박스: {bbox}")
        
        # 트래커 초기화 시도
        init_success = tracker.init(first_frame, bbox)
        if not init_success:
            st.warning("트래커 초기화 문제 발생 - 단순 추적으로 계속합니다.")
        else:
            st.success("트래커가 성공적으로 초기화되었습니다!")
            
    except Exception as e:
        st.error(f"트래커 초기화 중 오류 발생: {str(e)}")
        st.info("단순 추적으로 전환합니다.")
        tracker = create_stable_tracker()  # 새로운 트래커 생성 시도

    # 분석 변수 초기화
    prev_pos = None
    speed_queue = deque(maxlen=5)
    speeds = []
    kinetic_energies = []
    potential_energies = []
    mechanical_energies = []
    frames = []

    # Streamlit 디스플레이 요소
    progress_bar = st.progress(0)
    status_text = st.empty()
    video_frame = st.empty()
    speed_chart = st.empty()
    energy_chart = st.empty()

    frame_count = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        try:
            frame, center, bbox = track_ball(frame, tracker, bbox, lower_color, upper_color, 10, 50)
            
            if center:
                if prev_pos:
                    speed = calculate_speed(prev_pos, center, fps, pixels_per_meter)
                    speed_queue.append(speed)
                    avg_speed = sum(speed_queue) / len(speed_queue)
                    
                    h = (height_reference[1] - center[1]) / pixels_per_meter
                    ke, pe, me = calculate_energy(avg_speed, h, mass)
                    
                    # 속도 표시
                    cv2.putText(frame, f"Speed: {avg_speed*3.6:.2f} km/h", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 데이터 저장
                    speeds.append(avg_speed*3.6)
                    kinetic_energies.append(ke)
                    potential_energies.append(pe)
                    mechanical_energies.append(me)
                    frames.append(frame_count)
                
                prev_pos = center

            # 기준선 표시
            cv2.line(frame, (0, height_reference[1]), (width, height_reference[1]), (255, 0, 0), 2)
            
            # 프레임 표시
            video_frame.image(frame, channels="BGR", use_column_width=True)
            
            # 차트 업데이트 (30프레임마다)
            if frame_count % 30 == 0 and frames:
                update_charts(frames, speeds, kinetic_energies, potential_energies, 
                            mechanical_energies, speed_chart, energy_chart, frame_count)

        except Exception as e:
            st.error(f"프레임 {frame_count} 처리 중 오류 발생: {str(e)}")
            st.error(traceback.format_exc())

        frame_count += 1
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")

    video.release()
    status_text.text("Video processing completed!")

def main():
    """메인 실행 함수"""
    st.write(f"OpenCV 버전: {cv2.__version__}")
    
    # YOLO 모델 초기화
    net, output_layers, classes = initialize_yolo()
    if not all([net, output_layers, classes]):
        st.error("YOLO 모델 초기화 실패")
        return

    # 파일 업로드
    uploaded_file = st.file_uploader("비디오 파일을 선택하세요.", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        try:
            process_uploaded_video(uploaded_file, net, output_layers, classes)
        except Exception as e:
            st.error(f"비디오 처리 중 오류 발생: {str(e)}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
