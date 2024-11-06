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

# 앱 시작 부분에 CSS 수정
st.markdown("""
    <style>
    .stVideo {
        max-width: 384px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 앱 제목
st.title('객체(공) 속도 추적 프로그램_by ROHA')

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
    """속도 계산 - m/s 단위로 직접 반환"""
    distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
    distance_meters = distance / pixels_per_meter
    speed = distance_meters * fps  # m/s
    return speed  


def rgb_to_hsv(r, g, b):
    """RGB to HSV 변환"""
    r, g, b = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 179), int(s * 255), int(v * 255)

def update_charts(frames, speeds, speed_chart, frame_count, graph_color, trend_color, 
                 is_final=False, frame_images=None, ball_positions=None, fps=30):
    """차트 업데이트 - 안전한 최고/최저 속도 프레임 표시"""
    try:
        color = 'white' if graph_color == 'white' else 'black'
        trend_line_color = 'white' if trend_color == 'white' else 'black'
        
        # 프레임 번호를 시간(s)으로 변환
        times = [frame/fps for frame in frames]
        data = speeds if is_final else speeds[-100:]
        x_data = times if is_final else times[-100:]
        
        # 기본 그래프 생성
        speed_fig = go.Figure(go.Scatter(
            x=x_data, 
            y=data,
            mode='lines+markers',
            name='Speed (m/s)',
            line=dict(color=color),
            hovertemplate='Time: %{x:.2f} s<br>Speed: %{y:.2f} m/s<extra></extra>'
        ))
        
        # 추세선 추가
        if len(x_data) > 10:
            x_np = np.array(x_data)
            y_np = np.array(data)
            indices = range(0, len(x_data), 10)
            x_trend = x_np[indices]
            y_trend = y_np[indices]
            if len(x_trend) > 1:
                z = np.polyfit(x_trend, y_trend, 1)
                p = np.poly1d(z)
                speed_fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend (10 frames)',
                    line=dict(color=trend_line_color, dash='dot'),
                ))

        speed_fig.update_layout(
            title="Ball Speed Analysis" if is_final else "Real-time Speed",
            xaxis_title="Time (s)",
            yaxis_title="Speed (m/s)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=color),
            showlegend=True,
            height=400 if is_final else 300,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        if is_final and frame_images:
            # 통계 계산
            avg_speed = np.mean(data)
            max_speed = np.max(data)
            min_speed = np.min(data)
            total_time = max(times)
            
            # 최고/최저 속도 프레임 찾기 (최대 3개씩만)
            tolerance = 0.01  # 1cm/s 허용 오차
            max_speed_indices = []
            min_speed_indices = []
            
            # 시간순으로 정렬된 최대 3개의 최고/최저 속도 지점 찾기
            for i, speed in enumerate(speeds):
                if abs(speed - max_speed) < tolerance:
                    max_speed_indices.append(i)
                    if len(max_speed_indices) >= 3:  # 최대 3개까지만
                        break
                        
            for i, speed in enumerate(speeds):
                if abs(speed - min_speed) < tolerance:
                    min_speed_indices.append(i)
                    if len(min_speed_indices) >= 3:  # 최대 3개까지만
                        break
            
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

            # 그래프 표시
            st.plotly_chart(speed_fig, use_container_width=True)
            
            # 최고 속도 프레임들 표시
            if max_speed_indices:
                st.markdown(f"### 최고 속도 순간: {max_speed:.2f} m/s")
                max_cols = st.columns(len(max_speed_indices))
                
                for idx, col in enumerate(max_cols):
                    max_idx = max_speed_indices[idx]
                    max_frame = frames[max_idx]
                    if max_frame in frame_images:
                        col.markdown(f"시간: {times[max_idx]:.2f} 초")
                        col.image(frame_images[max_frame], channels="BGR", 
                                caption=f"Frame {max_frame}")
            
            # 최저 속도 프레임들 표시
            if min_speed_indices:
                st.markdown(f"### 최저 속도 순간: {min_speed:.2f} m/s")
                min_cols = st.columns(len(min_speed_indices))
                
                for idx, col in enumerate(min_cols):
                    min_idx = min_speed_indices[idx]
                    min_frame = frames[min_idx]
                    if min_frame in frame_images:
                        col.markdown(f"시간: {times[min_idx]:.2f} 초")
                        col.image(frame_images[min_frame], channels="BGR", 
                                caption=f"Frame {min_frame}")

            # CSV 다운로드
            df = pd.DataFrame({
                'Time (s)': times,
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
        else:
            speed_chart.plotly_chart(speed_fig, use_container_width=True, 
                                   key=f"speed_chart_{frame_count}")
                                   
    except Exception as e:
        st.error(f"차트 업데이트 중 오류 발생: {str(e)}")
        # 기본 차트라도 표시
        if not is_final:
            speed_chart.line_chart(data)


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
                 classes, lower_color, upper_color, graph_color, trend_color):
    """비디오 처리 및 분석"""
    # 전역 변수로 선언
    global real_time_speed
    
    # 로컬 변수로 frame_images 정의
    frame_images = {}
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Streamlit 레이아웃 설정
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_frame = st.empty()
    
    with col2:
        real_time_speed = st.empty()
        speed_chart = st.empty()

    analysis_container = st.container()

    # 프레임 카운터와 데이터 저장소
    frames = []
    speeds = []
    ball_positions = {}
    frame_interval = 10  # 매 10프레임마다 저장
    
    # 트래커 초기화
    tracker = create_stable_tracker()
    if tracker is None:
        st.error("트래커를 생성할 수 없습니다. 프로그램을 종료합니다.")
        return

    # 첫 프레임 읽기
    ret, first_frame = video.read()
    if not ret:
        st.error("비디오 첫 프레임을 읽을 수 없습니다.")
        return

    # bbox 초기화
    bbox = initial_bbox
    first_frame = resize_frame(first_frame)

    # YOLO로 공 검출 시도
    try:
        yolo_bbox = detect_ball_with_yolo(first_frame, net, output_layers, classes)
        if yolo_bbox is not None:
            bbox = yolo_bbox
    except Exception as e:
        st.warning(f"YOLO 검출 중 오류가 발생했지만, 색상 기반 추적을 계속합니다.")

    # 트래커 초기화
    try:
        init_success = tracker.init(first_frame, bbox)
        if not init_success:
            st.warning("트래커 초기화 실패, 단순 추적으로 계속합니다.")
    except Exception as e:
        st.error(f"트래커 초기화 중 오류 발생: {str(e)}")

    # 변수 초기화
    prev_pos = None
    speed_queue = deque(maxlen=5)
    frame_count = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 진행 상태 표시
    progress_bar = st.progress(0)
    status_text = st.empty()

    # FPS 제어
    frame_interval = 1.0 / fps
    last_frame_time = time.time()

    st.write("영상 처리 중... (실시간 재생)")
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        try:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()
            
            frame = resize_frame(frame)
            processed_frame, center, bbox = track_ball(frame, tracker, bbox, lower_color, upper_color, 10, 50)
            
            if center and prev_pos:
                speed = calculate_speed(prev_pos, center, fps, pixels_per_meter)
                speed_queue.append(speed)
                avg_speed = sum(speed_queue) / len(speed_queue)
                
                speeds.append(avg_speed)  # m/s로 저장
                frames.append(frame_count)
                ball_positions[frame_count] = center
                
                if frame_count % 10 == 0:  # 매 10프레임마다 이미지 저장
                    frame_images[frame_count] = processed_frame.copy()
                
                real_time_speed.markdown(f"### Current Speed\n{avg_speed:.2f} m/s")
                
                if frame_count % 5 == 0 and speeds:  # 실시간 차트 업데이트
                    update_charts(frames[-100:], speeds[-100:], speed_chart, frame_count, 
                                graph_color, trend_color, is_final=False, fps=fps)
            
            if center:
                prev_pos = center
            
            video_frame.image(processed_frame, channels="BGR", use_column_width=False)

        except Exception as e:
            st.error(f"프레임 {frame_count} 처리 중 오류 발생: {str(e)}")
            if 'bbox' not in locals():
                bbox = initial_bbox

        frame_count += 1
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"처리 중: {frame_count}/{total_frames} 프레임 ({progress}%)")

    video.release()
    status_text.text("영상 처리가 완료되었습니다!")

    # 분석 결과 표시
    if speeds:
        update_charts(frames, speeds, speed_chart, frame_count, 
                     graph_color, trend_color, is_final=True,
                     frame_images=frame_images, 
                     ball_positions=ball_positions,
                     fps=fps)
    else:
        st.warning("속도 데이터가 기록되지 않았습니다.")

        
def process_uploaded_video(uploaded_file, net, output_layers, classes):
    """업로드된 비디오 처리"""
    try:
        if 'video_settings' not in st.session_state:
            st.session_state.video_settings = {}

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        video = cv2.VideoCapture(tfile.name)
        ret, first_frame = video.read()
        video.release()
        
        if not ret or first_frame is None:
            st.error("비디오 프레임을 읽을 수 없습니다.")
            return
            
        # 첫 프레임 크기 조정
        first_frame = resize_frame(first_frame)
        if first_frame is None:
            st.error("프레임 처리 실패")
            return
            
        st.video(tfile.name)
        
        height, width = first_frame.shape[:2]
        
        # 그래프 설정을 위한 컬럼 생성
        col1, col2 = st.columns(2)
        
        with col1:
            # 메인 그래프 색상 선택
            graph_color = st.radio(
                "메인 그래프 색상:",
                ('white', 'black'),
                key='graph_color'
            )
            
        with col2:
            # 추세선 색상 선택
            trend_color = st.radio(
                "추세선 색상:",
                ('white', 'black'),
                key='trend_color'
            )
        
        # 색상 선택
        selected_color, lower_color, upper_color, click_pos = select_color_from_image(first_frame)
        if not any([selected_color is None, lower_color is None, upper_color is None, click_pos is None]):
            # video settings 업데이트 - 모든 설정을 한번에 업데이트
            st.session_state.video_settings.update({
                'selected_color': selected_color,
                'lower_color': lower_color,
                'upper_color': upper_color,
                'click_pos': click_pos,
                'graph_color': graph_color,
                'trend_color': trend_color
            })
        
            if all(k in st.session_state.video_settings for k in 
                  ['selected_color', 'lower_color', 'upper_color', 'click_pos']):
                # 거리 측정을 위한 점 선택
                settings_col1, settings_col2 = st.columns(2)
                
                with settings_col1:
                    x1 = st.slider('첫 번째 점 X 좌표', 0, width, 
                        st.session_state.video_settings.get('x1', width // 4),
                        key='x1_slider')
                    y1 = st.slider('첫 번째 점 Y 좌표', 0, height, 
                        st.session_state.video_settings.get('y1', height // 2),
                        key='y1_slider')
                
                with settings_col2:
                    x2 = st.slider('두 번째 점 X 좌표', 0, width, 
                        st.session_state.video_settings.get('x2', 3 * width // 4),
                        key='x2_slider')
                    y2 = st.slider('두 번째 점 Y 좌표', 0, height, 
                        st.session_state.video_settings.get('y2', height // 2),
                        key='y2_slider')

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
                real_distance = st.number_input("선택한 두 점 사이의 실제 거리(미터)를 입력해주세요:", 
                    min_value=0.1, 
                    value=st.session_state.video_settings.get('real_distance', 1.0), 
                    step=0.1)

                # 설정값 저장
                st.session_state.video_settings.update({
                    'real_distance': real_distance
                })

                # pixels_per_meter 계산
                pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                pixels_per_meter = pixel_distance / real_distance
                st.write(f"계산된 pixels_per_meter: {pixels_per_meter:.2f}")

                # 분석 시작 버튼
                if st.button('영상 내 공 추적 및 분석 시작하기'):
                    initial_bbox = (bbox_x, bbox_y, bbox_w, bbox_h)
                    st.write("Processing video...")
                    try:
                        process_video(tfile.name, initial_bbox, pixels_per_meter, 
                                   net, output_layers, classes, 
                                   st.session_state.video_settings['lower_color'],
                                   st.session_state.video_settings['upper_color'],
                                   st.session_state.video_settings['graph_color'],
                                   st.session_state.video_settings['trend_color'])
                    except Exception as e:
                        st.error(f"비디오 처리 중 오류 발생: {str(e)}")
                        st.error(traceback.format_exc())
            else:
                st.warning("색상을 선택해주세요.")

        try:
            os.unlink(tfile.name)
        except PermissionError:
            pass
            
    except Exception as e:
        st.error(f"비디오 처리 중 오류 발생: {str(e)}")
        st.error(traceback.format_exc())
        
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
