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

# Streamlit 페이지 설정
st.set_page_config(page_title="공 추적 및 에너지 분석기", layout="wide")
st.title('공 추적 및 에너지 분석기')

# YOLO 파일 경로 및 URL 설정
YOLO_DIR = "yolo"
YOLO_FILES = {
    "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
}

# 구글 드라이브 weights 파일 설정
WEIGHTS_FILE_ID = "1XWTMChKOcrVpo-uaIldGp6bRzBfYIGqJ"
WEIGHTS_FILENAME = "yolov4.weights"

def download_yolo_files():
    """YOLO 모델 파일 다운로드"""
    try:
        os.makedirs(YOLO_DIR, exist_ok=True)
        
        # cfg와 names 파일 다운로드
        for filename, url in YOLO_FILES.items():
            filepath = os.path.join(YOLO_DIR, filename)
            if not os.path.exists(filepath):
                with st.spinner(f"Downloading {filename}..."):
                    try:
                        urllib.request.urlretrieve(url, filepath)
                        st.success(f"Downloaded {filename}")
                    except Exception as e:
                        st.error(f"Error downloading {filename}: {str(e)}")
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
                        file_size = os.path.getsize(weights_path) / (1024 * 1024)  # Convert to MB
                        if file_size > 1:  # Check if file size is reasonable
                            st.success(f"Successfully downloaded YOLOv4 weights! (File size: {file_size:.1f} MB)")
                        else:
                            st.error("Downloaded weights file appears to be incomplete or corrupted.")
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
        else:
            st.info("YOLOv4 weights file already exists.")
        
        return True
    
    except Exception as e:
        st.error(f"Error in download process: {str(e)}")
        return False

def verify_yolo_files():
    """YOLO 파일들의 존재 여부와 무결성 확인"""
    required_files = {
        "yolov4.cfg": 100 * 1024,  # 최소 100KB
        "coco.names": 1024,        # 최소 1KB
        "yolov4.weights": 200 * 1024 * 1024  # 최소 200MB
    }
    
    for filename, min_size in required_files.items():
        filepath = os.path.join(YOLO_DIR, filename)
        if not os.path.exists(filepath):
            st.warning(f"Missing required file: {filename}")
            return False
        if os.path.getsize(filepath) < min_size:
            st.warning(f"File {filename} appears to be incomplete or corrupted")
            return False
    
    return True

# YOLO 초기화 함수 업데이트
def initialize_yolo():
    """YOLO 모델 초기화"""
    if not verify_yolo_files():
        st.warning("YOLO 모델 파일이 없거나 불완전합니다. 다운로드가 필요합니다.")
        if st.button("YOLO 파일 다운로드"):
            if not download_yolo_files():
                st.error("파일 다운로드에 실패했습니다. 다시 시도해주세요.")
                return None, None, None
            if not verify_yolo_files():
                st.error("다운로드된 파일이 올바르지 않습니다.")
                return None, None, None
    
    try:
        weights_path = os.path.join(YOLO_DIR, WEIGHTS_FILENAME)
        cfg_path = os.path.join(YOLO_DIR, "yolov4.cfg")
        names_path = os.path.join(YOLO_DIR, "coco.names")
        
        with st.spinner("YOLO 모델을 로드하는 중..."):
            net = cv2.dnn.readNet(weights_path, cfg_path)
            
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
        st.error(f"YOLO 모델 초기화 오류: {str(e)}")
        return None, None, None

def create_stable_tracker():
    """안정적인 CSRT 트래커 생성"""
    try:
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            tracker = cv2.legacy.TrackerCSRT_create()
        elif hasattr(cv2, 'TrackerCSRT_create'):
            tracker = cv2.TrackerCSRT_create()
        else:
            raise AttributeError("CSRT 트래커를 생성할 수 없습니다.")
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
        
        # 거리 측정을 위한 점 선택
        st.write("알고 있는 실제 거리에 해당하는 두 점을 선택해주세요.")
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.slider('첫 번째 점 X 좌표', 0, width, width // 4)
            y1 = st.slider('첫 번째 점 Y 좌표', 0, height, height // 2)
        with col2:
            x2 = st.slider('두 번째 점 X 좌표', 0, width, 3 * width // 4)
            y2 = st.slider('두 번째 점 Y 좌표', 0, height, height // 2)
        
        # 높이 기준점 선택
        st.write("높이의 기준점(h=0)을 선택해주세요.")
        height_reference_y = st.slider('높이 기준점 Y 좌표', 0, height, height)
        height_reference = (0, height_reference_y)
        
        # 초기 바운딩 박스 설정
        st.write("추적할 공의 초기 위치와 크기를 선택해주세요.")
        col1, col2 = st.columns(2)
        with col1:
            bbox_x = st.slider('공의 X 좌표', 0, width, width // 2)
            bbox_w = st.slider('너비', 10, width - bbox_x, 50)
        with col2:
            bbox_y = st.slider('공의 Y 좌표', 0, height, height // 2)
            bbox_h = st.slider('높이', 10, height - bbox_y, 50)
        
        initial_bbox = (bbox_x, bbox_y, bbox_w, bbox_h)
        
        # 프레임에 정보 표시
        frame_with_info = first_frame.copy()
        cv2.circle(frame_with_info, (x1, y1), 5, (0, 255, 0), -1)
        cv2.circle(frame_with_info, (x2, y2), 5, (0, 255, 0), -1)
        cv2.line(frame_with_info, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(frame_with_info, (0, height_reference_y), (width, height_reference_y), (255, 0, 0), 2)
        cv2.rectangle(frame_with_info, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 255), 2)
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
                process_video(tfile.name, initial_bbox, pixels_per_meter, mass, height_reference, net, output_layers, classes)
            except Exception as e:
                st.error(f"비디오 처리 중 오류 발생: {str(e)}")
                st.error(traceback.format_exc())
        
        try:
            os.unlink(tfile.name)
        except PermissionError:
            pass
    else:
        st.error("Failed to read the first frame of the video.")

def process_video(video_path, initial_bbox, pixels_per_meter, mass, height_reference, net, output_layers, classes):
    """비디오 처리 및 분석"""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 색상 선택
    color = st.color_picker("공 색상 선택", "#00ff00")
    selected_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    
    # HSV 색상 범위 설정
    b, g, r = selected_color
    h, s, v = rgb_to_hsv(r, g, b)
    color_tolerance = st.slider("색상 허용 범위", 0, 50, 20)
    lower_color = np.array([max(0, h - color_tolerance), max(0, s - 50), max(0, v - 50)])
    upper_color = np.array([min(179, h + color_tolerance), min(255, s + 50), min(255, v + 50)])

    # 트래커 초기화
    tracker = create_stable_tracker()
    if tracker is None:
        return

    # YOLO로 첫 프레임에서 공 탐지
    ret, first_frame = video.read()
    if not ret:
        st.error("비디오 첫 프레임을 읽을 수 없습니다.")
        return

    bbox = detect_ball_with_yolo(first_frame, net, output_layers, classes)
    if bbox is None:
        bbox = initial_bbox
    
    tracker.init(first_frame, tuple(bbox))

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
