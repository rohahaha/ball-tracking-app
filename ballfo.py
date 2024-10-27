import cv2
import numpy as np
import streamlit as st
from collections import deque
import plotly.graph_objs as go
from scipy.signal import savgol_filter
import colorsys
import tempfile
import os
import traceback


# YOLOv4 모델 파일 경로 설정
cfg_path = os.path.join("yolo", "yolov4.cfg")
weights_path = os.path.join("yolo", "yolov4.weights")
names_path = os.path.join("yolo", "coco.names")

# YOLOv4 모델 로드
net = cv2.dnn.readNet(weights_path, cfg_path)

# COCO 클래스 파일 로드
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 레이어 이름 가져오기
layer_names = net.getLayerNames()

# 출력 레이어 이름 가져오기
unconnected_layers = net.getUnconnectedOutLayers()

# unconnected_layers가 배열, 리스트, 스칼라 값일 수 있으므로 모든 경우를 처리
if isinstance(unconnected_layers, np.ndarray):
    output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]
elif isinstance(unconnected_layers, list):
    output_layers = [layer_names[int(i[0]) - 1] if isinstance(i, (list, np.ndarray)) else layer_names[int(i) - 1] for i in unconnected_layers]
else:
    output_layers = [layer_names[int(unconnected_layers) - 1]]

selected_color = None  # 전역 변수로 선택한 색상을 저장

def create_stable_tracker():
    try:
        # OpenCV 4.9.0 이후 버전에서는 legacy 모듈을 통해 접근
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            tracker = cv2.legacy.TrackerCSRT_create()
        elif hasattr(cv2, 'TrackerCSRT_create'):
            tracker = cv2.TrackerCSRT_create()
        else:
            raise AttributeError("CSRT 트래커를 생성할 수 없습니다. OpenCV 버전을 확인하세요.")
        
        st.success("CSRT 트래커를 성공적으로 생성했습니다.")
        return tracker

    except Exception as e:
        st.error(f"CSRT 트래커 생성 실패: {str(e)}")
        return None

# 마우스 클릭 이벤트 처리 함수
# def on_mouse_click(event, x, y, flags, param):
    global selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        # 클릭한 위치의 색상(BGR)
        selected_color = frame[y, x]
        st.write(f"선택한 색상: BGR {selected_color}")

def print_opencv_info():
    st.write(f"OpenCV 버전: {cv2.__version__}")
    st.write("CSRT 트래커 사용 가능: 생성 성공")

def create_stable_tracker():
    try:
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            # OpenCV 4.9.0에서 CSRT 트래커를 legacy 모듈에서 검증함
            tracker = cv2.legacy.TrackerCSRT_create()
        else:
            # 마다 legacy 모듈이 없다면 일반적인 TrackerCSRT_create 사용
            tracker = cv2.TrackerCSRT_create()
        
        st.success("CSRT 트래커를 성공적으로 생성했습니다.")
        return tracker
    except Exception as e:
        st.error(f"CSRT 트래커 생성 실패: {str(e)}")
        return None

def detect_ball_with_yolo(frame):
    height, width, channels = frame.shape

    # YOLO 입력 준비
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 탐지된 객체 중에서 공만 선택
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if classes[class_id] == "sports ball" and confidence > 0.5:
                # 탐지된 객체의 바운딩 박스 정보
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 바운딩 박스 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max Suppression으로 가장 신뢰도 높은 객체만 선택
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 결과가 비어있는지 확인
    if len(indexes) > 0:
        # 가장 신뢰도 높은 공 탐지 정보 반환
        if isinstance(indexes[0], (list, np.ndarray)):
            i = indexes[0][0]
        else:
            i = indexes[0]
        return boxes[i]
    else:
        # 탐지된 객체가 없을 경우 None 반환
        st.warning("YOLO 모델이 공을 탐지하지 못했습니다. 다른 영상이나 공의 색상을 확인하세요.")
        return None
    height, width, channels = frame.shape

    # YOLO 입력 준비
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 탐지된 객체 중에서 공만 선택
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if classes[class_id] == "sports ball" and confidence > 0.5:
                # 탐지된 객체의 바운딩 박스 정보
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 바운딩 박스 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max Suppression으로 가장 신뢰도 높은 객체만 선택
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 결과가 비어있는지 확인
    if len(indexes) > 0:
        # 가장 신뢰도 높은 공 탐지 정보 반환
        if isinstance(indexes[0], (list, np.ndarray)):
            i = indexes[0][0]
        else:
            i = indexes[0]
        return boxes[i]
    else:
        # 탐지된 객체가 없을 경우 None 반환
        return None
    height, width, channels = frame.shape

    # YOLO 입력 준비
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 탐지된 객체 중에서 공만 선택
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if classes[class_id] == "sports ball" and confidence > 0.5:
                # 탐지된 객체의 바운딩 박스 정보
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 바운딩 박스 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max Suppression으로 가장 신뢰도 높은 객체만 선택
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 결과가 비어있는지 확인
    if len(indexes) > 0 and isinstance(indexes, (list, np.ndarray)):
        # 가장 신뢰도 높은 공 탐지 정보 반환
        if isinstance(indexes[0], (list, np.ndarray)):
            i = indexes[0][0]
        else:
            i = indexes[0]
        return boxes[i]
    else:
        # 탐지된 객체가 없을 경우 None 반환
        return None

def detect_ball(frame, lower_color, upper_color, min_radius, max_radius):
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
    color_detection, color_center = detect_ball(frame, lower_color, upper_color, min_radius, max_radius)
    
    success, box = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(v) for v in box]
        tracker_center = (x + w//2, y + h//2)
        
        if color_detection:
            # 색상 검출과 트래커 결과를 비교
            color_x, color_y, color_radius = color_detection
            distance = np.sqrt((color_x - tracker_center[0])**2 + (color_y - tracker_center[1])**2)
            
            if distance > color_radius:  # 트래커와 색상 검출 결과가 크게 다른 경우
                # 색상 검출 결과로 트래커 재초기화
                new_bbox = (color_x - color_radius, color_y - color_radius, 
                            2*color_radius, 2*color_radius)
                tracker.init(frame, new_bbox)
                bbox = new_bbox
                center = color_center
            else:
                # 트래커와 색상 검출 결과의 평균 사용
                center = ((tracker_center[0] + color_center[0])//2, 
                          (tracker_center[1] + color_center[1])//2)
                bbox = (x, y, w, h)
        else:
            center = tracker_center
            bbox = (x, y, w, h)
    elif color_detection:
        # 트래커 실패 시 색상 검출 결과로 재초기화
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
    distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
    distance_meters = distance / pixels_per_meter
    speed = distance_meters * fps  # m/s
    return speed

def calculate_energy(speed, height, mass):
    kinetic_energy = 0.5 * mass * (speed ** 2)
    potential_energy = mass * 9.8 * height
    mechanical_energy = kinetic_energy + potential_energy
    return kinetic_energy, potential_energy, mechanical_energy

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(h * 179), int(s * 255), int(v * 255)

def process_video(video_path, initial_bbox, lower_color, upper_color, min_radius, max_radius, pixels_per_meter, mass, height_reference):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_px = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = video.read()
    if not ret:
        st.error("Failed to read the first frame of the video")
        return

    # 색상 선택을 위한 Streamlit 컨트롤
    st.write("공의 색상을 선택하세요:")
    color = st.color_picker("공 색상 선택", "#00ff00")  # 기본값: 녹색
    
    # HTML 색상 코드를 BGR로 변환
    selected_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # RGB to BGR
    
    if selected_color is not None:
        b, g, r = selected_color
        h, s, v = rgb_to_hsv(r, g, b)
        color_tolerance = st.slider("색상 허용 범위", 0, 50, 20)
        lower_color = np.array([max(0, h - color_tolerance), max(0, s - 50), max(0, v - 50)])
        upper_color = np.array([min(179, h + color_tolerance), min(255, s + 50), min(255, v + 50)])
    else:
        st.error("색상을 선택하지 않았습니다. 기본 설정을 사용합니다.")
        lower_color = np.array([35, 50, 50])  # 기본 초록색 범위
        upper_color = np.array([85, 255, 255])

    # YOLOv4로 첫 번째 프레임에서 공 탐지
    bbox = detect_ball_with_yolo(first_frame)
    if bbox is None:
        st.error("첫 번째 프레임에서 공을 탐지할 수 없습니다. 다른 영상이나 공의 색상을 확인해 주세요.")
        return

    # CSRT 트래커 초기화
    tracker = create_stable_tracker()
    if tracker is None:
        st.error("CSRT 트래커를 생성할 수 없습니다. 프로그램을 종료합니다.")
        return

    tracker.init(first_frame, tuple(bbox))

    # 색상 선택 모드 추가
    # global selected_color
    # 색상 선택을 위한 Streamlit 위젯 사용
    #st.image(first_frame, channels="BGR", use_column_width=True)
    #clicked = st.button("이미지에서 색상 선택")
    #if clicked:
        # 기본 색상값 사용 또는 다른 방식으로 색상 선택
        #selected_color = (0, 0, 255)  # 예시: 빨간색

    #st.write("비디오 화면에서 공의 색상을 선택하려면 클릭하세요.")
    #while selected_color is None:
        #cv2.imshow(window_name, first_frame)
        #if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 루프 종료
            #break

    #cv2.destroyWindow(window_name)

    #if selected_color is not None:
        #b, g, r = selected_color
        #h, s, v = rgb_to_hsv(r, g, b)
        #color_tolerance = st.slider("색상 허용 범위", 0, 50, 20)
        #lower_color = np.array([max(0, h - color_tolerance), max(0, s - 50), max(0, v - 50)])
        #upper_color = np.array([min(179, h + color_tolerance), min(255, s + 50), min(255, v + 50)])
    #else:
        #st.error("색상을 선택하지 않았습니다. 기본 설정을 사용합니다.")

    # YOLOv4로 첫 번째 프레임에서 공 탐지
    bbox = detect_ball_with_yolo(first_frame)
    if bbox is None:
        st.error("첫 번째 프레임에서 공을 탐지할 수 없습니다. 다른 영상이나 공의 색상을 확인해 주세요.")
        return

    # CSRT 트래커 초기화
    tracker = create_stable_tracker()
    if tracker is None:
        st.error("CSRT 트래커를 생성할 수 없습니다. 프로그램을 종료합니다.")
        return

    tracker.init(first_frame, tuple(bbox))
    
    prev_pos = None
    speed_queue = deque(maxlen=5)
    
    speeds = []
    kinetic_energies = []
    potential_energies = []
    mechanical_energies = []
    frames = []
    
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
            frame, center, bbox = track_ball(frame, tracker, bbox, lower_color, upper_color, min_radius, max_radius)
            
            if center:
                if prev_pos:
                    speed = calculate_speed(prev_pos, center, fps, pixels_per_meter)
                    speed_queue.append(speed)
                    avg_speed = sum(speed_queue) / len(speed_queue)
                    
                    h = (height_reference[1] - center[1]) / pixels_per_meter
                    
                    ke, pe, me = calculate_energy(avg_speed, h, mass)
                    
                    cv2.putText(frame, f"Speed: {avg_speed*3.6:.2f} km/h", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    speeds.append(avg_speed*3.6)
                    kinetic_energies.append(ke)
                    potential_energies.append(pe)
                    mechanical_energies.append(me)
                    frames.append(frame_count)
                
                prev_pos = center
            
            cv2.line(frame, (0, height_reference[1]), (width, height_reference[1]), (255, 0, 0), 2)
            
            video_frame.image(frame, channels="BGR", use_column_width=True)
            
            if frame_count % 30 == 0:  # 30프레임마다 차트 업데이트
                speed_fig = go.Figure(go.Scatter(x=frames[-100:], y=speeds[-100:], mode='lines', name='Speed (km/h)'))
                speed_fig.update_layout(title="Speed over time (last 100 frames)", xaxis_title="Frame", yaxis_title="Speed (km/h)")
                speed_chart.plotly_chart(speed_fig, use_container_width=True, key=f"speed_chart_{frame_count}")
                
                energy_fig = go.Figure()
                energy_fig.add_trace(go.Scatter(x=frames[-100:], y=kinetic_energies[-100:], mode='lines', name='Kinetic Energy (J)'))
                energy_fig.add_trace(go.Scatter(x=frames[-100:], y=potential_energies[-100:], mode='lines', name='Potential Energy (J)'))
                energy_fig.add_trace(go.Scatter(x=frames[-100:], y=mechanical_energies[-100:], mode='lines', name='Mechanical Energy (J)'))
                energy_fig.update_layout(title="Energy over time (last 100 frames)", xaxis_title="Frame", yaxis_title="Energy (J)")
                energy_chart.plotly_chart(energy_fig, use_container_width=True, key=f"energy_chart_{frame_count}")
        
        except Exception as e:
            st.error(f"프레임 {frame_count} 처리 중 오류 발생: {str(e)}")
            st.error(traceback.format_exc())
        
        frame_count += 1
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    video.release()
    status_text.text("Video processing completed!")

# Streamlit UI
st.title('개선된 공 추적 및 에너지 분석기')

print_opencv_info()

uploaded_file = st.file_uploader("비디오 파일을 선택하세요.", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        video = cv2.VideoCapture(tfile.name)
        ret, first_frame = video.read()
        video.release()
        
        if ret:
            st.video(tfile.name)
            
            height, width = first_frame.shape[:2]
            
            st.write("알고 있는 실제 거리에 해당하는 두 점을 선택해주세요.")
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.slider('첫 번째 점 X 좌표', 0, width, width // 4)
                y1 = st.slider('첫 번째 점 Y 좌표', 0, height, height // 2)
            with col2:
                x2 = st.slider('두 번째 점 X 좌표', 0, width, 3 * width // 4)
                y2 = st.slider('두 번째 점 Y 좌표', 0, height, height // 2)
            
            st.write("높이의 기준점(h=0)을 선택해주세요.")
            height_reference_y = st.slider('높이 기준점 Y 좌표', 0, height, height)
            height_reference = (0, height_reference_y)
            
            st.write("추적할 공의 초기 위치와 크기를 선택해주세요.")
            col1, col2 = st.columns(2)
            with col1:
                bbox_x = st.slider('공의 X 좌표', 0, width, width // 2)
                bbox_w = st.slider('너비', 10, width - bbox_x, 50)
            with col2:
                bbox_y = st.slider('공의 Y 좌표', 0, height, height // 2)
                bbox_h = st.slider('높이', 10, height - bbox_y, 50)
            
            initial_bbox = (bbox_x, bbox_y, bbox_w, bbox_h)
            
            frame_with_info = first_frame.copy()
            cv2.circle(frame_with_info, (x1, y1), 5, (0, 255, 0), -1)
            cv2.circle(frame_with_info, (x2, y2), 5, (0, 255, 0), -1)
            cv2.line(frame_with_info, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame_with_info, (0, height_reference_y), (width, height_reference_y), (255, 0, 0), 2)
            cv2.rectangle(frame_with_info, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 255), 2)
            st.image(frame_with_info, channels="BGR", use_column_width=True)
            
            real_distance = st.number_input("선택한 두 점 사이의 실제 거리(미터)를 입력해주세요:", min_value=0.1, value=1.0, step=0.1)
            
            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            pixels_per_meter = pixel_distance / real_distance
            st.write(f"계산된 pixels_per_meter: {pixels_per_meter:.2f}")
            
            mass = st.number_input("공의 질량(kg)을 입력해주세요:", min_value=0.1, value=0.1, step=0.1)
            
            if st.button('영상 내 공 추적 및 에너지 분석 시작하기'):
                st.write("Processing video...")
                process_video(tfile.name, initial_bbox, None, None, 10, 50, pixels_per_meter, mass, height_reference)
            
            try:
                os.unlink(tfile.name)
            except PermissionError:
                pass
        else:
            st.error("Failed to read the first frame of the video.")
    
    except Exception as e:
        st.error(f"비디오 파일 처리 중 오류 발생: {str(e)}")
        st.error(traceback.format_exc())