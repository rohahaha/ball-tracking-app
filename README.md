# Ball Tracking and Energy Analysis Application

이 애플리케이션은 비디오에서 공을 추적하고 에너지를 분석하는 Streamlit 웹 애플리케이션입니다.

## 기능

- 비디오에서 공 자동 탐지 및 추적
- 실시간 속도 계산
- 운동 에너지 및 위치 에너지 분석
- 실시간 데이터 시각화

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/ball-tracking-app.git
cd ball-tracking-app
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run ballfo.py
```

## 사용 방법

1. 웹 브라우저에서 애플리케이션 열기
2. 분석할 비디오 파일 업로드
3. 거리 측정을 위한 기준점 설정
4. 공의 색상 선택 및 추적 매개변수 조정
5. 분석 시작 버튼 클릭

## 주의사항

- 처음 실행 시 YOLO 모델 파일이 자동으로 다운로드됩니다
- 비디오 파일은 mp4, avi, mov 형식을 지원합니다
- 충분한 메모리와 CPU 성능이 필요합니다

## 라이선스

MIT License
