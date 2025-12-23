# DLSW 프로젝트 구조 초안

## 최상위 (`DLSW/`)
- `main.py`: 진입점. `dl_software.app.run_app()` 호출로 GUI 실행.
- `datasets/`: 학습·추론용 데이터셋 저장 위치(원본/전처리 결과).
- `projects/`: 라벨링 프로젝트 루트. 새 프로젝트 생성 시 하위 폴더로 관리.
- `runs/`: 학습/추론 결과(체크포인트, 로그, 예측 이미지·JSON 등) 저장 위치.
- 기타 리소스: 테스트용 모델 가중치(`yolo11n.pt` 등).

### 간단 트리
```text
DLSW/
├─ main.py
├─ datasets/
├─ projects/
├─ runs/
├─ dl_software/
│  ├─ app.py
│  ├─ label_tool.py
│  ├─ assets/
│  ├─ models/
│  └─ ui/
```

## 앱 부트스트랩 (`dl_software/app.py`)
- `run_app()`: `QApplication` 생성, `DLMainWindow` 인스턴스화 후 실행.
- CLI 실행(`python main.py`) 시 여기로 진입.

## 핵심 로직 (`dl_software/label_tool.py`)
- 프로젝트/라벨 관리, 최근 프로젝트 목록, 데이터 로드·저장 담당.
- UI 탭들이 공유하는 컨트롤러로 버튼·신호를 연결하고 상태를 업데이트.

## UI 프레임워크 (`dl_software/ui/`)
- `main_window.py`: 메뉴(Help/Project/Settings), 시작 페이지, 탭 스택 관리. About 메뉴는 GitHub Wiki 링크를 엽니다.
- `styles.py`: 전역 QSS 스타일.
- `widgets.py`: 커스텀 위젯 모음.
- `augment_dialog.py`: 증강 옵션 설정/미리보기 다이얼로그.

### 탭별 화면 (`dl_software/ui/tabs/`)
- `label.py`: 라벨링 뷰와 도구(박스/폴리곤/Smart Polygon 등). 학습 탭 이동 버튼 포함.
- `train.py`: 학습 파라미터 설정, 실행/중단, 로그·체크포인트 관리.
- `infer.py`: 단일 이미지/폴더/영상/웹캠 추론 모드, 오버레이 저장, JSON 내보내기 등.
- 모든 탭이 하나의 `label_tool` 컨트롤러를 공유해 상태바 및 탭 전환을 연동.

## 에셋·모델
- `dl_software/assets/`: 증강 예제 이미지(회전, 블러, 색상 변화 등)로 UI에서 시각 안내에 사용.
- `dl_software/models/`: 모델 가중치 보관(SAM PTH/ONNX 샘플 포함). 추가 체크포인트도 여기에 저장.

## 데이터/출력 경로 요약
- 입력/데이터셋: `datasets/`
- 사용자 프로젝트: `projects/<프로젝트명>/`
- 학습·추론 결과: `runs/`
- 모델 체크포인트: `dl_software/models/`

이 구조를 위키의 "프로젝트 구조" 섹션으로 옮기고, 필요 시 각 폴더별 예시 스크린샷이나 사용 예를 추가하면 됩니다.
