모델 추론 탭입니다. 학습된 가중치(weight)를 선택한 뒤, 입력 이미지(또는 폴더)를 지정하여 **추론을 실행**하고, 결과를 **확인/필터링**한 후 **파일로 내보내기** `Export Results`까지 수행합니다.

- [Model Inference](#1-inference)
- [Results & Display](#2-results--display)
- [Export Results](#3-export-results)

**Infer Tab Overview**

|<img width="1000" alt="image" src="https://github.com/user-attachments/assets/4bff58a6-d36c-4b10-9192-16e7bcff9b20" />|
|:--|
|**왼쪽 패널**<br>1. `Model`: 추론에 사용할 가중치(weight) 파일 선택 및 task 확인(Object Detection / Instance Segmentation).<br>2. `Source`: 입력 데이터 모드 선택(예: Folder) 및 경로(path) 지정, 추론 대상 파일 목록 확인.<br>3. `Run Inference`: 선택한 모델과 입력 데이터로 추론 실행.<br><br>**가운데 패널**<br>4. `Preview`: 추론 결과 시각화(박스/라벨, 필요 시 마스크). 상단 네비게이션으로 파일 이동(예: 1/7).<br><br>**오른쪽 패널**<br>5. `Detections / Instances`: 예측 결과 테이블(ID, class, conf, 좌표 등) 확인.<br>6. `Display`: score(신뢰도 임계값) 등 표시 옵션 조정. 값 조정으로 결과 표시 기준을 빠르게 튜닝.<br>7. `Export Results`: 추론 결과 내보내기(예: `Export_json`, `Export_txt`, 필요 시 `export mask`).|

---

### 1. Inference

추론에 사용할 모델 가중치를 로드하고, 태스크를 설정합니다. 추론 할 소스 파일의 종류를 선택하고, 소스 파일을 업로드하여 추론 준비를 마칩니다. 마지막으로 `Run Inference`버튼을 클릭하여 소스 파일에 대한 모델 추론을 시작합니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/53ad8822-9b41-4437-bb95-136fabaec67c" />

1. **Load Model Weights:** Model 패널에서 `Browse` 버튼을 클릭하여 추론에 사용할 모델 가중치 (`weights`) 파일을 선택합니다. <br>일반적으로 Train Tab 에서 학습한 모델을 학습 결과가 저장된 디렉토리 (`.../weights/best.pt`)에서 찾아 업로드합니다. <br>
그 밖의 ONNX 포맷의 모델은 `*.onnx` 파일을, OpenVINO 포맷의 모델은 `openvino` 디렉토리를 찾아 경로로 입력합니다.

2. **Select Task:** 모델 가중치가 학습한 Task에 맞춰 `Object Detection` 혹은 `Instance Segmentation`으로 설정합니다.

3. **Set Source Type:** 추론을 진행할 소스파일의 모드를 설정합니다. `Current Image`, `Single Image`, `Folder` 중 선택합니다. 

    * `Current Image`: 현재 캔버스에 업로드되어있는 이미지에 대해서 추론을 진행합니다.
    * `Single Image`: 추론을 진행할 단일 이미지를 파일 탐색기에서 불러와 업로드하여 추론을 진행합니다.
    * `Folder`: 추론을 진행할 폴더 디렉토리를 파일 탐색기에서 열어 경로 내의 모든 이미지에 대하여 추론을 진행합니다.
4. **Browse Images**: `Browse` 버튼을 눌러 위에서 선택한 모드에 맞게 경로를 파일 탐색기에서 불러옵니다. 아래의 파일 리스트에 경로 내 선택한 모드에 맞는 이미지들을 불러와 리스트업 합니다. 

---

### 2. Results Display

선택한 모델 (weights)과 입력 소스 (images) 를 기반으로 추론 (inference)을 수행하고, 그 결과를 원본 위에 시각화 (overlay) 합니다. 오른쪽 위의 `Detections / Instances` 패널에서는 추론 된 결과들을 테이블로 확인할 수 있으며, 오른쪽 아래의 `Display`패널에서 표시 옵션을 조절하여 결과를 필터링하여 화면에 출력할 수 있습니다.

아래는 각각 **Object Detection(객체 검출)** 과 **Instance Segmentation(객체 분할)** 태스크를 진행하는 경우의 결과 시각화 방법에 대한 설명입니다.

#### Object Detection

<img width="800" alt="image" src="https://github.com/user-attachments/assets/4988534f-9ff9-492e-a33f-d2a8e5c92593" />

1. **Run Inference**: 좌측 하단의 `Run Inference` 버튼을 클릭하면, 현재 선택된 모델과 입력 파일에 대해 추론을 진행합니다. 추론이 완료되면 결과가 오버레이된 이미지, 검출 결과 리스트가 표시되며, 해당 결과를 `내보내기`할 수 있습니다.

2. **Results Visualization**: 중앙 캔버스는 추론 결과를 원본 이미지 위에 직접 오버레이하여 확인할 수 있는 영역입니다.

    * 이미지 위에 검출된 객체의 **Bounding Box** 가 그려지고, 박스 상단에는 **클래스명과 신뢰도(Confidence)** 가 표시됩니다.
    * 상단의 `좌/우` 이동 버튼과 `현재/전체` 표시는 여러 이미지 (또는 여러 프레임) 결과를 **이전/다음**으로 넘겨 확인할 때 사용합니다. 키보드 방향키 (좌, 우)로도 이동이 가능합니다.
    * 표시되는 결과는 우측의 Display패널에서 설정한 옵션 값에 따라 즉시 반영됩니다.

3. **Detections/Instances**: 우측 상단의 Detections / Instances 테이블은 이미지에서 검출된 객체를 구조화하여 보여줍니다.

    * 각 행은 하나의 검출 결과(인스턴스)를 의미합니다.
    * 일반적으로 아래와 같은 정보를 포함합니다.

        * `ID`: 검출 결과의 인덱스(식별 번호)
        * `Class`: 예측된 클래스 라벨
        * `Conf`: 예측된 클래스 라벨에 대한 신뢰도 (Confidence)
        * `좌표 정보`: 박스 좌표 (`x1, y1, x2, y2`)
    * 결과가 많을 경우 스크롤하여 전체 목록을 확인할 수 있습니다.
    * 각 행을 선택하면 선택한 행에 해당하는 객체를 빨간색 바운딩 박스로 표시합니다.

4. **Display**: 우측 중단의 **Display** 영역은 결과 시각화 방식을 조정하는 옵션입니다.

    * **score**: 결과를 표시할 **신뢰도 임계값(Threshold)** 입니다.
      값을 올리면 낮은 신뢰도의 검출 결과가 숨겨지고, 값을 내리면 더 많은 결과가 표시됩니다.
    * **mask α**: 마스크가 표시되는 작업(세그멘테이션/인스턴스 마스크 등)에서 **마스크 투명도(Alpha)** 를 조정합니다.
      값이 클수록 마스크가 더 진하게 보이고, 값이 작을수록 더 투명하게 보입니다.

#### Instance Segmentation


<img width="800" alt="image" src="https://github.com/user-attachments/assets/9be53bce-407d-4e08-85a1-9b8e1c22c114" />

1. **Run Inference**: 좌측 하단의 `Run Inference` 버튼을 클릭하면, 현재 선택된 모델과 입력 파일에 대해 객체 분할에 대한 모델 추론을 진행합니다. 입력한 이미지에 대한 객체 분할 추론이 완료되면 각 객체에 대한 마스크가 오버레이된 이미지, 검출 결과 리스트가 표시되며, 해당 결과를 `Export (내보내기)`할 수 있습니다.

2. **Results Visualization**: 중앙 캔버스는 추론 결과를 원본 이미지 위에 직접 오버레이하여 확인할 수 있는 영역입니다.

    * Instance Segmentation 결과가 있을 경우 class별 이미지 위에 분할된 객체의 **Mask** 가 그려집니다.
    * 상단의 `좌/우` 이동 버튼과 `현재/전체` 표시는 여러 이미지 (또는 여러 프레임) 결과를 **이전/다음**으로 넘겨 확인할 때 사용합니다. 키보드 방향키 (좌, 우)로도 이동이 가능합니다.
    * 표시되는 결과는 우측의 Display 패널에서 설정한 옵션 값에 따라 즉시 반영됩니다.

3. **Detections/Instances**: 우측 상단의 Detections / Instances 테이블은 이미지에서 검출된 객체를 구조화하여 보여줍니다.

    |<img width="500" alt="image" src="https://github.com/user-attachments/assets/0bf61f98-cfe5-4cea-8fc6-694f5d0808bb" />|<img width="500" alt="image" src="https://github.com/user-attachments/assets/78627d5c-ea4f-4ac1-a525-875b89f742b5" />|
    |:--:|:--:|
    |`nylon_caster` 행 선택 결과| `nylon_caster` visualization 선택 해제 결과|

    * 각 행은 검출된 클래스를 의미합니다.
    * 테이블에서 특정 행을 선택하면, 선택된 행의 추론 결과가 중앙 캔버스에 빨간색 테두리로 표시되어 선택 상태를 확인할 수 있습니다.
    * 일반적으로 아래와 같은 정보를 포함합니다.

        * `ID`: 검출 결과의 인덱스(식별 번호)
        * `Class`: 예측된 클래스 라벨
        * `👁️ (체크박스)`: 해당 mask layer를 중앙 캔버스에서 표시할지 여부를 나타냅니다. 
    * 결과가 많을 경우 스크롤하여 전체 목록을 확인할 수 있습니다.

4. **Display**: 우측 중단의 **Display** 영역은 결과 시각화 방식을 조정하는 옵션입니다.

    * **score**: 결과를 표시할 **신뢰도 임계값(Threshold)** 입니다.
      값을 올리면 낮은 신뢰도의 검출 결과가 숨겨지고, 값을 내리면 더 많은 결과가 표시됩니다.
    * **mask α**: 마스크 오버레이의 표시 강도 **마스크 투명도(Alpha)** 를 조정합니다. 값이 클수록 마스크가 더 진하게 보이고, 값이 작을수록 더 투명하게 보입니다.

---

### 3. Export Results


#### Object Detection

Object Detection Task의 경우 `*.json` 형식과, `*.txt` 형식으로 결과를 내보낼 수 있습니다.
* `JSON`파일의 경우 모든 결과가 하나의 파일에 통합되어, 이미지 파일 명을 `key`값으로 하여 `value`에 추론 결과들이 저장됩니다. 
* `txt`파일의 경우 각 이미지에 대한 추론 결과가 각각의 `txt`파일에 생성되어 추론된 이미지 파일 개수만큼 결과 파일이 생성됩니다.

#### Export as `*.json`

|<img width="650" alt="image" src="https://github.com/user-attachments/assets/1bdc4449-c1b5-44b6-bd54-75e8e0b9dd1c" />|<img width="650" alt="image" src="https://github.com/user-attachments/assets/fbe8c304-bb97-4d37-be5a-b1fa1653f7ff" />|
|:--|:--|
|`Exp. as json` 버튼을 눌러 저장할 폴더 명 입력 후 `OK`버튼 클릭|저장할 `json`파일 명 입력 후 `Save`|
|<img width="650" alt="image" src="https://github.com/user-attachments/assets/65c60ea1-2a50-4de9-b405-dc649f19018d" />|<img width="650" alt="image" src="https://github.com/user-attachments/assets/0341a9a9-3535-4016-bfe0-dd940ce09d17" />|
|저장이 완료되면 결과 파일이 저장된 위치가 출력됩니다.|`*.json` 파일의 구조입니다. `class_id`, `class_name`, <br>`confidence`, `bbox`값이 저장되며, bbox 형식은 <br>`x1, y1, x2, y2`입니다.|

#### Export as `*.txt`

|<img alt="image" src="https://github.com/user-attachments/assets/652bf28d-eba2-4a06-a267-1f9574a10b99" />|<img alt="image" src="https://github.com/user-attachments/assets/afeb2dd9-5763-47e3-b0f4-1b901294c926" />|
|:--|:--|
|`Exp. as txt` 버튼을 눌러 저장할 폴더 명 입력 후 `OK`버튼 클릭| 이미지 당 각각의 `*.txt` 파일이 설정한 위치에 저장됩니다.|
|<img  alt="image" src="https://github.com/user-attachments/assets/b539f2e9-422e-45b1-9300-a32421deb293" />|<img alt="image" src="https://github.com/user-attachments/assets/c3e0c674-9722-4341-9590-ca7077866228" />|저장할 `json`파일 명 입력 후 `Save`|
|설정한 위치에서 내보낸 `txt`파일들을 확인할 수 있습니다.|`txt`파일은 YOLO 형식으로 저장됩니다. `<class_id> <confidence> <x_center> <y_center> <width> <height>` 형식이며, <br>상자 좌표는 정규화된 `xywh`형식 (0에서 1사이) 입니다. 클래스 번호는 0부터 시작합니다.|

#### Instance Segmentation

Instance Segmentation Task의 경우 `*.json` 형식과, `*.txt` 형식, 그리고 `*.png` 형식으로 결과를 내보낼 수 있습니다.
* `JSON`파일의 경우 모든 결과가 하나의 파일에 통합되며, 이미지 파일 명을 `key`값으로 하여 `value`에 추론 결과 값 및 polygon point 들이 저장됩니다.
* `txt`파일의 경우 각 이미지에 대한 추론 결과가 각각의 `txt`파일에 생성되어 추론된 이미지 파일 개수만큼 결과 파일이 생성됩니다.
* `png`파일의 경우 각 이미지에 대한 추론 결과인 마스킹이 이미지로 저장됩니다. 
    * `class-wise`를 선택한 경우 클래스가 같은 객체의 마스크들을 하나의 이미지에 통합하여 저장합니다. 
    * `instance wise`인 경우 각각의 인스턴스의 마스크 이미지를 따로 저장합니다.

#### Export as `*.json`

|<img width="500" alt="image" src="https://github.com/user-attachments/assets/910a9ecc-2124-4495-ad49-e18a6f7284b7" />|<img width="500" alt="image" src="https://github.com/user-attachments/assets/b1a0e4dc-207d-43e1-941b-81480425e9c5" />|
|:--|:--|
|`Exp. as json`버튼을 눌러 저장할 폴더 명 입력 후 `OK`버튼 클릭|저장할 `json`파일 명 입력 후 `Save`|
|<img width="500" alt="image" src="https://github.com/user-attachments/assets/0f0b242f-f4e9-4b56-91e0-c59d89edf9e7" />|<img width="500" alt="image" src="https://github.com/user-attachments/assets/6b72860b-39a3-412d-8e4a-d0f70e7f604d" />|
|저장이 완료되면 결과 파일이 저장된 위치가 출력됩니다.|`*.json`파일의 구조입니다. `class_id`, `class_name`, <br>`confidence(score)`, `bbox`값과 함께 mask를 이루는 점들이 <br>`polygons`에 `[[x1, y1], [x2, y2], ..., [xn, yn]]` <br>형식으로 저장됩니다.|



#### Export as `*.txt`

|<img width="500" alt="image" src="https://github.com/user-attachments/assets/7379bcdf-66d6-4d53-8d7b-47d0195a30cf" />|<img width="500" alt="image" src="https://github.com/user-attachments/assets/496fe57f-8939-4bec-8277-930bcccb0a89" />|
|:--|:--|
|`Exp. as txt`버튼을 눌러 저장할 폴더 명 입력 후 `OK` 클릭|이미지 별로 각각의 `*.txt`파일이 설정한 위치에 저장됩니다.|
|<img width="500" alt="image" src="https://github.com/user-attachments/assets/ac2f5952-2f9d-416d-904f-9c6c713c631c" />|<img width="500" alt="image" src="https://github.com/user-attachments/assets/437dc4fd-3cff-4de9-9eaf-30c3a96beeb6" />|
|설정한 위치에서 내보낸 `txt`파일들을 확인할 수 있습니다.|`txt`파일은  YOLO 형식으로 저장됩니다. `<class_id>`<br>`<confidence> <x1> <y1> <x2> <y2> ... <xn> <yn>` <br>형식이며, 상자 좌표는 정규화된 0에서 1사이의 값 입니다. <br>클래스 번호는 0부터 시작합니다.|

#### Export masks (png)

|<img width="500" alt="image" src="https://github.com/user-attachments/assets/50c3464e-49c5-46b0-94e9-9a3208e56879" />|<img width="500" alt="image" src="https://github.com/user-attachments/assets/9b6395b9-32a4-4840-9c58-36fb9f75174f" />|
|:--|:--|
|`Exp. masks`버튼을 눌러 저장할 타입 선택 후 `OK`클릭|설정한 위치에서 각 이미지 파일명에 해당하는 폴더들을 확인할 수 있습니다.|

|<img width="1000" alt="image" src="https://github.com/user-attachments/assets/613671d0-8ed4-4d3d-8dce-3f29ab9818a4" />|
|:--|
|`Class-wise`를 선택한 경우 클래스 별로 분리되어 저장된 마스크 이미지를 확인할 수 있습니다.|
|<img width="1000" alt="image" src="https://github.com/user-attachments/assets/6c9daf75-d401-4cd2-9ada-074524ef284c" />|
|`Instance-wise`를 선택한 경우 인스턴스 별로 분리되어 저장된 마스크 이미지를 확인할 수 있습니다.|

---

**To Do**

- [ ] Video Inference 구현
- [ ] Real time Inference 구현
