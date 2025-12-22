# DLSW

## Installation 
### Prerequisites

- Python 3.10 or higher
- PyTorch 2.8 or higher
- CUDA-compatible GPU with CUDA 12.8 or higher
- anaconda or miniconda

1. **Create a new Conda environment:**

```bash
conda create -n dlsw python=3.10 -y
conda activate dlsw
```

2. **Install PyTorch with CUDA support:**

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

3. **Clone the repository and install the package:**

```bash
git clone https://github.com/zihos/DLSW.git
cd DLSW
pip install PySide6
pip install pillow ultralytics
```

4. **Install additional dependencies for example notebooks or development:**

```bash
# Required packages for running models in ONNX and OpenVINO formats
pip install opencv-python pycocotools matplotlib onnxruntime onnx openvino

# Required packages for running smart polygon mode
pip install git+https://github.com/facebookresearch/segment-anything.git

```

5. **Click the links below to download the checkpoint for the SAM model.**

    Copy the checkpoint to `DLSW/dl_software/models/`

    **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**

## Run
```bash
python main.py
```
