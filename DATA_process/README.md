# 3DDFA Face Cropping & Depth Generation Pipeline

This repository contains the data processing pipeline for generating tightly cropped RGB face images and corresponding 3DDFA-aligned depth maps (Z-buffers) for Face Anti-Spoofing datasets.

Currently supported datasets:
- **MSU-MFSD**
- **OULU-NPU**

## 1. Directory Structure
For the scripts to work seamlessly, ensure your workspace is structured as follows:

```text
d:\DATASET\
│
├── 3DDFA_V2/                  # Cloned from https://github.com/cleardusk/3DDFA_V2
│   ├── FaceBoxes/             # FaceBoxes submodule (must be built via sh ./build.sh)
│   ├── configs/               # Configurations (e.g. mb1_120x120.yml)
│   ├── weights/               # Pre-trained models (.pth)
│   └── ...                    # Other 3DDFA files
│
├── MSU-MFSD/                  # Original MSU-MFSD Dataset
│   ├── scene01/
│   │   ├── real/
│   │   │   ├── *.mp4          # Video files
│   │   │   └── *.mp4.face     # MSU bounding box annotation files
│   │   └── attack/
│   └── ...
│
├── OULU_NPU/                  # Original OULU-NPU Dataset
│   ├── Train_files/           
│   │   ├── Train_files/       # Raw avi and txt files
│   │   │   ├── *.avi
│   │   │   └── *.txt          # OULU annotation (eye coordinates)
│   ├── Dev_files/
│   └── Test_files/
│
├── process_msu_3ddfa.py       # Processing script for MSU-MFSD
└── process_oulu_3ddfa.py      # Processing script for OULU-NPU
```

## 2. Setting Up 3DDFA_V2
The scripts heavily depend on [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) for dense 3D facial geometry alignment and mask extraction.

1. **Clone the repository:** Clone `3DDFA_V2` directly into your `DATASET` folder.
   ```bash
   cd d:\DATASET
   git clone https://github.com/cleardusk/3DDFA_V2.git
   cd 3DDFA_V2
   ```
2. **Build internal components:** Refer to the official 3DDFA_V2 documentation to compile FaceBoxes and Sim3DR (usually via `sh ./build.sh`).
3. **Environment:** Ensure your python environment has `torch`, `torchvision`, `cv2` (opencv-python), `numpy`, `scipy`, `tqdm`, and `yaml`.

## 3. Usage Instructions

The scripts are designed to be run directly from the `DATASET` root directory. The pipeline will automatically navigate into the 3DDFA directory internally.

### Processing MSU-MFSD
Extracts tightly cropped frames and depth maps for the entire MSU-MFSD dataset.
```bash
conda activate Project
cd d:\DATASET

# Run on all subdirectories
python process_msu_3ddfa.py --split all --frames-per-video 25
```
**Output:** Results are saved to `d:\DATASET\MSU-MFSD\processed_3ddfa\`.

### Processing OULU-NPU
Extracts frames and depth maps for the OULU-NPU dataset. **Note:** OULU annotations use eye coordinates; this script automatically interpolates them into a robust bounding box before applying the 3DDFA mask.
```bash
conda activate Project
cd d:\DATASET

# Run on the whole dataset (Train, Dev, Test)
python process_oulu_3ddfa.py --split all --frames-per-video 25

# Or run on a specific split
python process_oulu_3ddfa.py --split Train_files --frames-per-video 25
```
**Output:** Results are saved to `d:\DATASET\OULU_NPU\processed_3ddfa\`. The script automatically separates attacks into specific directories (`print1`, `print2`, `replay1`, `replay2`) based on the OULU-NPU `file_id` format to strictly support protocol cross-evaluations.

## 4. Key Features
- **Intelligent Resumption:** If execution is interrupted (e.g. `Ctrl+C`), the scripts will automatically skip fully processed videos upon restart.
- **Pure OpenCV Pipeline:** Video decoding relies purely on `cv2.VideoCapture` combined with direct frame-seek (`CAP_PROP_POS_FRAMES`), avoiding slow FFMPEG shell executions and temporary images.
- **Unified Format:** Outputs consistently masked $256 \times 256$ RGB frames and corresponding 8-bit depth maps with a black background (`0`), perfectly aligned to the facial geometry.
