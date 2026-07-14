# 3DDFA_V2 Face & Depth Map Extraction Pipeline

This repository contains highly optimized Python scripts designed to extract cropped facial RGB images and generate corresponding 3D depth maps for two major Facial Anti-Spoofing (FAS) datasets: **OULU-NPU** and **MSU-MFSD**.

The pipeline utilizes the **3DDFA_V2** framework (with C++ Cython acceleration) to reconstruct 3D faces from 2D frames, accurately rendering pixel-level depth maps.

---

## 1. Expected Directory Structure

When migrating this project to a new machine, ensure your workspace is organized as follows. Replace `<YOUR_WORKSPACE_PATH>` with your actual directory path:

```text
<YOUR_WORKSPACE_PATH>/
│
├── process_oulu_3ddfa.py    # Processing script for OULU-NPU
├── process_msu_3ddfa.py     # Processing script for MSU-MFSD
│
├── 3DDFA_V2/                # Cloned 3DDFA_V2 repository
│   ├── FaceBoxes/
│   ├── Sim3DR/
│   └── ...
│
├── OULU_NPU/                # Original OULU dataset
│   ├── Train_files/
│   ├── Dev_files/
│   ├── Test_files/
│   └── processed_3ddfa/     # Auto-generated Output directory
│
└── MSU-MFSD/                # Original MSU dataset
    ├── MSU-MFSD-Publish.zip/ # (Extracted MSU directory structure)
    │   └── scene01/
    └── processed_3ddfa/     # Auto-generated Output directory
```

---

## 2. Installation & Setup

When deploying to a new Windows machine, please follow these steps to prepare your environment and compile the C++ extensions.

### Step 1: Prerequisites
1. Install **Anaconda** or **Miniconda**.
2. Install **Visual Studio C++ Build Tools** (Required for compiling C++ on Windows). Open the Visual Studio Installer and select the "Desktop development with C++" workload.

### Step 2: Create Conda Environment
Open Anaconda Prompt (or Terminal) and run:
```bash
cd <YOUR_WORKSPACE_PATH>
conda create -n Project python=3.12 -y
conda activate Project
pip install torch torchvision numpy opencv-python pyyaml tqdm cython
```

### Step 3: Build C++ Cython Modules
Since compiled `.pyd` files (like `cpu_nms.cp312-win_amd64.pyd`) are tightly bound to the CPU architecture and the exact Python version, you must compile them on the new machine. It is strongly recommended to **delete any old `.pyd` and `.so` files** in the repository before building.

1. **Build FaceBoxes (NMS):**
```bash
cd <YOUR_WORKSPACE_PATH>\3DDFA_V2\FaceBoxes\utils
python build.py build_ext --inplace
```
*(Note: Linux flags like `-Wno-cpp` have already been removed from `build.py` to ensure MSVC compatibility).*

2. **Build Sim3DR:**
```bash
cd <YOUR_WORKSPACE_PATH>\3DDFA_V2\Sim3DR
python setup.py build_ext --inplace
```
*(Similarly, the `-std=c++11` flag has been removed from `setup.py`).*

---

## 3. Usage Instructions

Return to the root directory before running the processing scripts.

### Process OULU-NPU
```bash
cd <YOUR_WORKSPACE_PATH>
conda activate Project

# Run the full pipeline (Train, Dev, Test), extracting 25 frames per video
python process_oulu_3ddfa.py --split all --frames-per-video 25

# Process a specific subset only
python process_oulu_3ddfa.py --split Train_files

# Dry-run (lists total tasks without actually processing/saving images)
python process_oulu_3ddfa.py --dry-run
```

### Process MSU-MFSD
```bash
cd <YOUR_WORKSPACE_PATH>
conda activate Project

# Run the full pipeline
python process_msu_3ddfa.py --split all --frames-per-video 25
```

### Advanced Arguments
- `--workers <int>`: Number of parallel threads. Default is 4. Increase this depending on your CPU/GPU capabilities.
- `--frames-per-video <int>`: The target number of frames to extract from each video. Default is 25.

---

## 4. Output Format

The output RGB images and 1-channel Depth Maps (resized to 256x256, strictly cropped around the face bounding box) will be generated at:
- OULU: `<YOUR_WORKSPACE_PATH>\OULU_NPU\processed_3ddfa\...`
- MSU: `<YOUR_WORKSPACE_PATH>\MSU-MFSD\processed_3ddfa\...`

Inside the output folder, files are automatically sorted into standard subdirectories (`real`, `attack`, `print1`, `replay1`, etc.). 

The naming convention uses exactly 4-digit indices tracking the successful extractions:
- `frame_0000.jpg` (RGB)
- `frame_0000_depth.jpg` (Depth Map)
- `frame_0001.jpg`
- `frame_0001_depth.jpg`
- ... up to `frame_0024.jpg` (for 25 frames).
