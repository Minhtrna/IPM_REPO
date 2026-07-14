# MSU-MFSD Data Processing Guide

This folder contains the script used to reprocess the MSU-MFSD dataset into 3DDFA-based face crops with paired depth maps.

## What the script does

`process_msu_3ddfa.py` performs the following steps for each annotated video frame:

1. Reads the frame-level face bounding boxes from the `.face` file.
2. Decodes the video frames with FFmpeg.
3. Crops the face region with extra margin.
4. Runs 3DDFA dense reconstruction on the full frame.
5. Builds a convex-hull face mask and blacks out the background.
6. Saves two outputs per frame:
   - masked RGB image
   - grayscale depth image

The output is organized as RGB/depth pairs and mirrors the train/test and real/attack split structure.

## Required folder layout

The script expects this project layout:

```text
IPM_REPO/
  DATA_process/
    process_msu_3ddfa.py
    3DDFA_V2/
    MSU-MFSD/
      MSU-MFSD-Publish.zip/
        scene01/
          real/
          attack/
        train_sub_list.txt
        test_sub_list.txt
        ffmpeg/
          bin/
            ffmpeg.exe
            ffprobe.exe
```

Important:

- Keep the `MSU-MFSD-Publish.zip` folder name exactly as shown, because the script uses that path directly.
- The script writes processed files into `MSU-MFSD/MSU-MFSD-Publish.zip/processed_3ddfa/`.

## Requirements

- Python environment with the project dependencies installed.
- `ffmpeg.exe` and `ffprobe.exe` available in the dataset folder path used by the script.
- The 3DDFA_V2 codebase placed in `DATA_process/3DDFA_V2/`.
- MSU-MFSD videos, `.face` annotations, and split files placed in the expected folder structure.

## How the input data is organized

For each video, the script expects a matching `.face` file with per-frame face bounding boxes.

The dataset split files are used to decide whether a video belongs to the train or test split:

- `train_sub_list.txt`
- `test_sub_list.txt`

The script processes both `real` and `attack` videos under `scene01/`.

## Output structure

Processed files are written here:

```text
MSU-MFSD/MSU-MFSD-Publish.zip/processed_3ddfa/
  train/
    real/<video_name>/frame_XXXX.jpg
    real/<video_name>/frame_XXXX_depth.jpg
    attack/<video_name>/frame_XXXX.jpg
    attack/<video_name>/frame_XXXX_depth.jpg
  test/
    real/<video_name>/frame_XXXX.jpg
    real/<video_name>/frame_XXXX_depth.jpg
    attack/<video_name>/frame_XXXX.jpg
    attack/<video_name>/frame_XXXX_depth.jpg
```

File naming:

- `frame_XXXX.jpg` = masked RGB crop
- `frame_XXXX_depth.jpg` = grayscale depth map

Image size:

- Final output size is `256 x 256`

## Processing rules

- Real samples use the 3DDFA depth map.
- Attack samples save a zero-filled depth image.
- The script samples a fixed number of frames per video by default.
- If the 3DDFA reconstruction fails for a real frame, the script falls back to a simple bbox crop and a blank depth image.

## Run instructions

### 1. Open the project folder

Run the script from inside `DATA_process` so the relative paths resolve correctly.

### 2. Dry run first

Use this to verify that the dataset structure is correct before writing files:

```bash
python process_msu_3ddfa.py --dry-run
```

### 3. Process the dataset

Process both train and test splits:

```bash
python process_msu_3ddfa.py
```

Process only one split:

```bash
python process_msu_3ddfa.py --split train
python process_msu_3ddfa.py --split test
```

Change the number of sampled frames per video:

```bash
python process_msu_3ddfa.py --frames-per-video 25
python process_msu_3ddfa.py --frames-per-video 0
```

`--frames-per-video 0` means process all annotated frames.

## Example Conda command

If you use Conda, run the script with the environment that contains the dependencies:

```bash
conda run -n Project python process_msu_3ddfa.py
```

Replace `Project` with your actual Conda environment name.

## Expected console output

The script prints:

- source and output paths
- selected splits
- number of videos found
- approximate frame count
- per-video success or failure messages
- final totals for processed videos and saved frames

## Troubleshooting

- If no videos are found, check the MSU-MFSD folder path and confirm the `scene01/real` and `scene01/attack` folders exist.
- If the script exits without writing files, run `--dry-run` first and verify the folder structure.
- If FFmpeg fails, confirm that `ffmpeg.exe` and `ffprobe.exe` exist at the expected path.
- If 3DDFA fails, confirm that `DATA_process/3DDFA_V2/` is present and that the Python environment can import its modules.

## Notes

- The script skips videos that are already fully processed.
- Output images are saved as JPEG files.
- The processing order is determined by the split lists and the client ID parsed from the video name.