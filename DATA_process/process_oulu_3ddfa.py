"""
process_oulu_3ddfa.py
=====================
OULU-NPU reprocessing pipeline using 3DDFA_V2:
  1. Parse video lists from Train_files, Dev_files, Test_files.
  2. Parse bounding boxes from corresponding .txt files (frame_idx, x, y, w, h).
  3. Decode frames from .avi video.
  4. Run 3DDFA_V2 dense reconstruction on FULL image using bounding box.
  5. Apply convex-hull mask (CASIA style: face oval, black background).
  6. Crop tightly around 3D vertices, pad to square, resize to 256x256.
  7. Save masked RGB + grayscale depth (255=near, 0=bg). Attack -> zero depth.

Output structure mirrors input sets:
  d:/DATASET/OULU_NPU/processed_3ddfa/
    Train_files/
      real/
        1_1_01_1/
          frame_0000.jpg, frame_0000_depth.jpg
      attack/
        1_1_01_2/
          frame_0000.jpg, frame_0000_depth.jpg
    Dev_files/
    Test_files/

Usage:
    cd d:/DATASET/3DDFA_V2
    conda run -n Project python ../process_oulu_3ddfa.py --dry-run
    conda run -n Project python ../process_oulu_3ddfa.py --split Train_files
    conda run -n Project python ../process_oulu_3ddfa.py --split all
"""

import sys, os, argparse, subprocess, tempfile
import numpy as np
import cv2
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
TDDFA_DIR   = SCRIPT_DIR / '3DDFA_V2'
OULU_DIR    = SCRIPT_DIR / 'OULU_NPU'
OUTPUT_DIR  = OULU_DIR / 'processed_3ddfa'
CONFIG      = 'configs/mb1_120x120.yml'

OUTPUT_SIZE = 256

sys.path.insert(0, str(TDDFA_DIR))
os.chdir(TDDFA_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# TDDFA Core Logic
# ─────────────────────────────────────────────────────────────────────────────

def init_tddfa():
    from TDDFA import TDDFA
    cfg = yaml.load(open(CONFIG), Loader=yaml.SafeLoader)
    return TDDFA(gpu_mode=False, **cfg)

def make_masked_rgb_and_depth(img_bgr, tddfa, box):
    """
    Given FULL image and a face box [x1, y1, x2, y2, 1.0]:
      - Run 3DDFA dense reconstruction.
      - Build convex-hull face mask on full image.
      - Crop both masked RGB and depth tightly around the 3D vertices.
    """
    from Sim3DR import rasterize
    from utils.tddfa_util import _to_ctype

    h, w = img_bgr.shape[:2]

    try:
        param_lst, roi_lst = tddfa(img_bgr, [box])
        ver_lst = tddfa.recon_vers(param_lst, roi_lst, dense_flag=True)
    except Exception:
        return None, None

    ver   = ver_lst[0]
    tri   = tddfa.tri
    ver_T = _to_ctype(ver.T.astype(np.float32))

    # --- Convex hull mask ---
    pts = ver_T[:, :2].astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # --- Masked RGB ---
    masked_bgr = img_bgr.copy()
    masked_bgr[mask == 0] = 0

    # --- Z-buffer depth ---
    z      = ver_T[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    z_3ch  = np.repeat(z_norm[:, None], 3, axis=1).astype(np.float32)
    blank  = np.zeros((h, w, 3), dtype=np.uint8)
    d_rgb  = rasterize(ver_T, tri, z_3ch, bg=blank)
    d_f    = d_rgb[:, :, 0].astype(np.float32) / 255.0
    d_f[mask == 0] = 0.0
    face_v = d_f[mask > 0]
    if len(face_v) > 0:
        vmin, vmax = face_v.min(), face_v.max()
        if vmax > vmin:
            d_f[mask > 0] = (d_f[mask > 0] - vmin) / (vmax - vmin)
    depth_gray = (d_f * 255).clip(0, 255).astype(np.uint8)

    # --- Tight Crop around Mesh ---
    padding_ratio = 0.05
    xs = ver[0, :]
    ys = ver[1, :]
    x0 = max(0, int(xs.min()) - int(padding_ratio * (xs.max() - xs.min())))
    x1 = min(w, int(xs.max()) + int(padding_ratio * (xs.max() - xs.min())))
    y0 = max(0, int(ys.min()) - int(padding_ratio * (ys.max() - ys.min())))
    y1 = min(h, int(ys.max()) + int(padding_ratio * (ys.max() - ys.min())))
    
    # Make crop square
    cw = x1 - x0
    ch = y1 - y0
    if cw > ch:
        diff = cw - ch
        y0 = max(0, y0 - diff // 2)
        y1 = min(h, y1 + (diff - diff // 2))
    elif ch > cw:
        diff = ch - cw
        x0 = max(0, x0 - diff // 2)
        x1 = min(w, x1 + (diff - diff // 2))

    crop_rgb   = masked_bgr[y0:y1, x0:x1]
    crop_depth = depth_gray[y0:y1, x0:x1]
    return crop_rgb, crop_depth


# ─────────────────────────────────────────────────────────────────────────────
# OULU-NPU Parsing
# ─────────────────────────────────────────────────────────────────────────────

import math

def parse_bbox_file(txt_path):
    """
    OULU-NPU txt format: num_frame, x_eye_left, y_eye_left, x_eye_right, y_eye_right
    Returns list of dicts: { 'frame_idx': int, 'box': [x1, y1, x2, y2, 1.0] }
    """
    entries = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                idx = int(parts[0])
                xl, yl, xr, yr = map(int, parts[1:5])
                
                if xl == 0 and yl == 0 and xr == 0 and yr == 0:
                    continue # Failed to detect
                    
                d = math.hypot(xr - xl, yr - yl)
                if d == 0:
                    continue
                    
                cx = (xl + xr) / 2.0
                cy = (yl + yr) / 2.0
                
                w_box = 3.0 * d
                h_box = 3.0 * d
                
                x1 = max(0, int(cx - w_box / 2))
                y1 = max(0, int(cy - 0.4 * h_box))
                x2 = int(cx + w_box / 2)
                y2 = int(cy + 0.6 * h_box)
                
                entries.append({
                    'frame_idx': idx,
                    'box': [x1, y1, x2, y2, 1.0]
                })
    return entries

def sample_frames(entries, n):
    if n <= 0 or len(entries) <= n:
        return entries
    idx = np.linspace(0, len(entries) - 1, n, dtype=int)
    return [entries[i] for i in idx]

def build_tasks(split_name, frames_per_video):
    if split_name == 'all':
        subdirs = ['Train_files', 'Dev_files', 'Test_files']
    else:
        subdirs = [split_name]

    tasks = []
    for sdir in subdirs:
        video_dir = OULU_DIR / sdir
        if not video_dir.exists():
            continue

        # Each set folder contains a subfolder with the same name containing the actual files
        # e.g., Train_files/Train_files/1_1_01_1.avi
        inner_dir = video_dir / sdir
        if not inner_dir.exists():
            inner_dir = video_dir # fallback if directly inside

        for avi_path in sorted(inner_dir.glob('*.avi')):
            stem = avi_path.stem
            txt_path = avi_path.with_suffix('.txt')
            if not txt_path.exists():
                continue

            # OULU file name: Phone_Session_User_FileID
            parts = stem.split('_')
            if len(parts) < 4:
                continue
            
            file_id = int(parts[3])
            
            if file_id == 1: label = 'real'
            elif file_id == 2: label = 'print1'
            elif file_id == 3: label = 'print2'
            elif file_id == 4: label = 'replay1'
            elif file_id == 5: label = 'replay2'
            else: label = 'attack'
            
            out_dir = OUTPUT_DIR / sdir / label / stem

            entries = parse_bbox_file(txt_path)
            if not entries:
                continue

            entries = sample_frames(entries, frames_per_video)

            if out_dir.exists():
                done = len(list(out_dir.glob('*_depth.jpg')))
                if done >= len(entries):
                    continue

            tasks.append({
                'video': avi_path,
                'txt': txt_path,
                'label': label,
                'out_dir': out_dir,
                'entries': entries
            })
    return tasks

# ─────────────────────────────────────────────────────────────────────────────
# Process Video
# ─────────────────────────────────────────────────────────────────────────────

def process_video(task, tddfa):
    video_path = task['video']
    entries    = task['entries']
    out_dir    = task['out_dir']
    label      = task['label']
    is_real    = (label == 'real')

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, f'[FAIL] cv2 could not open: {video_path.name}'

    saved = ok_depth = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for entry in entries:
        fidx = entry['frame_idx']
        
        # OULU .txt indices are 0-based.
        target_idx = fidx
        if target_idx < 0 or target_idx >= total_frames:
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        box = entry['box'] # [x1, y1, x2, y2, 1.0]

        if is_real:
            cropped_bgr, cropped_depth = make_masked_rgb_and_depth(frame, tddfa, box)
            if cropped_bgr is None:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped_bgr = frame[max(0, y1):y2, max(0, x1):x2]
                if cropped_bgr.size == 0: continue
                cropped_depth = np.zeros(cropped_bgr.shape[:2], dtype=np.uint8)
            else:
                ok_depth += 1
        else:
            cropped_bgr, _ = make_masked_rgb_and_depth(frame, tddfa, box)
            if cropped_bgr is None:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped_bgr = frame[max(0, y1):y2, max(0, x1):x2]
                if cropped_bgr.size == 0: continue
            cropped_depth = np.zeros(cropped_bgr.shape[:2], dtype=np.uint8)
            ok_depth += 1

        # Resize to OUTPUT_SIZE x OUTPUT_SIZE
        rgb_out   = cv2.resize(cropped_bgr, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        depth_out = cv2.resize(cropped_depth, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)

        base = f'frame_{fidx:04d}'
        cv2.imwrite(str(out_dir / f'{base}.jpg'), rgb_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dir / f'{base}_depth.jpg'), depth_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    return saved, ok_depth, None

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='OULU-NPU Reprocess with 3DDFA oval-mask crop + depth.')
    parser.add_argument('--split', choices=['Train_files', 'Dev_files', 'Test_files', 'all'], default='all')
    parser.add_argument('--frames-per-video', type=int, default=25, help='Frames to sample per video (0=all)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print('=' * 55)
    print(' OULU-NPU Reprocess  (3DDFA oval mask + depth)')
    print('=' * 55)
    print(f' Output  : {OUTPUT_DIR}')
    print(f' Split   : {args.split}')
    print(f' Frames  : {args.frames_per_video or "all"} / video')
    print('=' * 55)

    print('\n[INFO] Building task list...')
    tasks = build_tasks(args.split, args.frames_per_video)
    n_real = sum(1 for t in tasks if t['label'] == 'real')
    n_attack = sum(1 for t in tasks if t['label'] != 'real')
    total_frames = sum(len(t['entries']) for t in tasks)
    print(f'       Videos : {len(tasks)}  ({n_real} real, {n_attack} attack)')
    print(f'       Frames : ~{total_frames:,}')

    if args.dry_run:
        print('[DRY RUN] No files written.\n')
        return

    if not tasks:
        print('[INFO] Nothing to process.\n')
        return

    print('\n[INFO] Loading TDDFA...')
    tddfa = init_tddfa()
    print('[OK] TDDFA ready\n')

    total_saved = total_ok_depth = total_fail = 0
    bar = tqdm(tasks, desc='Videos', unit='vid', ascii=True)
    for task in bar:
        saved, ok_d, err = process_video(task, tddfa)
        total_saved += saved
        total_ok_depth += ok_d
        if err:
            total_fail += 1
            tqdm.write(err)
        else:
            tqdm.write(f'  [OK] {task["video"].stem}: {saved} frames')

    print(f'\n{"="*55}')
    print(f' Done!')
    print(f'  Videos processed : {len(tasks) - total_fail}')
    print(f'  Frames saved     : {total_saved:,}  (RGB + depth pairs)')
    print(f'  Depth ok (3DDFA) : {total_ok_depth:,}')
    print(f'  Video failures   : {total_fail}')
    print(f'{"="*55}\n')

if __name__ == '__main__':
    main()
