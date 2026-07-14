import os
import sys
import math
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm
import yaml
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
OULU_DIR   = SCRIPT_DIR / 'OULU_NPU'
OUTPUT_DIR = SCRIPT_DIR / 'OULU_NPU' / 'processed_3ddfa'
TDDFA_DIR  = SCRIPT_DIR / '3DDFA_V2'

OUTPUT_SIZE = 256

sys.path.insert(0, str(TDDFA_DIR))

def init_tddfa():
    cwd = os.getcwd()
    try:
        os.chdir(str(TDDFA_DIR))
        from FaceBoxes import FaceBoxes
        from TDDFA import TDDFA
        
        cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        tddfa = TDDFA(gpu_mode=True, **cfg)
        return tddfa
    finally:
        os.chdir(cwd)

def make_masked_rgb_and_depth(img_bgr, tddfa, box):
    from Sim3DR import rasterize
    from utils.tddfa_util import _to_ctype
    
    boxes = [box]
    with torch.no_grad():
        param_lst, roi_box_lst = tddfa(img_bgr, boxes)
    if not param_lst:
        return None, None

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    ver = _to_ctype(ver_lst[0].T)
    z = ver[:, 2]
    z_min, z_max = min(z), max(z)
    
    # Normalize z so that the closest points (z_min, nose) map to 1.0 (white 255)
    # and the farthest points (z_max, cheeks) map to 10/255.0 (dark gray 10).
    # Sim3DR uses 'ver' for Z-buffering (occlusion), so inverting 'z_norm' as colors is perfectly safe!
    z_norm = (z - z_min) / (z_max - z_min)
    z_norm = z_norm * (245.0 / 255.0) + (10.0 / 255.0)
    z_norm = np.repeat(z_norm[:, np.newaxis], 3, axis=1).astype(np.float32)
    
    overlap = np.zeros_like(img_bgr, dtype=np.uint8)
    depth_map = rasterize(ver, tddfa.tri, z_norm, bg=overlap)
    
    mask = depth_map[:, :, 0] > 0
    mask_uint8 = mask.astype(np.uint8)
    masked_bgr = img_bgr.copy()
    masked_bgr[mask_uint8 == 0] = 0

    x_min, x_max = np.min(ver_lst[0][0, :]), np.max(ver_lst[0][0, :])
    y_min, y_max = np.min(ver_lst[0][1, :]), np.max(ver_lst[0][1, :])

    h, w = img_bgr.shape[:2]
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(w, int(x_max))
    y_max = min(h, int(y_max))
    
    margin_w = int((x_max - x_min) * 0.05)
    margin_h = int((y_max - y_min) * 0.05)
    
    x1 = max(0, x_min - margin_w)
    y1 = max(0, y_min - margin_h)
    x2 = min(w, x_max + margin_w)
    y2 = min(h, y_max + margin_h)

    cropped_bgr = masked_bgr[y1:y2, x1:x2]
    cropped_depth = depth_map[y1:y2, x1:x2]
    
    return cropped_bgr, cropped_depth

def parse_bbox_file(txt_path):
    entries = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                idx = int(parts[0])
                xl, yl, xr, yr = map(int, parts[1:5])
                if xl == 0 and yl == 0 and xr == 0 and yr == 0: continue
                d = math.hypot(xr - xl, yr - yl)
                if d == 0: continue
                cx = (xl + xr) / 2.0
                cy = (yl + yr) / 2.0
                w_box = 3.0 * d
                h_box = 3.0 * d
                x1 = max(0, int(cx - w_box / 2))
                y1 = max(0, int(cy - 0.4 * h_box))
                x2 = int(cx + w_box / 2)
                y2 = int(cy + 0.6 * h_box)
                entries.append({'frame_idx': idx, 'box': [x1, y1, x2, y2, 1.0]})
    return entries

def sample_frames(entries, n):
    if n <= 0 or len(entries) <= n: return entries
    idx = np.linspace(0, len(entries) - 1, n, dtype=int)
    return [entries[i] for i in idx]

def build_tasks(split_name, frames_per_video):
    subdirs = ['Train_files', 'Dev_files', 'Test_files'] if split_name == 'all' else [split_name]
    tasks = []
    for sdir in subdirs:
        video_dir = OULU_DIR / sdir
        if not video_dir.exists(): continue
        inner_dir = video_dir / sdir
        if not inner_dir.exists(): inner_dir = video_dir

        for avi_path in sorted(inner_dir.glob('*.avi')):
            stem = avi_path.stem
            txt_path = avi_path.with_suffix('.txt')
            if not txt_path.exists(): continue

            parts = stem.split('_')
            if len(parts) < 4: continue
            
            file_id = int(parts[3])
            if file_id == 1: label = 'real'
            elif file_id == 2: label = 'print1'
            elif file_id == 3: label = 'print2'
            elif file_id == 4: label = 'replay1'
            elif file_id == 5: label = 'replay2'
            else: label = 'attack'
            
            out_dir = OUTPUT_DIR / sdir / label / stem
            entries = parse_bbox_file(txt_path)
            if not entries: continue
            if not entries: continue
            tasks.append({
                'video': avi_path,
                'txt': txt_path,
                'label': label,
                'out_dir': out_dir,
                'entries': entries,
                'frames_per_video': frames_per_video
            })
    return tasks

import random

def process_video(task, tddfa):
    video_path = task['video']
    all_entries = task['entries']
    out_dir = task['out_dir']
    label = task['label']
    frames_per_video = task['frames_per_video']
    is_real = (label == 'real')

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, f'[FAIL] cv2 could not open: {video_path.name}'

    n_entries = len(all_entries)
    if n_entries == 0: return 0, 0, None

    if n_entries <= frames_per_video:
        queue = list(range(n_entries))
        remaining_indices = []
    else:
        queue = np.linspace(0, n_entries - 1, frames_per_video, dtype=int).tolist()
        remaining_indices = list(set(range(n_entries)) - set(queue))
        random.shuffle(remaining_indices)

    saved = ok_depth = 0

    while queue and saved < frames_per_video:
        idx = queue.pop(0)
        entry = all_entries[idx]
        frame_idx = entry['frame_idx']

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            if remaining_indices: queue.append(remaining_indices.pop())
            continue

        box = entry['box']
        
        if is_real:
            cropped_bgr, cropped_depth = make_masked_rgb_and_depth(frame, tddfa, box)
            if cropped_bgr is None:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped_bgr = frame[max(0, y1):y2, max(0, x1):x2]
                if cropped_bgr.size != 0:
                    cropped_depth = np.zeros(cropped_bgr.shape[:2], dtype=np.uint8)
            else:
                ok_depth += 1
        else:
            cropped_bgr, _ = make_masked_rgb_and_depth(frame, tddfa, box)
            if cropped_bgr is None:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped_bgr = frame[max(0, y1):y2, max(0, x1):x2]
            if cropped_bgr is not None and cropped_bgr.size != 0:
                cropped_depth = np.zeros(cropped_bgr.shape[:2], dtype=np.uint8)

        if cropped_bgr is not None and cropped_bgr.size != 0:
            rgb_out = cv2.resize(cropped_bgr, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            if len(cropped_depth.shape) == 3:
                cropped_depth = cropped_depth[:, :, 0] # ensure 1 channel
            depth_out = cv2.resize(cropped_depth, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)
            
            cv2.imwrite(str(out_dir / f'frame_{saved:04d}.jpg'), rgb_out)
            cv2.imwrite(str(out_dir / f'frame_{saved:04d}_depth.jpg'), depth_out)
            saved += 1
        else:
            if remaining_indices: queue.append(remaining_indices.pop())

    cap.release()
    return saved, ok_depth, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all')
    parser.add_argument('--frames-per-video', type=int, default=25)
    parser.add_argument('--workers', type=int, default=4, help='Number of threads')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print('=' * 55)
    print(' OULU-NPU Reprocess  (Multithreaded Fast)')
    print('=' * 55)

    tasks = build_tasks(args.split, args.frames_per_video)
    print(f'[INFO] Found {len(tasks)} videos to process.')
    if args.dry_run or not tasks: return

    print('\n[INFO] Loading TDDFA...')
    tddfa = init_tddfa() # TDDFA is thread-safe for inference if using CPU or multiple streams
    print('[OK] TDDFA ready\n')

    total_saved = total_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, task, tddfa): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing', ascii=True):
            try:
                saved, ok_d, err = future.result()
                total_saved += saved
                if err: total_fail += 1
            except Exception as e:
                total_fail += 1

    expected_frames = sum(min(len(t['entries']), args.frames_per_video) for t in tasks)
    print(f'\nDone! Processed {len(tasks) - total_fail} videos. Saved {total_saved} frames.')
    if total_saved == expected_frames:
        print(f'[VERIFICATION SUCCESS] Exact expected frames ({expected_frames}) were successfully saved.')
    else:
        print(f'[VERIFICATION WARNING] Expected {expected_frames} frames but saved {total_saved}. Missing: {expected_frames - total_saved}')

if __name__ == '__main__':
    main()
