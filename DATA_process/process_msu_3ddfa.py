import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm
import yaml
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
MSU_DIR    = SCRIPT_DIR / 'MSU-MFSD' / 'MSU-MFSD-Publish.zip'
OUTPUT_DIR = SCRIPT_DIR / 'MSU-MFSD' / 'processed_3ddfa'
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
    if not param_lst: return None, None

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    ver = _to_ctype(ver_lst[0].T)
    z = ver[:, 2]
    z_min, z_max = min(z), max(z)

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
    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
    
    margin_w = int((x_max - x_min) * 0.05)
    margin_h = int((y_max - y_min) * 0.05)
    x1, y1 = max(0, x_min - margin_w), max(0, y_min - margin_h)
    x2, y2 = min(w, x_max + margin_w), min(h, y_max + margin_h)

    return masked_bgr[y1:y2, x1:x2], depth_map[y1:y2, x1:x2]

def get_label_from_filename(stem):
    lower_stem = stem.lower()
    if 'real' in lower_stem: return 'real'
    if 'attack' in lower_stem: return 'attack'
    if 'ipad' in lower_stem or 'iphone' in lower_stem or 'print' in lower_stem or 'printed' in lower_stem:
        return 'attack'
    return 'real'

def parse_msu_face_file(face_path):
    entries = []
    with open(face_path, 'r') as f:
        for idx, line in enumerate(f):
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) >= 5:
                x, y, w, h = map(int, parts[1:5])
                if w == 0 or h == 0: continue
                margin_x, margin_y = int(w * 0.3), int(h * 0.3)
                x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
                x2, y2 = x + w + margin_x, y + h + margin_y
                entries.append({'frame_idx': idx, 'box': [x1, y1, x2, y2, 1.0]})
    return entries

def sample_frames(entries, n):
    if n <= 0 or len(entries) <= n: return entries
    idx = np.linspace(0, len(entries) - 1, n, dtype=int)
    return [entries[i] for i in idx]

def build_tasks(split_name, frames_per_video):
    subdirs = ['scene01'] if split_name == 'all' else [split_name]
    tasks = []
    for sdir in subdirs:
        scene_dir = MSU_DIR / sdir
        if not scene_dir.exists(): continue
        for label_dir in ['real', 'attack']:
            ldir = scene_dir / label_dir
            if not ldir.exists(): continue
            for video_path in sorted(list(ldir.glob('*.mp4')) + list(ldir.glob('*.mov'))):
                face_path = video_path.with_suffix('.face')
                if not face_path.exists(): continue
                out_dir = OUTPUT_DIR / sdir / label_dir / video_path.stem
                entries = parse_msu_face_file(face_path)
                if not entries: continue
                tasks.append({'video': video_path, 'label': label_dir, 'out_dir': out_dir, 'entries': entries, 'frames_per_video': frames_per_video})
    return tasks

import random

def process_video(task, tddfa):
    video_path, all_entries, out_dir, label = task['video'], task['entries'], task['out_dir'], task['label']
    frames_per_video = task['frames_per_video']
    is_real = (label == 'real')
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return 0, 0, f'[FAIL] cv2 could not open'

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
                if cropped_bgr.size != 0: cropped_depth = np.zeros(cropped_bgr.shape[:2], dtype=np.uint8)
            else: ok_depth += 1
        else:
            cropped_bgr, _ = make_masked_rgb_and_depth(frame, tddfa, box)
            if cropped_bgr is None:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped_bgr = frame[max(0, y1):y2, max(0, x1):x2]
            if cropped_bgr is not None and cropped_bgr.size != 0:
                cropped_depth = np.zeros(cropped_bgr.shape[:2], dtype=np.uint8)

        if cropped_bgr is not None and cropped_bgr.size != 0:
            rgb_out = cv2.resize(cropped_bgr, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            if len(cropped_depth.shape) == 3: cropped_depth = cropped_depth[:, :, 0]
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
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print('=' * 55)
    print(' MSU-MFSD Reprocess  (Multithreaded Fast)')
    print('=' * 55)

    tasks = build_tasks(args.split, args.frames_per_video)
    expected_frames = sum(min(len(t['entries']), args.frames_per_video) for t in tasks)

    print(f'\n[INFO] Building task list...')
    print(f'       Videos : {len(tasks)}')
    print(f'       Frames : ~{expected_frames}')

    if args.dry_run or not tasks: return

    print('\n[INFO] Loading TDDFA...')
    tddfa = init_tddfa()
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

    print(f'\nDone! Processed {len(tasks) - total_fail} videos. Saved {total_saved} frames.')
    if total_saved == expected_frames:
        print(f'[VERIFICATION SUCCESS] Exact expected frames ({expected_frames}) were successfully saved.')
    else:
        print(f'[VERIFICATION WARNING] Expected {expected_frames} frames but saved {total_saved}. Missing: {expected_frames - total_saved}')

if __name__ == '__main__':
    main()
