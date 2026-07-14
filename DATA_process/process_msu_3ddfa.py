"""
process_msu_3ddfa.py
====================
Full MSU-MFSD reprocessing pipeline:
  1. Read .face bbox annotations
  2. Decode frames from video with ffmpeg
  3. Crop face with margin -> run 3DDFA dense mesh
  4. Apply convex-hull mask (CASIA style: face oval, black background)
  5. Save masked RGB + grayscale depth (255=near, 0=bg)

Output structure (mirrors processed_faces_256x256):
  processed_3ddfa/
    train/
      real/  <video_name>/frame_XXXX.jpg + frame_XXXX_depth.jpg
      attack/<video_name>/frame_XXXX.jpg + frame_XXXX_depth.jpg
    test/
      ...

Convention:
  - Real   -> 3DDFA depth (Z-buffer grayscale)
  - Attack -> zero depth (black)

Usage:
    cd d:/DATASET/3DDFA_V2
    conda run -n Project python ../process_msu_3ddfa.py
    conda run -n Project python ../process_msu_3ddfa.py --split train
    conda run -n Project python ../process_msu_3ddfa.py --frames-per-video 25
    conda run -n Project python ../process_msu_3ddfa.py --dry-run
"""

import sys, os, csv, re, argparse, subprocess, tempfile
import numpy as np
import cv2
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
TDDFA_DIR   = SCRIPT_DIR / '3DDFA_V2'
MSU_DIR     = SCRIPT_DIR / 'MSU-MFSD' / 'MSU-MFSD-Publish.zip'
REAL_DIR    = MSU_DIR / 'scene01' / 'real'
ATTACK_DIR  = MSU_DIR / 'scene01' / 'attack'
TRAIN_LIST  = MSU_DIR / 'train_sub_list.txt'
TEST_LIST   = MSU_DIR / 'test_sub_list.txt'
FFMPEG_BIN  = MSU_DIR / 'ffmpeg' / 'bin' / 'ffmpeg.exe'
FFPROBE_BIN = MSU_DIR / 'ffmpeg' / 'bin' / 'ffprobe.exe'
OUTPUT_DIR  = SCRIPT_DIR / 'MSU-MFSD' / 'MSU-MFSD-Publish.zip' / 'processed_3ddfa'
CONFIG      = 'configs/mb1_120x120.yml'
CLIENT_RE   = re.compile(r'client(\d{3})')

OUTPUT_SIZE = 256   # final image size (square canvas, oval face inside)
MARGIN      = 0.3   # extra margin around .face bbox before 3DDFA

sys.path.insert(0, str(TDDFA_DIR))
os.chdir(TDDFA_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# TDDFA
# ─────────────────────────────────────────────────────────────────────────────

def init_tddfa():
    from TDDFA import TDDFA
    cfg = yaml.load(open(CONFIG), Loader=yaml.SafeLoader)
    return TDDFA(gpu_mode=False, **cfg)


def make_masked_rgb_and_depth(img_bgr, tddfa, box):
    """
    Given the FULL original image and a face box:
      - Run 3DDFA dense reconstruction using the box
      - Build convex-hull face mask on the full image
      - Crop both masked RGB and depth tightly around the 3D vertices (like debug script)
      - Return (cropped_masked_bgr, cropped_depth_gray)
      - Returns (None, None) on failure
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

    # --- Convex hull face mask on full image ---
    pts = ver_T[:, :2].astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # --- Masked RGB ---
    masked_bgr = img_bgr.copy()
    masked_bgr[mask == 0] = 0

    # --- Z-buffer depth on full image ---
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

    # --- Crop tightly around 3D face vertices (like debug script) ---
    padding_ratio = 0.05
    xs = ver[0, :]
    ys = ver[1, :]
    x0 = max(0, int(xs.min()) - int(padding_ratio * (xs.max() - xs.min())))
    x1 = min(w, int(xs.max()) + int(padding_ratio * (xs.max() - xs.min())))
    y0 = max(0, int(ys.min()) - int(padding_ratio * (ys.max() - ys.min())))
    y1 = min(h, int(ys.max()) + int(padding_ratio * (ys.max() - ys.min())))
    
    # Make the crop region square to avoid stretching when resizing to 256x256
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

    crop_rgb = masked_bgr[y0:y1, x0:x1]
    crop_depth = depth_gray[y0:y1, x0:x1]

    return crop_rgb, crop_depth


# ─────────────────────────────────────────────────────────────────────────────
# Video / annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ids(path):
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(line))
    return ids


def parse_face_file(face_path):
    entries = []
    with open(face_path) as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            row = [x.strip() for x in row]
            entries.append({
                'frame_idx': int(row[0]),
                'left': int(row[1]), 'top': int(row[2]),
                'right': int(row[3]), 'bottom': int(row[4]),
            })
    return entries


def get_client_id(name):
    m = CLIENT_RE.search(name)
    return int(m.group(1)) if m else None


def discover_videos(directory):
    vids = []
    for ext in ('*.mov', '*.mp4'):
        vids.extend(directory.glob(ext))
    return sorted(vids)


def get_fps(video_path):
    try:
        cmd = [str(FFPROBE_BIN), '-v', 'error',
               '-select_streams', 'v:0',
               '-show_entries', 'stream=avg_frame_rate',
               '-of', 'csv=p=0', str(video_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and '/' in r.stdout.strip():
            n, d = r.stdout.strip().split('/')
            return float(n) / float(d)
    except Exception:
        pass
    return None


def decode_video(video_path, tmp_dir, fps):
    stem = video_path.stem
    cmd = [str(FFMPEG_BIN), '-y', '-i', str(video_path)]
    if fps is not None:
        cmd += ['-r', f'{fps:.2f}']
    cmd.append(str(tmp_dir / f'{stem}_%03d.bmp'))
    subprocess.run(cmd, capture_output=True, timeout=600)
    return len(list(tmp_dir.glob('*.bmp')))


def is_rotated_180(video_path):
    try:
        cmd = [str(FFPROBE_BIN), '-show_streams', str(video_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return 'TAG:rotate=180' in r.stdout
    except Exception:
        return False


def sample_frames(entries, n):
    if n <= 0 or len(entries) <= n:
        return entries
    idx = np.linspace(0, len(entries) - 1, n, dtype=int)
    return [entries[i] for i in idx]


def crop_with_margin(img_pil, left, top, right, bottom, margin):
    w, h = img_pil.size
    mx = int((right - left) * margin)
    my = int((bottom - top) * margin)
    l = max(0, left  - mx)
    t = max(0, top   - my)
    r = min(w, right + mx)
    b = min(h, bottom + my)
    return img_pil.crop((l, t, r, b))


def pad_to_square(img_bgr):
    """Pad shorter dimension to make square (center, black)."""
    h, w = img_bgr.shape[:2]
    if h == w:
        return img_bgr
    s = max(h, w)
    out = np.zeros((s, s, 3), dtype=img_bgr.dtype)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    out[y0:y0+h, x0:x0+w] = img_bgr
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build task list
# ─────────────────────────────────────────────────────────────────────────────

def build_tasks(split, frames_per_video):
    train_ids = load_ids(TRAIN_LIST)
    test_ids  = load_ids(TEST_LIST)

    type_dirs = [('real', REAL_DIR), ('attack', ATTACK_DIR)]
    tasks = []

    for label, src_dir in type_dirs:
        for video_path in discover_videos(src_dir):
            cid = get_client_id(video_path.name)
            if cid is None:
                continue

            if split == 'train' and cid not in train_ids:
                continue
            if split == 'test' and cid not in test_ids:
                continue
            if split == 'all':
                if cid in train_ids:
                    split_name = 'train'
                elif cid in test_ids:
                    split_name = 'test'
                else:
                    continue
            else:
                split_name = split

            face_file = video_path.parent / (video_path.stem + '.face')
            if not face_file.exists():
                continue

            entries = parse_face_file(face_file)
            if not entries:
                continue

            entries = sample_frames(entries, frames_per_video)

            out_dir = OUTPUT_DIR / split_name / label / video_path.stem

            # Check already done
            if out_dir.exists():
                done = len(list(out_dir.glob('frame_*.jpg')))
                if done >= len(entries):
                    continue

            tasks.append({
                'video':      video_path,
                'face_file':  face_file,
                'label':      label,
                'split':      split_name,
                'entries':    entries,
                'out_dir':    out_dir,
            })

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Process one video
# ─────────────────────────────────────────────────────────────────────────────

def process_video(task, tddfa, zero_depth):
    video_path = task['video']
    entries    = task['entries']
    out_dir    = task['out_dir']
    label      = task['label']
    is_real    = (label == 'real')

    out_dir.mkdir(parents=True, exist_ok=True)
    rotated = is_rotated_180(video_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fps = get_fps(video_path)
        n_raw = decode_video(video_path, tmp_path, fps)
        if n_raw == 0:
            return 0, 0, f'[FAIL] decode: {video_path.name}'

        saved = ok_depth = 0
        for entry in entries:
            fidx = entry['frame_idx']
            stem = video_path.stem
            bmp  = tmp_path / f'{stem}_{fidx + 1:03d}.bmp'
            if not bmp.exists():
                continue

            try:
                img_pil = Image.open(bmp)
                if rotated:
                    img_pil = img_pil.rotate(180)
                full_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception:
                continue

            # Original bbox from .face
            box = [entry['left'], entry['top'], entry['right'], entry['bottom'], 1.0]

            # 3DDFA mask + depth (handled inside on full image, returning tight crop)
            if is_real:
                cropped_bgr, cropped_depth = make_masked_rgb_and_depth(full_bgr, tddfa, box)
                if cropped_bgr is None:
                    # Fallback if TDDFA fails: fallback crop using just the .face bbox
                    l, t, r, b = entry['left'], entry['top'], entry['right'], entry['bottom']
                    cropped_bgr = full_bgr[t:b, l:r]
                    if cropped_bgr.size == 0: continue
                    cropped_depth = np.zeros(cropped_bgr.shape[:2], dtype=np.uint8)
                else:
                    ok_depth += 1
            else:
                # Attack: keep RGB masked, zero depth
                cropped_bgr, _ = make_masked_rgb_and_depth(full_bgr, tddfa, box)
                if cropped_bgr is None:
                    l, t, r, b = entry['left'], entry['top'], entry['right'], entry['bottom']
                    cropped_bgr = full_bgr[t:b, l:r]
                    if cropped_bgr.size == 0: continue
                
                cropped_depth = np.zeros(cropped_bgr.shape[:2], dtype=np.uint8)
                ok_depth += 1

            # Resize crop to OUTPUT_SIZE x OUTPUT_SIZE
            rgb_out   = cv2.resize(cropped_bgr, (OUTPUT_SIZE, OUTPUT_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
            depth_out = cv2.resize(cropped_depth, (OUTPUT_SIZE, OUTPUT_SIZE),
                                   interpolation=cv2.INTER_NEAREST)

            base = f'frame_{fidx:04d}'
            cv2.imwrite(str(out_dir / f'{base}.jpg'),    rgb_out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out_dir / f'{base}_depth.jpg'), depth_out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

    return saved, ok_depth, None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Reprocess MSU-MFSD: 3DDFA oval-mask crop + depth.')
    parser.add_argument('--split', choices=['train', 'test', 'all'],
                        default='all')
    parser.add_argument('--frames-per-video', type=int, default=25,
                        help='Frames to sample per video (0=all)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    splits = ['train', 'test'] if args.split == 'all' else [args.split]

    print('=' * 55)
    print(' MSU-MFSD Reprocess  (3DDFA oval mask + depth)')
    print('=' * 55)
    print(f' Source  : {MSU_DIR}')
    print(f' Output  : {OUTPUT_DIR}')
    print(f' Splits  : {splits}')
    print(f' Frames/video: {args.frames_per_video or "all"}')
    print(f' Out size: {OUTPUT_SIZE}x{OUTPUT_SIZE} (square canvas)')
    print('=' * 55)

    print('\n[INFO] Building task list...')
    tasks = build_tasks(args.split, args.frames_per_video)
    n_real   = sum(1 for t in tasks if t['label'] == 'real')
    n_attack = sum(1 for t in tasks if t['label'] == 'attack')
    total_frames = sum(len(t['entries']) for t in tasks)
    print(f'       Videos : {len(tasks)}  ({n_real} real, {n_attack} attack)')
    print(f'       Frames : ~{total_frames:,}')

    if args.dry_run:
        print('[DRY RUN] No files written.\n')
        return

    if not tasks:
        print('[INFO] Nothing to process (all done or no videos found).\n')
        return

    print('\n[INFO] Loading TDDFA...')
    tddfa = init_tddfa()
    print('[OK] TDDFA ready\n')

    total_saved = total_ok_depth = total_fail = 0
    bar = tqdm(tasks, desc='Videos', unit='video', ascii=True)
    for task in bar:
        saved, ok_d, err = process_video(task, tddfa,
                                          zero_depth=np.zeros(
                                              (OUTPUT_SIZE, OUTPUT_SIZE),
                                              dtype=np.uint8))
        total_saved    += saved
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
    print(f'  Output dir       : {OUTPUT_DIR}')
    print(f'{"="*55}\n')


if __name__ == '__main__':
    main()
