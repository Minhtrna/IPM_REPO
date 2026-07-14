import os
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def check_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    print(f"\n{'='*55}")
    print(f" Scanning dataset: {dataset_dir.name}")
    print(f"{'='*55}")
    
    processed_dir = dataset_dir / 'processed_3ddfa'
    if not processed_dir.exists():
        print(f"[ERROR] Processed directory not found: {processed_dir}")
        return
        
    video_dirs = []
    for root, dirs, files in os.walk(processed_dir):
        if any(f.endswith('.jpg') for f in files):
            video_dirs.append(Path(root))
            
    if not video_dirs:
        print(f"[WARNING] No video folders found in {processed_dir}")
        return
        
    print(f"[INFO] Found {len(video_dirs)} video folders. Checking for missing depth maps...")
    
    total_rgb = 0
    total_depth = 0
    missing_depth_files = []
    empty_dirs = []
    frame_counts = defaultdict(int)
    
    # Dictionary to hold stats by Split -> Label
    # stats[split][label] = {'videos': 0, 'rgb': 0, 'depth': 0}
    stats = defaultdict(lambda: defaultdict(lambda: {'videos': 0, 'rgb': 0, 'depth': 0}))
    
    for vdir in tqdm(video_dirs, desc="Scanning", ascii=True):
        files = set(os.listdir(vdir))
        
        rgb_files = [f for f in files if f.endswith('.jpg') and not f.endswith('_depth.jpg')]
        depth_files = [f for f in files if f.endswith('_depth.jpg')]
        
        if not rgb_files:
            empty_dirs.append(vdir)
            continue
            
        n_rgb = len(rgb_files)
        n_depth = len(depth_files)
        
        total_rgb += n_rgb
        total_depth += n_depth
        frame_counts[n_rgb] += 1
        
        # Try to infer split and label from path
        # Example path: D:/DATASET/MSU-MFSD/processed_3ddfa/train/real/video_name
        # parts[-1] = video_name, parts[-2] = label, parts[-3] = split
        parts = vdir.parts
        if len(parts) >= 3:
            split_name = parts[-3]
            label_name = parts[-2]
            stats[split_name][label_name]['videos'] += 1
            stats[split_name][label_name]['rgb'] += n_rgb
            stats[split_name][label_name]['depth'] += n_depth
            
        for rgb in rgb_files:
            expected_depth = rgb.replace('.jpg', '_depth.jpg')
            if expected_depth not in files:
                missing_depth_files.append(vdir / expected_depth)

    print("\n--- Summary ---")
    print(f"Total Video Folders (with images): {len(video_dirs) - len(empty_dirs)}")
    print(f"Total RGB Frames : {total_rgb}")
    print(f"Total Depth Maps : {total_depth}")
    
    print("\n--- Breakdown by Split & Label ---")
    for split, labels in stats.items():
        print(f"[{split.upper()}]")
        for label, counts in labels.items():
            print(f"  - {label.capitalize()}: {counts['videos']} videos | {counts['rgb']} RGB frames | {counts['depth']} Depth maps")
    
    if empty_dirs:
        print(f"\n[WARNING] Found {len(empty_dirs)} folders with NO RGB frames!")
        
    print("\n--- Frames per video distribution ---")
    for count, num_videos in sorted(frame_counts.items()):
        print(f"  - {num_videos} videos have {count} frames")
        
    if missing_depth_files:
        print(f"\n[WARNING] Found {len(missing_depth_files)} missing depth maps!")
        for mf in missing_depth_files[:20]:
            print(f"  Missing: {mf}")
        if len(missing_depth_files) > 20:
            print(f"  ... and {len(missing_depth_files) - 20} more.")
    else:
        print("\n[SUCCESS] No missing depth maps! Every RGB frame has a corresponding depth map.")

def main():
    parser = argparse.ArgumentParser(description="Check for missing depth maps in processed datasets")
    parser.add_argument('--dataset', type=str, choices=['oulu', 'msu', 'all'], default='all')
    parser.add_argument('--base-dir', type=str, default='d:/DATASET')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if args.dataset in ['oulu', 'all']:
        check_dataset(base_dir / 'OULU_NPU')
        
    if args.dataset in ['msu', 'all']:
        check_dataset(base_dir / 'MSU-MFSD')

if __name__ == '__main__':
    main()
