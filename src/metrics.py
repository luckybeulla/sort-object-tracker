import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import motmetrics as mm
from utils import iou

def load_mot_file(filepath):
    """
    Load tracking results from MOT format file.
    Format: frame, id, x, y, w, h, conf, -1, -1, -1
    Returns: dictionary mapping frame -> list of (id, x, y, w, h)
    """
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                
                if frame not in results:
                    results[frame] = []
                results[frame].append((track_id, x, y, w, h))
    return results

def calculate_mean_iou(tracked_results, ground_truth_results):
    """
    Calculate mean IoU between tracked results and ground truth.
    
    Args:
        tracked_results: dict mapping frame -> list of (id, x, y, w, h)
        ground_truth_results: dict mapping frame -> list of (id, x, y, w, h)
    
    Returns:
        mean_iou: average IoU across all matched boxes
    """
    all_ious = []
    
    # Get all frames that exist in both
    common_frames = set(tracked_results.keys()) & set(ground_truth_results.keys())
    
    for frame in common_frames:
        tracked = tracked_results[frame]
        gt = ground_truth_results[frame]
        
        # Create boxes from tracked results
        tracked_boxes = {}
        for tid, x, y, w, h in tracked:
            tracked_boxes[tid] = [x, y, w, h]
        
        # Match with ground truth by ID and calculate IoU
        for gt_id, gt_x, gt_y, gt_w, gt_h in gt:
            if gt_id in tracked_boxes:
                track_box = tracked_boxes[gt_id]
                gt_box = [gt_x, gt_y, gt_w, gt_h]
                iou_value = iou(track_box, gt_box)
                all_ious.append(iou_value)
    
    if len(all_ious) == 0:
        return 0.0
    
    return np.mean(all_ious)

def evaluate_with_motmetrics(tracked_file, gt_file):
    """
    Evaluate tracking results using motmetrics library.
    
    Args:
        tracked_file: path to tracking results (MOT format)
        gt_file: path to ground truth (MOT format)
    
    Returns:
        summary: DataFrame with metrics (MOTA, MOTP, IDF1, etc.)
    """
    # Load tracking results
    tracked_df = mm.io.loadtxt(tracked_file, fmt='mot15-2D')
    gt_df = mm.io.loadtxt(gt_file, fmt='mot15-2D')
    
    # Create accumulator
    acc = mm.utils.compare_to_groundtruth(gt_df, tracked_df, 'iou', distth=0.5)
    
    # Calculate metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='tracker')
    
    return summary

def print_summary(summary):
    """Print formatted summary of metrics."""
    print("\n" + "="*60)
    print("TRACKING EVALUATION METRICS")
    print("="*60)
    
    # Print key metrics
    metrics_to_show = [
        'idf1',      # ID F1 Score
        'idp',       # ID Precision
        'idr',       # ID Recall
        'mota',      # Multiple Object Tracking Accuracy
        'motp',      # Multiple Object Tracking Precision
        'num_frames',
        'num_objects',
        'num_predictions',
        'num_switches',      # ID switches
        'num_fragments',    # Fragmentations
        'num_misses',       # Missed detections
        'num_false_positives',
    ]
    
    for metric in metrics_to_show:
        if metric in summary.columns:
            value = summary[metric].iloc[0]
            if metric in ['mota', 'motp', 'idf1', 'idp', 'idr']:
                print(f"{metric.upper():25s}: {value:.4f}")
            else:
                print(f"{metric.upper():25s}: {int(value)}")
    
    print("="*60)

def main(tracked_file, gt_file=None):
    """
    Main evaluation function.
    
    Args:
        tracked_file: path to tracking results file (MOT format)
        gt_file: path to ground truth file (MOT format, optional)
    """
    print(f"Loading tracking results from: {tracked_file}")
    tracked_results = load_mot_file(tracked_file)
    
    print(f"Found {len(tracked_results)} frames with tracking data")
    
    # Calculate basic statistics
    total_tracks = 0
    for frame_tracks in tracked_results.values():
        total_tracks += len(frame_tracks)
    
    print(f"Total track detections: {total_tracks}")
    print(f"Average tracks per frame: {total_tracks / len(tracked_results):.2f}")
    
    # If ground truth provided, calculate metrics
    if gt_file:
        print(f"\nLoading ground truth from: {gt_file}")
        gt_results = load_mot_file(gt_file)
        
        # Calculate mean IoU
        mean_iou = calculate_mean_iou(tracked_results, gt_results)
        print(f"\nMean IoU: {mean_iou:.4f}")
        
        # Calculate MOT metrics
        try:
            summary = evaluate_with_motmetrics(tracked_file, gt_file)
            print_summary(summary)
        except Exception as e:
            print(f"\nError calculating MOT metrics: {e}")
            print("Make sure motmetrics is installed: pip install motmetrics")
    else:
        print("\nNo ground truth provided. Provide gt_file to calculate metrics.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <tracked_results.txt> [ground_truth.txt]")
        print("\nExample:")
        print("  python evaluate.py data/tracking_results.txt data/gt.txt")
        sys.exit(1)
    
    tracked_file = sys.argv[1]
    gt_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(tracked_file, gt_file)