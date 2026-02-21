import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: [x1, y1, w1, h1] - first box in xywh format
        box2: [x2, y2, w2, h2] - second box in xywh format
    
    Returns:
        IoU value between 0.0 and 1.0
    """
    # Extract coordinates
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to corner coordinates
    # Box 1: top-left (x1, y1) to bottom-right (x1+w1, y1+h1)
    box1_left = x1
    box1_top = y1
    box1_right = x1 + w1
    box1_bottom = y1 + h1
    
    # Box 2
    box2_left = x2
    box2_top = y2
    box2_right = x2 + w2
    box2_bottom = y2 + h2
    
    # Find the intersection rectangle - where the boxes overlap
    inter_left = max(box1_left, box2_left)      # leftmost right edge
    inter_top = max(box1_top, box2_top)        # topmost bottom edge
    inter_right = min(box1_right, box2_right)  # rightmost left edge
    inter_bottom = min(box1_bottom, box2_bottom) # bottommost top edge
    
    # Check if there's actually an intersection
    # If right < left or bottom < top, boxes don't overlap
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0  # No overlap, IoU = 0
    
    # Calculate intersection area
    inter_width = inter_right - inter_left
    inter_height = inter_bottom - inter_top
    inter_area = inter_width * inter_height
    
    # Calculate area of each box
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate union area
    # Union = box1_area + box2_area - intersection (subtract intersection because it's counted twice)
    union_area = box1_area + box2_area - inter_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    # IoU = intersection / union
    iou_value = inter_area / union_area
    
    return iou_value

def iou_matrix(track_boxes, detection_boxes):
    """
    Calculate IoU between all track boxes and all detection boxes.
    
    This creates a matrix where each row is a track and each column is a detection.
    The value at [i, j] is the IoU between track i and detection j.
    
    Args:
        track_boxes: List of track boxes, each is [x, y, w, h] in xywh format
        detection_boxes: List of detection boxes, each is [x, y, w, h] in xywh format
    
    Returns:
        iou_mat: numpy array of shape (n_tracks, n_detections) with IoU values
    """
    n_tracks = len(track_boxes)
    n_detections = len(detection_boxes)
    
    # Create empty matrix
    iou_mat = np.zeros((n_tracks, n_detections))
    
    # Fill matrix: for each track, calculate IoU with each detection
    for i, track_box in enumerate(track_boxes):
        for j, det_box in enumerate(detection_boxes):
            iou_mat[i, j] = iou(track_box, det_box)
    
    return iou_mat

def associate_detections_to_tracks(iou_matrix, min_iou=0.3):
    """
    Match detections to tracks using the Hungarian algorithm.
    
    The Hungarian algorithm finds the best matching that maximizes total IoU.
    We convert IoU to cost (cost = 1 - IoU) because Hungarian minimizes cost.
    
    Args:
        iou_matrix: numpy array of shape (n_tracks, n_detections) with IoU values
        min_iou: minimum IoU threshold for a valid match (default 0.3)
    
    Returns:
        matches: List of (track_idx, detection_idx) tuples for matched pairs
        unmatched_tracks: List of track indices that didn't match any detection
        unmatched_detections: List of detection indices that didn't match any track
    """
    # Handle empty cases
    if iou_matrix.size == 0:
        n_tracks = iou_matrix.shape[0] if len(iou_matrix.shape) > 0 else 0
        n_dets = iou_matrix.shape[1] if len(iou_matrix.shape) > 1 else 0
        return [], list(range(n_tracks)), list(range(n_dets))
    
    n_tracks, n_detections = iou_matrix.shape
    
    # Convert IoU to cost matrix
    # Higher IoU = better match, but Hungarian minimizes cost
    # So: cost = 1.0 - iou (higher IoU → lower cost)
    cost_matrix = 1.0 - iou_matrix
    
    # Run Hungarian algorithm
    # This finds the assignment that minimizes total cost
    track_indices, det_indices = linear_sum_assignment(cost_matrix)
    
    # Filter matches by minimum IoU threshold
    matches = []
    for t_idx, d_idx in zip(track_indices, det_indices):
        if iou_matrix[t_idx, d_idx] >= min_iou:
            matches.append((t_idx, d_idx))
    
    # Find unmatched tracks (tracks that didn't get matched)
    matched_track_indices = {t_idx for t_idx, _ in matches}
    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_track_indices]
    
    # Find unmatched detections (detections that didn't get matched)
    matched_det_indices = {d_idx for _, d_idx in matches}
    unmatched_detections = [j for j in range(n_detections) if j not in matched_det_indices]
    
    return matches, unmatched_tracks, unmatched_detections