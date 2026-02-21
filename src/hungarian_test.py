# src/hungarian_test.py
import cv2
import numpy as np
from kalman import KF
from utils import iou_matrix, associate_detections_to_tracks
from ultralytics import YOLO

def setup_kalman_filter():
    """Creates and returns the Kalman filter matrices."""
    A = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=float)
    
    H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=float)
    
    Q = 1.0 * np.eye(8)
    R = 10.0 * np.eye(4)
    P0 = 100.0 * np.eye(8)
    
    return A, H, Q, R, P0

class Track:
    """Represents a single tracked object with a Kalman filter."""
    def __init__(self, detection, track_id, A, H, Q, R, P0):
        self.track_id = track_id
        # Initialize with zero velocity (we'll update after getting more detections)
        x, y, w, h = detection
        x0 = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], dtype=float)
        self.kf = KF(x0, P0, Q, R, A, H)
        self.hits = 1  # Number of times matched
        self.time_since_update = 0  # Frames since last update
        self.age = 1  # Total age of track
    
    def predict(self):
        """Predict next state."""
        return self.kf.predict()
    
    def update(self, detection):
        """Update track with new detection."""
        z = np.array([[detection[0]], [detection[1]], [detection[2]], [detection[3]]], dtype=float)
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
    
    def get_state(self):
        """Get current box [x, y, w, h] from state."""
        x = self.kf.x[0, 0]
        y = self.kf.x[1, 0]
        w = self.kf.x[2, 0]
        h = self.kf.x[3, 0]
        return [x, y, w, h]
    
    def is_confirmed(self, min_hits=3):
        """Check if track is confirmed (seen enough times)."""
        return self.hits >= min_hits

def draw_box(img, box, color, label="", thickness=2):
    """Draws a bounding box on the image."""
    x, y, w, h = box
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        cv2.putText(img, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main(video_path, output_path="data/hungarian_tracked7.mp4", model_name="yolov8n.pt", mot_output_path=None):
    """
    Test Hungarian algorithm for matching detections to tracks.
    """
   # Open MOT output file if specified
    mot_output_file = None
    if mot_output_path:
        mot_output_file = open(mot_output_path, 'w')
        print(f"Saving MOT format results to: {mot_output_path}")    

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Load YOLO model
    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    
    # Setup Kalman filter matrices
    A, H, Q, R, P0 = setup_kalman_filter()
    
    # Track management
    tracks = []  # List of Track objects
    next_id = 1
    frame_count = 0
    
    # Colors for different tracks (BGR format) - more distinct colors
    track_colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
    ]
    
    # Tracking parameters
    min_hits = 3  # Minimum matches before track is "confirmed"
    max_age = 5   # Maximum frames a track can be unmatched
    min_iou = 0.4  # Increased from 0.3 to reduce ID swaps
    
    print("\nProcessing video...")
    print("Each track will have a different color")
    print("Hungarian algorithm matches detections to tracks")
    print(f"Only tracking 'person' class objects")
    print(f"Min IoU threshold: {min_iou} (higher = more stable IDs)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, conf=0.25, verbose=False)
        result = results[0]
        
        # Extract detections - ONLY PERSON CLASS (class 0 in COCO)
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()  # Get class IDs
            
            for box, score, cls in zip(boxes, scores, classes):
                # Only track "person" class (class 0 in COCO dataset)
                if int(cls) == 0 and score > 0.25:
                    x1, y1, x2, y2 = box
                    x = float(x1)
                    y = float(y1)
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    detections.append([x, y, w, h])
        
        # Step 1: Predict all tracks
        track_boxes = []
        confirmed_tracks = []
        unconfirmed_tracks = []
        
        for i, track in enumerate(tracks):
            track.predict()
            track_boxes.append(track.get_state())
            if track.is_confirmed(min_hits):
                confirmed_tracks.append(i)
            else:
                unconfirmed_tracks.append(i)
        
        # Step 2: Match detections to tracks using Hungarian algorithm
        # First match confirmed tracks, then unconfirmed
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        if len(tracks) > 0 and len(detections) > 0:
            # Calculate IoU matrix
            iou_mat = iou_matrix(track_boxes, detections)
            
            # Match confirmed tracks first (more stable)
            if len(confirmed_tracks) > 0:
                # Create sub-matrix for confirmed tracks only
                confirmed_iou = iou_mat[confirmed_tracks, :]
                confirmed_matches, _, unmatched_dets_temp = associate_detections_to_tracks(
                    confirmed_iou, min_iou=min_iou
                )
                
                # Map back to original indices
                for conf_track_idx, det_idx in confirmed_matches:
                    original_track_idx = confirmed_tracks[conf_track_idx]
                    matches.append((original_track_idx, det_idx))
                    unmatched_tracks.remove(original_track_idx)
                    if det_idx in unmatched_dets:
                        unmatched_dets.remove(det_idx)
            
            # Then match unconfirmed tracks with remaining detections
            if len(unconfirmed_tracks) > 0 and len(unmatched_dets) > 0:
                # Create sub-matrix for unconfirmed tracks and unmatched detections
                unconfirmed_iou = iou_mat[np.ix_(unconfirmed_tracks, unmatched_dets)]
                unconfirmed_matches, _, _ = associate_detections_to_tracks(
                    unconfirmed_iou, min_iou=min_iou
                )
                
                # Map back to original indices
                for unconf_track_idx, unmatched_det_idx in unconfirmed_matches:
                    original_track_idx = unconfirmed_tracks[unconf_track_idx]
                    original_det_idx = unmatched_dets[unmatched_det_idx]
                    matches.append((original_track_idx, original_det_idx))
                    unmatched_tracks.remove(original_track_idx)
                    unmatched_dets.remove(original_det_idx)
            
            # Step 3: Update matched tracks
            for track_idx, det_idx in matches:
                tracks[track_idx].update(detections[det_idx])
                iou_value = iou_mat[track_idx, det_idx]
                
                # Draw matched track (colored box)
                track_color = track_colors[tracks[track_idx].track_id % len(track_colors)]
                track_box = tracks[track_idx].get_state()
                
                # Show confirmation status
                status = "✓" if tracks[track_idx].is_confirmed(min_hits) else "?"
                draw_box(frame, track_box, track_color, 
                        f"ID:{tracks[track_idx].track_id} {status} (IoU:{iou_value:.2f})", thickness=2)
            
            # Step 4: Draw unmatched detections (these might be new objects)
            for det_idx in unmatched_dets:
                draw_box(frame, detections[det_idx], (128, 128, 128), 
                        "New?", thickness=1)
            
            # Step 5: Handle unmatched tracks (increment time since update)
            for track_idx in unmatched_tracks:
                tracks[track_idx].time_since_update += 1
                # Only draw confirmed tracks when unmatched (to reduce clutter)
                if tracks[track_idx].is_confirmed(min_hits):
                    track_color = track_colors[tracks[track_idx].track_id % len(track_colors)]
                    track_box = tracks[track_idx].get_state()
                    draw_box(frame, track_box, track_color, 
                            f"ID:{tracks[track_idx].track_id} (predicted)", thickness=1)
        
        # Step 6: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            if len(detections) > 0:
                new_track = Track(detections[det_idx], next_id, A, H, Q, R, P0)
                tracks.append(new_track)
                next_id += 1
                print(f"Frame {frame_count}: Created new track {new_track.track_id}")
        
        # Step 7: Remove old tracks (not matched for too long)
        tracks = [t for t in tracks if t.time_since_update < max_age]
        
        # Save MOT format results (for evaluation)
        # Format: frame, id, x, y, w, h, conf, -1, -1, -1
        if mot_output_file:
            for track in tracks:
                if track.is_confirmed(min_hits):  # Only save confirmed tracks
                    box = track.get_state()
                    x, y, w, h = box
                    mot_output_file.write(f"{frame_count},{track.track_id},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1.0,-1,-1,-1\n")
        
        # Display information
        confirmed_count = sum(1 for t in tracks if t.is_confirmed(min_hits))
        info_text = [
            f"Frame: {frame_count}",
            f"Tracks: {len(tracks)} (Confirmed: {confirmed_count})",
            f"Detections: {len(detections)}",
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Show progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames - {len(tracks)} active tracks")
    
    # Cleanup
    cap.release()
    out.release()
    if mot_output_file:
        mot_output_file.close()
    
    print(f"\nDone! Output saved to: {output_path}")
    print("\nLegend:")
    print("  Colored boxes = Matched tracks (each color = different track ID)")
    print("  ✓ = Confirmed track (stable, won't swap easily)")
    print("  ? = Unconfirmed track (new, might be removed)")
    print("  Gray boxes = Unmatched detections (new objects)")

if __name__ == "__main__":
    import sys
    
    # Default video path
    video_path = "data/two_objects.mp4"
    mot_output_path = None
    
    # Allow command line arguments: video_path [mot_output.txt]
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        mot_output_path = sys.argv[2]
    
    main(video_path, mot_output_path=mot_output_path)