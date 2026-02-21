import cv2
import numpy as np
from kalman import KF
from utils import iou
from ultralytics import YOLO

def setup_kalman_filter():
    """
    Creates and returns the Kalman filter matrices.
    """
    # State transition matrix A (8x8)
    # new_position = old_position + velocity
    A = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],  # x_new = x_old + vx
        [0, 1, 0, 0, 0, 1, 0, 0],  # y_new = y_old + vy
        [0, 0, 1, 0, 0, 0, 1, 0],  # w_new = w_old + vw
        [0, 0, 0, 1, 0, 0, 0, 1],  # h_new = h_old + vh
        [0, 0, 0, 0, 1, 0, 0, 0],  # vx stays same
        [0, 0, 0, 0, 0, 1, 0, 0],  # vy stays same
        [0, 0, 0, 0, 0, 0, 1, 0],  # vw stays same
        [0, 0, 0, 0, 0, 0, 0, 1],  # vh stays same
    ], dtype=float)
    
    # Measurement matrix H (4x8)
    # Extracts [x, y, w, h] from the 8D state
    H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],  # x
        [0, 1, 0, 0, 0, 0, 0, 0],  # y
        [0, 0, 1, 0, 0, 0, 0, 0],  # w
        [0, 0, 0, 1, 0, 0, 0, 0],  # h
    ], dtype=float)
    
    # Process noise Q (how much we trust our motion model)
    Q = 1.0 * np.eye(8)
    
    # Measurement noise R (how much we trust our detections)
    R = 10.0 * np.eye(4)
    
    # Initial covariance P0 (uncertainty in initial state)
    P0 = 100.0 * np.eye(8)
    
    return A, H, Q, R, P0

def draw_box(img, box, color, label="", thickness=2):
    """
    Draws a bounding box on the image.
    
    Args:
        img: OpenCV image
        box: [x, y, w, h] in xywh format
        color: (B, G, R) tuple
        label: text to display
        thickness: line thickness
    """
    x, y, w, h = box
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        cv2.putText(img, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main(video_path, output_path="data/kalman_tracked.mp4", model_name="yolov8n.pt"):
    """
    Main function to test Kalman filter on video.
    
    Args:
        video_path: path to input video
        output_path: path to save output video
        model_name: YOLO model to use
    """
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
    
    # Initialize the Kalman filter after getting 2 detections
    kf = None
    prev_detection = None
    frame_count = 0
    
    print("\nProcessing video...")
    print("Green box = YOLO detection")
    print("Red box = Kalman filter prediction")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, conf=0.25, verbose=False)
        result = results[0]
        
        # Get the best detection (highest confidence)
        detection = None
        if result.boxes is not None and len(result.boxes) > 0:
            # Get box with highest confidence
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            best_idx = np.argmax(scores)
            
            # Convert from xyxy to xywh format
            x1, y1, x2, y2 = boxes[best_idx]
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            score = float(scores[best_idx])
            
            detection = [x, y, w, h]
            
            # Draw detection box (green)
            draw_box(frame, detection, (0, 255, 0), f"Detection {score:.2f}")
        
        # Initialize Kalman filter if we have 2 detections
        if kf is None and detection is not None and prev_detection is not None:
            print(f"\nInitializing Kalman filter at frame {frame_count}...")
            x0 = KF.init_from_two_detections(prev_detection, detection)
            kf = KF(x0, P0, Q, R, A, H)
            print(f"Initial state: x={x0[0,0]:.1f}, y={x0[1,0]:.1f}, vx={x0[4,0]:.1f}, vy={x0[5,0]:.1f}")
        
        # If Kalman filter is initialized, use it
        if kf is not None:
            # Step 1: Predict where object should be
            x_pred, P_pred = kf.predict()
            
            # Extract predicted box [x, y, w, h]
            pred_box = [
                x_pred[0, 0],  # x
                x_pred[1, 0],  # y
                x_pred[2, 0],  # w
                x_pred[3, 0]   # h
            ]
            
            # Draw prediction box (red)
            draw_box(frame, pred_box, (0, 0, 255), "Kalman Prediction")

            # NEW: Calculate and display IoU if we have a detection
            if detection is not None:
                # Calculate IoU between prediction and detection
                iou_value = iou(pred_box, detection)
                
                # Display IoU on the frame
                iou_text = f"IoU: {iou_value:.2f}"
                cv2.putText(frame, iou_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Color-code based on IoU quality
                if iou_value > 0.5:
                    iou_color = (0, 255, 0)  # Green = good match
                elif iou_value > 0.3:
                    iou_color = (0, 255, 255)  # Yellow = okay match
                else:
                    iou_color = (0, 0, 255)  # Red = poor match
                
                # Draw a small indicator box
                cv2.rectangle(frame, (10, 90), (10 + int(iou_value * 100), 110), iou_color, -1)            
            
            # Step 2: Update with detection if we have one
            if detection is not None:
                z = np.array([[detection[0]], [detection[1]], [detection[2]], [detection[3]]], dtype=float)
                x_updated, P_updated = kf.update(z)
                
                # Draw updated box (blue) - this is the final estimate
                updated_box = [
                    x_updated[0, 0],
                    x_updated[1, 0],
                    x_updated[2, 0],
                    x_updated[3, 0]
                ]
                draw_box(frame, updated_box, (255, 0, 0), "Kalman Updated", thickness=1)
        
        # Store current detection for next frame
        if detection is not None:
            prev_detection = detection
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Show progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    print(f"\nDone! Output saved to: {output_path}")
    print("\nLegend:")
    print("  Green box = YOLO detection (raw measurement)")
    print("  Red box = Kalman prediction (before update)")
    print("  Blue box = Kalman updated (after blending prediction + detection)")

if __name__ == "__main__":
    import sys
    
    # Default video path
    video_path = "data/so_sequence.MP4"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    main(video_path)