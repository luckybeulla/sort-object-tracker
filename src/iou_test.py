import cv2
import numpy as np
from utils import iou
from ultralytics import YOLO

def draw_box(img, box, color, label="", thickness=2):
    """
    Draws a bounding box on the image.
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

def draw_intersection(img, box1, box2, color=(255, 255, 0), alpha=0.3):
    """
    Draws the intersection area between two boxes.
    This helps visualize what IoU is measuring.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to corner coordinates
    box1_left = x1
    box1_top = y1
    box1_right = x1 + w1
    box1_bottom = y1 + h1
    
    box2_left = x2
    box2_top = y2
    box2_right = x2 + w2
    box2_bottom = y2 + h2
    
    # Find intersection
    inter_left = max(box1_left, box2_left)
    inter_top = max(box1_top, box2_top)
    inter_right = min(box1_right, box2_right)
    inter_bottom = min(box1_bottom, box2_bottom)
    
    # Draw intersection if it exists
    if inter_right > inter_left and inter_bottom > inter_top:
        # Create overlay
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (int(inter_left), int(inter_top)), 
                     (int(inter_right), int(inter_bottom)), 
                     color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def main(video_path, output_path="data/iou_tracked.mp4", model_name="yolov8n.pt"):
    """
    Test IoU calculation and visualization on video.
    This script shows how IoU works by comparing detections across frames.
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
    
    # Store previous detection to compare with current
    prev_detection = None
    frame_count = 0
    iou_values = []  # Track IoU over time
    
    print("\nProcessing video...")
    print("Green box = Current detection")
    print("Blue box = Previous detection")
    print("Yellow overlay = Intersection area (what IoU measures)")
    
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
            
            # Draw current detection (green)
            draw_box(frame, detection, (0, 255, 0), f"Current {score:.2f}")
            
            # Calculate and visualize IoU if we have previous detection
            if prev_detection is not None:
                # Calculate IoU between current and previous detection
                iou_value = iou(detection, prev_detection)
                iou_values.append(iou_value)
                
                # Draw previous detection (blue)
                draw_box(frame, prev_detection, (255, 0, 0), "Previous")
                
                # Draw intersection area (yellow overlay)
                draw_intersection(frame, detection, prev_detection)
                
                # Display IoU prominently
                iou_text = f"IoU: {iou_value:.3f}"
                text_size = cv2.getTextSize(iou_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                
                # Background for text
                cv2.rectangle(frame, (10, 60), (20 + text_size[0], 100), (0, 0, 0), -1)
                
                # Color-code IoU text
                if iou_value > 0.7:
                    iou_color = (0, 255, 0)  # Green = excellent match
                elif iou_value > 0.5:
                    iou_color = (0, 255, 255)  # Yellow = good match
                elif iou_value > 0.3:
                    iou_color = (0, 165, 255)  # Orange = okay match
                else:
                    iou_color = (0, 0, 255)  # Red = poor match
                
                cv2.putText(frame, iou_text, (15, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, iou_color, 3)
                
                # Draw IoU bar
                bar_width = 200
                bar_height = 20
                bar_x = 10
                bar_y = 110
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + bar_height), 
                             (50, 50, 50), -1)
                
                # IoU bar (colored)
                bar_fill = int(iou_value * bar_width)
                cv2.rectangle(frame, (bar_x, bar_y), 
                             (bar_x + bar_fill, bar_y + bar_height), 
                             iou_color, -1)
                
                # Bar border
                cv2.rectangle(frame, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + bar_height), 
                             (255, 255, 255), 2)
                
                # Statistics
                if len(iou_values) > 0:
                    avg_iou = np.mean(iou_values)
                    min_iou = np.min(iou_values)
                    max_iou = np.max(iou_values)
                    
                    stats_text = [
                        f"Avg: {avg_iou:.3f}",
                        f"Min: {min_iou:.3f}",
                        f"Max: {max_iou:.3f}"
                    ]
                    
                    y_offset = 150
                    for i, stat in enumerate(stats_text):
                        cv2.putText(frame, stat, (10, y_offset + i * 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
            if len(iou_values) > 0:
                print(f"  Current IoU: {iou_values[-1]:.3f}, Average: {np.mean(iou_values):.3f}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Print final statistics
    if len(iou_values) > 0:
        print(f"\n=== IoU Statistics ===")
        print(f"Average IoU: {np.mean(iou_values):.3f}")
        print(f"Minimum IoU: {np.min(iou_values):.3f}")
        print(f"Maximum IoU: {np.max(iou_values):.3f}")
        print(f"Std Dev: {np.std(iou_values):.3f}")
    
    print(f"\nDone! Output saved to: {output_path}")
    print("\nLegend:")
    print("  Green box = Current detection")
    print("  Blue box = Previous detection")
    print("  Yellow overlay = Intersection area (overlap)")
    print("  IoU = Intersection / Union of both boxes")

if __name__ == "__main__":
    import sys
    
    # Default video path
    video_path = "data/so_sequence.MP4"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    main(video_path)