import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO

def parse_seqinfo(seq_dir):
    """
    Reads seqinfo.ini and returns a dict with sequence information.
    """
    ini_path = os.path.join(seq_dir, "seqinfo.ini")
    info = {}
    with open(ini_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
    
    info["seqLength"] = int(info.get("seqLength", "0"))
    info["frameRate"] = int(info.get("frameRate", "30"))
    info["imDir"] = os.path.join(seq_dir, info.get("imDir", "img1"))
    info["imExt"] = info.get("imExt", ".jpg")
    return info

def draw_detections(img, dets, color=(0, 255, 0), thickness=2, draw_score=True):
    """
    Draws bounding boxes on the image for each detection.
    dets: list of [x, y, w, h, score]
    """
    if dets is None or len(dets) == 0:
        return img
    
    for x, y, w, h, score in dets:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        if draw_score:
            label = f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Draw background for text
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return img

def main(
    seq_dir="data/MOT17-02-FRCNN",
    model_name="yolov8n.pt",
    conf_thresh=0.25,
    iou_thresh=0.45,
    show=False,
    save=True,
    output_dir=None,
    write_video=False,
    video_name="yolo_detections.mp4",
    fourcc_str="mp4v",
):
    """
    Main function that:
    1. Loads YOLO model
    2. Runs inference on all frames
    3. Saves detections to det/yolo.txt in MOT format
    4. Visualizes detections and saves to results folder
    """
    # Parse sequence info
    info = parse_seqinfo(seq_dir)
    img_dir = info["imDir"]
    im_ext = info["imExt"]
    seq_len = info["seqLength"]
    fps = info["frameRate"]
    
    # Load YOLO model
    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    
    # Prepare detection output file
    det_dir = os.path.join(seq_dir, "det")
    os.makedirs(det_dir, exist_ok=True)
    det_path = os.path.join(det_dir, "yolo.txt")
    
    # Prepare visualization output directory
    if output_dir is None:
        output_dir = os.path.join(seq_dir, "results", "yolo_vis")
    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video writer if needed
    writer = None
    if write_video:
        # Read first frame to get dimensions
        first_img_path = os.path.join(img_dir, f"{1:06d}{im_ext}")
        first_img = cv2.imread(first_img_path)
        if first_img is not None:
            h, w = first_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            video_path = os.path.join(output_dir, video_name)
            writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            print(f"Video will be saved to: {video_path}")
        else:
            print("Warning: Could not initialize video writer - first frame not found")
    
    # Open detection file for writing
    det_file = open(det_path, "w")
    print(f"Saving detections to: {det_path}")
    
    # Process each frame
    for frame_idx in tqdm(range(1, seq_len + 1), desc="Running YOLO & Visualizing"):
        img_path = os.path.join(img_dir, f"{frame_idx:06d}{im_ext}")
        if not os.path.exists(img_path):
            continue
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Run YOLO inference
        results = model(img_path, conf=conf_thresh, iou=iou_thresh, verbose=False)
        result = results[0]
        
        # Extract detections
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # xyxy format
            scores = result.boxes.conf.cpu().numpy()
            
            # Convert xyxy to xywh and save to file
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                x = float(x1)
                y = float(y1)
                w = float(x2 - x1)
                h = float(y2 - y1)
                
                # MOT format: frame, id(-1 for detections), x, y, w, h, score
                det_file.write(f"{frame_idx},-1,{x:.1f},{y:.1f},{w:.1f},{h:.1f},{score:.6f}\n")
                
                # Store for visualization
                detections.append([x, y, w, h, score])
        
        # Draw detections on image
        vis = draw_detections(img.copy(), detections)
        
        # Show if requested
        if show:
            cv2.imshow("YOLO Detections", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save visualized frame
        if save:
            out_path = os.path.join(output_dir, f"{frame_idx:06d}{im_ext}")
            cv2.imwrite(out_path, vis)
        
        # Write to video if requested
        if writer is not None:
            writer.write(vis)
    
    # Cleanup
    det_file.close()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    
    print(f"\nDone! Detections saved to: {det_path}")
    if save:
        print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run YOLO detector on MOT17 sequence and visualize results")
    parser.add_argument("--seq-dir", type=str, default="data/MOT17-02-FRCNN", 
                        help="Path to sequence folder containing seqinfo.ini")
    parser.add_argument("--model", type=str, default="yolov8n.pt", 
                        help="YOLO model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.45, 
                        help="IoU threshold for NMS")
    parser.add_argument("--show", action="store_true", 
                        help="Display frames as they are processed")
    parser.add_argument("--no-save", action="store_true", 
                        help="Do not save visualized frames to disk")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Directory to save visualized frames (defaults to <seq-dir>/results/yolo_vis)")
    parser.add_argument("--write-video", action="store_true", 
                        help="Additionally write a video file to the output directory")
    parser.add_argument("--video-name", type=str, default="yolo_detections.mp4", 
                        help="Output video filename")
    parser.add_argument("--fourcc", type=str, default="mp4v", 
                        help="FourCC codec (e.g., mp4v, avc1, XVID)")
    args = parser.parse_args()
    
    main(
        seq_dir=args.seq_dir,
        model_name=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        show=args.show,
        save=not args.no_save,
        output_dir=args.output_dir,
        write_video=args.write_video,
        video_name=args.video_name,
        fourcc_str=args.fourcc,
    )