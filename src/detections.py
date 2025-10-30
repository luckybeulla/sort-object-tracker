import os
import cv2
from tqdm import tqdm

def load_detections(det_path, score_thresh=0.0):
    """
    Reads MOT det.txt and returns a dict:
    { frame_idx: [ [x, y, w, h, score], ... ] }
    Only keeps detections with score >= score_thresh.
    """
    detections = {}
    with open(det_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            x, y, w, h, score = map(float, parts[2:7])
            if score < score_thresh:
                continue
            if frame not in detections:
                detections[frame] = []
            detections[frame].append([x, y, w, h, score])
    return detections

def parse_seqinfo(seq_dir):
    """
    Reads seqinfo.ini and returns a small dict with fields we need.
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
    info["imWidth"] = int(info.get("imWidth", "0"))
    info["imHeight"] = int(info.get("imHeight", "0"))
    info["imDir"] = os.path.join(seq_dir, info.get("imDir", "img1"))
    info["imExt"] = info.get("imExt", ".jpg")
    return info

def draw_detections(img, dets, color=(0, 255, 0), thickness=2, draw_score=True):
    """
    Draws rectangles (and scores) on the image for each detection [x, y, w, h, score].
    """
    if dets is None:
        return img
    for x, y, w, h, score in dets:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if draw_score:
            label = f"{score:.2f}"
            # background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def main(
    seq_dir="data/MOT17-02-FRCNN",
    score_thresh=0.0,
    show=False,
    save=True,
    output_dir=None,
    write_video=False,
    video_name="detections.mp4",
    fourcc_str="mp4v",
):
    # Read sequence info
    info = parse_seqinfo(seq_dir)
    img_dir = info["imDir"]
    im_ext = info["imExt"]
    seq_len = info["seqLength"]
    fps = info["frameRate"]

    # Read detections
    det_path = os.path.join(seq_dir, "det", "det.txt")
    detections = load_detections(det_path, score_thresh=score_thresh)

    # Prepare output
    if output_dir is None:
        output_dir = os.path.join(seq_dir, "vis_dets")
    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Video writer
    writer = None
    if write_video:
        # Try to get frame size by reading the first available frame
        first_frame_idx = None
        if detections:
            first_frame_idx = sorted(detections.keys())[0]
        else:
            first_frame_idx = 1
        first_img_path = os.path.join(img_dir, f"{first_frame_idx:06d}{im_ext}")
        first_img = cv2.imread(first_img_path)
        if first_img is None:
            # fallback to scanning for any image in img_dir
            candidates = sorted([p for p in os.listdir(img_dir) if p.endswith(im_ext)])
            if candidates:
                first_img = cv2.imread(os.path.join(img_dir, candidates[0]))
        if first_img is not None:
            h, w = first_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(os.path.join(output_dir, video_name), fourcc, fps, (w, h))
        else:
            print("Warning: Could not initialize video writer because no image could be read.")

    # Iterate frames. Loop through 1..seq_len for completeness.
    # If a frame has no detections, still show/save the raw frame.
    for frame_idx in tqdm(range(1, seq_len + 1), desc="Rendering"):
        img_path = os.path.join(img_dir, f"{frame_idx:06d}{im_ext}")
        img = cv2.imread(img_path)
        if img is None:
            continue

        dets = detections.get(frame_idx, [])
        vis = draw_detections(img, dets)

        if show:
            cv2.imshow("detections", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if save:
            out_path = os.path.join(output_dir, f"{frame_idx:06d}{im_ext}")
            cv2.imwrite(out_path, vis)

        if writer is not None:
            writer.write(vis)

    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize MOT detections on frames.")
    parser.add_argument("--seq-dir", type=str, default="data/MOT17-02-FRCNN", help="Path to sequence folder containing seqinfo.ini")
    parser.add_argument("--score-thresh", type=float, default=0.0, help="Discard detections below this confidence score")
    parser.add_argument("--show", action="store_true", help="Display frames as they are processed")
    parser.add_argument("--no-save", action="store_true", help="Do not save visualized frames to disk")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save visualized frames (defaults to <seq-dir>/results/vis_dets)")
    parser.add_argument("--write-video", action="store_true", help="Additionally write a video file to the output directory")
    parser.add_argument("--video-name", type=str, default="detections.mp4", help="Output video filename")
    parser.add_argument("--fourcc", type=str, default="mp4v", help="FourCC (e.g., mp4v, avc1, XVID)")
    args = parser.parse_args()

    main(
        seq_dir=args.seq_dir,
        score_thresh=args.score_thresh,
        show=args.show,
        save=not args.no_save,
        output_dir=args.output_dir,
        write_video=args.write_video,
        video_name=args.video_name,
        fourcc_str=args.fourcc,
    )