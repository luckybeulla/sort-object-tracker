"""
SORT: Simple Online and Realtime Tracking.

Uses Kalman filter for prediction, IoU for similarity, and Hungarian algorithm
for detection-to-track association. Designed for use with MOT-format detections
and outputs.
"""

import os
import numpy as np
from kalman import KF
from utils import iou_matrix, associate_detections_to_tracks


def get_kalman_matrices():
    """Return constant-velocity Kalman filter matrices (A, H, Q, R, P0)."""
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
    """Single object track with Kalman filter."""

    def __init__(self, det_xywh, track_id, A, H, Q, R, P0):
        x, y, w, h = det_xywh
        x0 = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], dtype=float)
        self.kf = KF(x0, P0, Q, R, A, H)
        self.track_id = track_id
        self.hits = 1
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, det_xywh):
        x, y, w, h = det_xywh
        z = np.array([[x], [y], [w], [h]], dtype=float)
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0

    def get_state(self):
        return [
            float(self.kf.x[0, 0]),
            float(self.kf.x[1, 0]),
            float(self.kf.x[2, 0]),
            float(self.kf.x[3, 0]),
        ]

    def is_confirmed(self, min_hits):
        return self.hits >= min_hits


class SORT:
    """
    SORT tracker: predict, associate via IoU + Hungarian, update/create/delete tracks.
    """

    def __init__(self, min_hits=3, max_age=5, min_iou=0.3):
        self.min_hits = min_hits
        self.max_age = max_age
        self.min_iou = min_iou
        self.A, self.H, self.Q, self.R, self.P0 = get_kalman_matrices()
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        detections: list of [x, y, w, h].
        Returns: list of (track_id, x, y, w, h) for confirmed tracks.
        """
        for t in self.tracks:
            t.predict()
        track_boxes = [t.get_state() for t in self.tracks]
        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed(self.min_hits)]
        unconfirmed = [i for i in range(len(self.tracks)) if i not in confirmed]

        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(detections)))

        if self.tracks and detections:
            iou_mat = iou_matrix(track_boxes, detections)
            if confirmed:
                c_iou = iou_mat[confirmed, :]
                c_m, _, _ = associate_detections_to_tracks(c_iou, self.min_iou)
                for ci, dj in c_m:
                    ti = confirmed[ci]
                    matches.append((ti, dj))
                    unmatched_tracks.remove(ti)
                    if dj in unmatched_dets:
                        unmatched_dets.remove(dj)
            if unconfirmed and unmatched_dets:
                u_iou = iou_mat[np.ix_(unconfirmed, unmatched_dets)]
                u_m, _, _ = associate_detections_to_tracks(u_iou, self.min_iou)
                for ui, uj in u_m:
                    ti = unconfirmed[ui]
                    dj = unmatched_dets[uj]
                    matches.append((ti, dj))
                    unmatched_tracks.remove(ti)
                    unmatched_dets.remove(dj)
            for ti, dj in matches:
                self.tracks[ti].update(detections[dj])

        for ti in unmatched_tracks:
            self.tracks[ti].time_since_update += 1
        for dj in unmatched_dets:
            self.tracks.append(Track(detections[dj], self.next_id,
                                    self.A, self.H, self.Q, self.R, self.P0))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        out = []
        for t in self.tracks:
            if t.is_confirmed(self.min_hits):
                x, y, w, h = t.get_state()
                out.append((t.track_id, x, y, w, h))
        return out


def load_detections_mot(det_path):
    """Load MOT-format det file. Returns dict frame_id -> list of [x,y,w,h]."""
    by_frame = {}
    with open(det_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if frame not in by_frame:
                by_frame[frame] = []
            by_frame[frame].append([x, y, w, h])
    return by_frame


def run_on_detection_file(seq_dir, det_file, output_path, seqinfo_from_det=False):
    """Run SORT on MOT sequence using a detection file. Writes MOT-format results."""
    det_path = os.path.join(seq_dir, det_file) if not os.path.isabs(det_file) else det_file
    dets = load_detections_mot(det_path)
    if not dets:
        print("No detections loaded from", det_path)
        return
    if seqinfo_from_det:
        frame_ids = sorted(dets.keys())
    else:
        from detections import parse_seqinfo
        info = parse_seqinfo(seq_dir)
        frame_ids = list(range(1, info["seqLength"] + 1))
    tracker = SORT(min_hits=3, max_age=5, min_iou=0.3)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as out:
        for frame_id in frame_ids:
            dets_frame = dets.get(frame_id, [])
            tracks = tracker.update(dets_frame)
            for tid, x, y, w, h in tracks:
                out.write(f"{frame_id},{tid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1.0,-1,-1,-1\n")
    print("Tracking results saved to:", output_path)


def run_on_video(video_path, output_video_path=None, mot_output_path=None, model_name="yolov8n.pt"):
    """Run SORT on video with YOLO. Optional output video and MOT file."""
    import cv2
    from ultralytics import YOLO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: could not open video", video_path)
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if output_video_path else None
    mot_file = open(mot_output_path, "w") if mot_output_path else None
    model = YOLO(model_name)
    tracker = SORT(min_hits=3, max_age=5, min_iou=0.3)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        results = model(frame, conf=0.25, verbose=False)
        result = results[0]
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes, scores, classes = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()
            for box, score, cls in zip(boxes, scores, classes):
                if int(cls) == 0 and score > 0.25:
                    x1, y1, x2, y2 = box
                    detections.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        tracks = tracker.update(detections)
        if mot_file:
            for tid, x, y, w, h in tracks:
                mot_file.write(f"{frame_id},{tid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1.0,-1,-1,-1\n")
        if writer:
            for tid, x, y, w, h in tracks:
                c = colors[tid % len(colors)]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                cv2.putText(frame, str(tid), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            writer.write(frame)
    cap.release()
    if writer:
        writer.release()
    if mot_file:
        mot_file.close()
    print("Done. Video:", output_video_path or "none", "| MOT:", mot_output_path or "none")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SORT: run on video (YOLO) or MOT sequence (det file)")
    p.add_argument("--video", type=str, help="Input video")
    p.add_argument("--seq-dir", type=str, help="MOT sequence directory")
    p.add_argument("--det", type=str, default="det/yolo.txt", help="Detection file (with --seq-dir)")
    p.add_argument("--out-video", type=str, help="Output video path")
    p.add_argument("--out-mot", type=str, help="Output MOT tracking file")
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--no-seqinfo", action="store_true", help="Infer frame range from det file")
    args = p.parse_args()
    if args.video:
        run_on_video(args.video, args.out_video, args.out_mot, args.model)
    elif args.seq_dir:
        out = args.out_mot or os.path.join(args.seq_dir, "results", "sort.txt")
        run_on_detection_file(args.seq_dir, args.det, out, seqinfo_from_det=args.no_seqinfo)
    else:
        print("Use --video <path> or --seq-dir <path>. See --help.")