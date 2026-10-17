"""
Zero-Shot Detection + Tracking on a KITTI raw-data drive.

Pipeline: 
COCO-pretrained Faster R-CNN (torchvision) -> 
ByteTrack (supervision) -> 
annotated video with track IDs (+ optional CSV of tracks), with a live latency plot stacked below the video.

Example:
    python3 main_object_detection_track.py \
        --images 2011_09_26/2011_09_26_drive_0001_sync/image_02/data \
        --out drive0001_tracked.mp4
"""
import argparse
import csv
import glob
import os
import time

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

# torchvision's COCO label ids -> our simplified classes
COCO_TO_CLASS = {
    1: (0, "Pedestrian"),   # person
    2: (1, "Bicycle"),      # bicycle
    3: (2, "Vehicle"),      # car
    4: (1, "Bicycle"),      # motorcycle (grouped with bicycle)
    6: (2, "Vehicle"),      # bus
    8: (2, "Vehicle"),      # truck
}
CLASS_NAMES = {0: "Pedestrian", 1: "Bicycle", 2: "Vehicle"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True,
                   help="Folder of frames, e.g. .../2011_09_26_drive_0001_sync/image_02/data")
    p.add_argument("--out", default="tracked.mp4", help="Output video path")
    p.add_argument("--csv", default=None, help="Optional CSV of tracks "
                   "(frame,track_id,class,x1,y1,x2,y2,score)")
    p.add_argument("--score", type=float, default=0.5, help="Detection score threshold")
    p.add_argument("--fps", type=float, default=10.0, help="KITTI is recorded at 10 Hz")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--plot-height", type=int, default=300,
                   help="Height in pixels of the latency plot below the video")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_tracker(fps):
    """supervision renamed ByteTrack's arguments between versions; try both."""
    
    return ByteTrackTracker(frame_rate=float(fps), lost_track_buffer=30)


def color_for_id(track_id):
    rng = np.random.default_rng(int(track_id) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))  # BGR


@torch.inference_mode()
def detect(model, frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out = model([to_tensor(rgb).to(device)])[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()

    keep = [i for i, l in enumerate(labels) if int(l) in COCO_TO_CLASS]
    if not keep:
        return sv.Detections.empty()
    class_ids = np.array([COCO_TO_CLASS[int(labels[i])][0] for i in keep], dtype=int)
    return sv.Detections(
        xyxy=boxes[keep].astype(np.float32),
        confidence=scores[keep].astype(np.float32),
        class_id=class_ids,
    )


def draw(frame, dets):
    for i in range(len(dets)):
        x1, y1, x2, y2 = dets.xyxy[i].astype(int)
        tid = int(dets.tracker_id[i])
        name = CLASS_NAMES.get(int(dets.class_id[i]), "?")
        color = color_for_id(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid} {name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


class LatencyPlot:
    """Live latency plot drawn with Matplotlib's Agg backend (no GUI window needed).

    Starts empty and grows by one point per frame; both axes rescale every cycle.
    """

    def __init__(self, width_px, height_px, dpi=100):
        self.width_px, self.height_px = width_px, height_px
        self.fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.frames, self.det_ms, self.trk_ms, self.n_tracks = [], [], [], []
        (self.det_line,) = self.ax.plot([], [], color="tab:blue", label="Detection ms")
        (self.trk_line,) = self.ax.plot(
            [], [], color="tab:orange", label="Tracking ms")
        self.ax.set_title("Detection & Tracking Latency, and Number of Objects Tracked", fontweight="bold")
        self.ax.set_xlabel("Frame ID", fontweight="bold")
        self.ax.set_ylabel("Latency (ms)", fontweight="bold")
        self.ax.grid(True)

        # Right-hand axis: number of live tracks per frame
        self.ax2 = self.ax.twinx()
        (self.cnt_line,) = self.ax2.plot(
            [], [], color="tab:green", linestyle="--", label="Num Objects")
        self.ax2.set_ylabel("Number of Objects-Tracked", fontweight="bold")
        self.ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

        # One combined legend (drawn on ax2 so it sits above all lines)
        self.ax2.legend(handles=[self.det_line, self.trk_line, self.cnt_line], loc="upper right")
        self.fig.tight_layout()

    def update(self, frame_id, det_ms, trk_ms, n_tracks):
        """Add one sample, rescale the axes, and return the plot as a BGR image."""
        self.frames.append(frame_id)
        self.det_ms.append(det_ms)
        self.trk_ms.append(trk_ms)
        self.n_tracks.append(n_tracks)
        self.det_line.set_data(self.frames, self.det_ms)
        self.trk_line.set_data(self.frames, self.trk_ms)
        self.cnt_line.set_data(self.frames, self.n_tracks)
        self.ax.set_xlim(0, max(frame_id, 1))
        ymax = max(max(self.det_ms), max(self.trk_ms)) * 1.1
        self.ax.set_ylim(0, ymax if ymax > 0 else 1.0)
        self.ax2.set_ylim(0, max(max(self.n_tracks), 1) * 1.1)
        self.canvas.draw()
        rgb = np.ascontiguousarray(np.asarray(self.canvas.buffer_rgba())[..., :3])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bgr.shape[:2] != (self.height_px, self.width_px):  # guard against rounding
            bgr = cv2.resize(bgr, (self.width_px, self.height_px))
        return bgr


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.images, "*.png")) +
                   glob.glob(os.path.join(args.images, "*.jpg")))
    if args.max_frames:
        files = files[: args.max_frames]
    if not files:
        raise SystemExit(f"No .png/.jpg images found in {args.images}")

    device = torch.device(args.device)
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_score_thresh=args.score,
    ).to(device).eval()
    tracker = build_tracker(args.fps)

    first = cv2.imread(files[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (w, h + args.plot_height))
    plot = LatencyPlot(w, args.plot_height)
    detect(model, first, device)  # warm-up run (not timed) so one-off startup cost
                                  # does not dominate the latency plot

    csv_file = open(args.csv, "w", newline="") if args.csv else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(["frame", "track_id", "class", "x1", "y1", "x2", "y2", "score"])

    for idx, path in enumerate(tqdm(files, desc="Tracking")):
        frame = cv2.imread(path)
        if frame.shape[:2] != (h, w):  # a few KITTI drives vary slightly in size
            frame = cv2.resize(frame, (w, h))
        t0 = time.perf_counter()
        dets = detect(model, frame, device)
        t1 = time.perf_counter()
        tracked = tracker.update(dets)
        t2 = time.perf_counter()

        if len(tracked) and tracked.tracker_id is not None:
            tracked = tracked[tracked.tracker_id >= 0]
            frame = draw(frame, tracked)
            if csv_writer:
                for i in range(len(tracked)):
                    x1, y1, x2, y2 = tracked.xyxy[i]
                    csv_writer.writerow([
                        idx, int(tracked.tracker_id[i]),
                        CLASS_NAMES.get(int(tracked.class_id[i]), "?"),
                        f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                        f"{float(tracked.confidence[i]):.3f}",
                    ])
        panel = plot.update(idx, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, len(tracked))
        writer.write(np.vstack([frame, panel]))

    writer.release()
    if csv_file:
        csv_file.close()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()