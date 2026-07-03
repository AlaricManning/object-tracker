import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from norfair import Detection, Tracker

import klv
import display as hud
from uploader import S3Uploader


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class YOLONorfairTracker:
    def __init__(self, config: dict):
        model_cfg    = config.get('model', {})
        tracking_cfg = config.get('tracking', {})
        capture_cfg  = config.get('capture', {})
        upload_cfg   = config.get('upload', {})

        self.conf_threshold = model_cfg.get('conf_threshold', 0.25)
        self.track_all = tracking_cfg.get('mode', 'all') == 'all'
        self.selected_track_ids = set()
        self.clicking_pos = None

        # Capture config
        self.target_classes      = set(capture_cfg.get('target_classes') or [])
        self.hit_threshold       = capture_cfg.get('hit_threshold', 0.80)
        self.near_miss_threshold = capture_cfg.get('near_miss_threshold', 0.30)
        self.export              = capture_cfg.get('export', False)
        self.output_dir          = capture_cfg.get('output_dir', './sessions')

        # S3 uploader
        self.uploader = None
        if upload_cfg.get('enabled') and upload_cfg.get('s3_bucket'):
            self.uploader = S3Uploader(
                bucket=upload_cfg['s3_bucket'],
                prefix=upload_cfg.get('s3_prefix', ''),
                region=upload_cfg.get('region'),
            )
            print(f"S3 upload enabled: s3://{upload_cfg['s3_bucket']}/{upload_cfg.get('s3_prefix', '')}")

        # Session state
        self.session_id    = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        self.session_start = datetime.now()
        self.frame_index   = 0
        self.clip_count    = 0
        self.object_counts = {}

        # Clip state
        self.active_clip   = None
        self.preroll       = None
        self.postroll_frames = 0

        # Output handles
        self.session_dir   = None
        self.metadata_file = None

        # YOLO model
        model_size = model_cfg.get('size', 'n')
        print(f"Loading YOLOv11{model_size} model...")
        self.model = YOLO(f'yolo11{model_size}.pt')
        print("Model loaded successfully!")

        # Norfair tracker
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=100,
            hit_counter_max=10,
            initialization_delay=3,
            pointwise_hit_counter_max=4
        )

        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.width      = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height     = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.record_fps = max(self.cap.get(cv2.CAP_PROP_FPS) or 20, 20)

        preroll_frames       = int(self.record_fps * 2)
        self.postroll_frames = int(self.record_fps * 2)
        self.preroll         = deque(maxlen=preroll_frames)

        if self.target_classes or self.export:
            self.session_dir = os.path.join(self.output_dir, self.session_id)
            os.makedirs(self.session_dir, exist_ok=True)

        if self.export and self.target_classes:
            jsonl_path = os.path.join(self.session_dir, 'detections.jsonl')
            self.metadata_file = open(jsonl_path, 'w')
            print(f"Exporting detection metadata to: {jsonl_path}")

        self.window_name = 'YOLOv11 + Norfair Object Tracker'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.colors = self._generate_colors(80)

        print(f"Webcam resolution: {self.width}x{self.height}")
        print(f"Tracking mode: {'All objects' if self.track_all else 'Click to select'}")
        if self.target_classes:
            print(f"Target classes: {', '.join(self.target_classes)}")
            print(f"Hit threshold: {self.hit_threshold:.0%} | Near-miss threshold: {self.near_miss_threshold:.0%}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_colors(self, n):
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicking_pos = (x, y)

    def yolo_detections_to_norfair_detections(self, yolo_results):
        norfair_detections = []
        result = yolo_results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return norfair_detections

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
            if c < self.conf_threshold:
                continue
            center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
            norfair_detections.append(Detection(
                points=center,
                scores=np.array([c]),
                data={
                    'bbox':       [x1, y1, x2, y2],
                    'class_id':   cl,
                    'class_name': self.model.names[cl],
                    'confidence': float(c),
                }
            ))
        return norfair_detections

    def _target_detections(self, tracked_objects):
        if not self.target_classes:
            return []
        hits = []
        for obj in tracked_objects:
            if not obj.live_points.any() or obj.last_detection is None:
                continue
            data = obj.last_detection.data
            if 'class_name' not in data or 'bbox' not in data:
                continue
            if (data['class_name'] in self.target_classes
                    and data['confidence'] >= self.near_miss_threshold):
                hits.append(obj)
        return hits

    def check_click_on_object(self, tracked_objects):
        if self.clicking_pos is None:
            return
        click_x, click_y = self.clicking_pos
        self.clicking_pos = None

        for obj in tracked_objects:
            if obj.last_detection is None or 'bbox' not in obj.last_detection.data:
                continue
            x1, y1, x2, y2 = obj.last_detection.data['bbox']
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                if obj.id in self.selected_track_ids:
                    self.selected_track_ids.remove(obj.id)
                    print(f"Unselected object #{obj.id}")
                else:
                    self.selected_track_ids.add(obj.id)
                    class_name = obj.last_detection.data.get('class_name', 'unknown')
                    print(f"Selected object #{obj.id} ({class_name})")
                break

    # ------------------------------------------------------------------
    # Clip recording
    # ------------------------------------------------------------------

    def _start_clip(self, class_name):
        self.clip_count += 1
        clip_id  = f"clip_{self.clip_count:04d}"
        tmp_mp4  = os.path.join(self.session_dir, f'_{clip_id}_tmp.mp4')
        tmp_klv  = os.path.join(self.session_dir, f'_{clip_id}_tmp.klv')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_mp4, fourcc, self.record_fps, (self.width, self.height))

        for buffered_frame in self.preroll:
            writer.write(buffered_frame)

        self.active_clip = {
            'clip_id':              clip_id,
            'tmp_mp4':              tmp_mp4,
            'tmp_klv':              tmp_klv,
            'writer':               writer,
            'klv_file':             open(tmp_klv, 'wb'),
            'class_name':           class_name,
            'peak_confidence':      0.0,
            'frames_since_detection': 0,
            'detections':           [],
            'start_time':           datetime.now().isoformat(),
            'start_frame':          self.frame_index,
        }
        print(f"[CLIP] Started {clip_id} for '{class_name}'")

    def _remux_to_ts(self, mp4_path: str, ts_path: str):
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', mp4_path,
             '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast',
             '-f', 'mpegts', ts_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg transcode failed:\n{result.stderr.decode()}")

    def _close_clip(self):
        clip = self.active_clip
        clip['writer'].release()
        clip['klv_file'].close()

        tier    = 'hits' if clip['peak_confidence'] >= self.hit_threshold else 'near_misses'
        tier_dir = os.path.join(self.session_dir, tier)
        os.makedirs(tier_dir, exist_ok=True)

        clip_id  = clip['clip_id']
        ts_path  = os.path.join(tier_dir, f'{clip_id}.ts')
        klv_path = os.path.join(tier_dir, f'{clip_id}.klv')

        # Remux MP4 → MPEG-TS, then clean up temp MP4
        self._remux_to_ts(clip['tmp_mp4'], ts_path)
        os.remove(clip['tmp_mp4'])

        # Move KLV sidecar into tier folder
        os.rename(clip['tmp_klv'], klv_path)

        # JSON sidecar for human inspection
        sidecar = {
            'session_id':       self.session_id,
            'clip_id':          clip_id,
            'tier':             tier,
            'class_name':       clip['class_name'],
            'peak_confidence':  round(clip['peak_confidence'], 4),
            'start_time':       clip['start_time'],
            'end_time':         datetime.now().isoformat(),
            'total_frames':     self.frame_index - clip['start_frame'],
            'detections':       clip['detections'],
        }
        with open(os.path.join(tier_dir, f'{clip_id}.json'), 'w') as f:
            json.dump(sidecar, f, indent=2)

        print(f"[CLIP] Saved {tier}/{clip_id}.ts (peak conf: {clip['peak_confidence']:.0%})")

        if self.uploader:
            try:
                ts_uri, klv_uri = self.uploader.upload_clip(
                    ts_path, klv_path, self.session_id, clip_id, tier
                )
                print(f"[S3]  Uploaded → {ts_uri}")
                print(f"[S3]          → {klv_uri}")
            except RuntimeError as e:
                print(f"[S3]  Upload failed: {e}")

        self.active_clip = None

    def _update_clip(self, raw_frame, target_hits):
        if not self.target_classes:
            return

        if self.active_clip is None and target_hits:
            best = max(target_hits, key=lambda o: o.last_detection.data['confidence'])
            self._start_clip(best.last_detection.data['class_name'])

        if self.active_clip is not None:
            self.active_clip['writer'].write(raw_frame)
            if target_hits:
                self.active_clip['frames_since_detection'] = 0
                for obj in target_hits:
                    conf = obj.last_detection.data['confidence']
                    if conf > self.active_clip['peak_confidence']:
                        self.active_clip['peak_confidence'] = conf

                    now = datetime.now().isoformat()
                    packet = klv.encode_packet(
                        frame_id=self.frame_index,
                        timestamp=now,
                        session_id=self.session_id,
                        clip_id=self.active_clip['clip_id'],
                        track_id=int(obj.id),
                        class_name=obj.last_detection.data['class_name'],
                        confidence=float(conf),
                        bbox=obj.last_detection.data['bbox'],
                    )
                    self.active_clip['klv_file'].write(packet)

                    self.active_clip['detections'].append({
                        'frame_id':   self.frame_index,
                        'timestamp':  now,
                        'track_id':   obj.id,
                        'confidence': round(float(conf), 4),
                        'bbox':       [round(float(v), 1) for v in obj.last_detection.data['bbox']],
                    })
                    cls = obj.last_detection.data['class_name']
                    self.object_counts.setdefault(cls, set()).add(obj.id)
            else:
                self.active_clip['frames_since_detection'] += 1
                if self.active_clip['frames_since_detection'] >= self.postroll_frames:
                    self._close_clip()

    def _write_export(self, target_hits):
        if not self.metadata_file or not target_hits:
            return
        record = {
            'session_id': self.session_id,
            'frame_id':   self.frame_index,
            'timestamp':  datetime.now().isoformat(),
            'detections': [
                {
                    'track_id':   obj.id,
                    'class_name': obj.last_detection.data['class_name'],
                    'class_id':   int(obj.last_detection.data['class_id']),
                    'confidence': round(obj.last_detection.data['confidence'], 4),
                    'bbox':       [round(float(v), 1) for v in obj.last_detection.data['bbox']],
                }
                for obj in target_hits
            ],
        }
        self.metadata_file.write(json.dumps(record) + '\n')

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print(f"\nStarting YOLOv11 + Norfair Object Tracker")
        print("\nControls:")
        print("  Click - Select/Deselect specific objects (in selective mode)")
        print("  M - Toggle between 'Track All' and 'Click to Select' modes")
        print("  R - Reset selected objects")
        print("  Q - Quit\n")

        fps_display = 0
        fps_counter = 0
        start_time  = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break

                raw_frame = frame.copy()

                results        = self.model(frame, conf=self.conf_threshold, verbose=False)
                detections     = self.yolo_detections_to_norfair_detections(results)
                tracked_objects = self.tracker.update(detections=detections)

                target_hits = self._target_detections(tracked_objects)

                self._update_clip(raw_frame, target_hits)
                self._write_export(target_hits)

                self.preroll.append(raw_frame)
                self.frame_index += 1

                if not self.track_all:
                    self.check_click_on_object(tracked_objects)

                hud.draw_tracked_objects(frame, tracked_objects,
                                         self.track_all, self.selected_track_ids,
                                         self.target_classes, self.colors)

                fps_counter += 1
                if time.time() - start_time > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    start_time  = time.time()

                if self.track_all:
                    num_tracked = len([obj for obj in tracked_objects if obj.live_points.any()])
                else:
                    num_tracked = len([obj for obj in tracked_objects
                                       if obj.live_points.any() and obj.id in self.selected_track_ids])

                hud.draw_info(frame, fps_display, len(detections), num_tracked,
                              self.track_all, self.selected_track_ids,
                              self.target_classes, self.active_clip, self.clip_count)
                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    self.selected_track_ids.clear()
                    print("Reset: cleared all selected objects")
                elif key == ord('m') or key == ord('M'):
                    self.track_all = not self.track_all
                    mode = "Track All Objects" if self.track_all else "Click to Select"
                    print(f"Switched to: {mode}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            if self.active_clip:
                self._close_clip()

            if self.session_dir:
                summary = {
                    'session_id':         self.session_id,
                    'start_time':         self.session_start.isoformat(),
                    'end_time':           datetime.now().isoformat(),
                    'total_frames':       self.frame_index,
                    'clips_saved':        self.clip_count,
                    'target_classes':     list(self.target_classes),
                    'hit_threshold':      self.hit_threshold,
                    'near_miss_threshold': self.near_miss_threshold,
                    'object_counts':      {k: len(v) for k, v in self.object_counts.items()},
                }
                summary_path = os.path.join(self.session_dir, 'session_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"Session summary: {summary_path}")

                if self.uploader:
                    try:
                        uri = self.uploader.upload_summary(summary_path, self.session_id)
                        print(f"[S3]  Summary → {uri}")
                    except RuntimeError as e:
                        print(f"[S3]  Summary upload failed: {e}")

            if self.metadata_file:
                self.metadata_file.close()

            self.cap.release()
            cv2.destroyAllWindows()
            print("Tracker closed")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='YOLOv11 + Norfair Object Tracker'
    )
    parser.add_argument(
        '--config', default='config.yaml',
        help='Path to YAML config file (default: config.yaml)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        print("Copy config.example.yaml to config.yaml and edit it.")
        sys.exit(1)

    config = load_config(args.config)

    try:
        tracker_app = YOLONorfairTracker(config)
        tracker_app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
