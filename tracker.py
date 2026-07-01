import json
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker


class YOLONorfairTracker:
    def __init__(self, model_size='n', conf_threshold=0.25, track_all=True,
                 target_classes=None, hit_threshold=0.80, near_miss_threshold=0.30,
                 export=False, output_dir='./sessions'):
        self.conf_threshold = conf_threshold
        self.track_all = track_all
        self.selected_track_ids = set()
        self.clicking_pos = None

        # Capture config
        self.target_classes = set(target_classes) if target_classes else set()
        self.hit_threshold = hit_threshold
        self.near_miss_threshold = near_miss_threshold
        self.export = export
        self.output_dir = output_dir

        # Session state
        self.session_id = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        self.session_start = datetime.now()
        self.frame_index = 0
        self.clip_count = 0
        self.object_counts = {}  # class_name -> set of unique track_ids seen

        # Clip state
        self.active_clip = None
        self.preroll = None       # deque, sized after FPS is known
        self.postroll_frames = 0

        # Output handles
        self.session_dir = None
        self.metadata_file = None

        # Initialize YOLO model
        print(f"Loading YOLOv11{model_size} model...")
        self.model = YOLO(f'yolo11{model_size}.pt')
        print("Model loaded successfully!")

        # Initialize Norfair tracker
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=100,
            hit_counter_max=10,
            initialization_delay=3,
            pointwise_hit_counter_max=4
        )

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.record_fps = max(self.cap.get(cv2.CAP_PROP_FPS) or 20, 20)

        # Size pre/post-roll buffers at 2 seconds each
        preroll_frames = int(self.record_fps * 2)
        self.postroll_frames = int(self.record_fps * 2)
        self.preroll = deque(maxlen=preroll_frames)

        # Create session directory and open metadata file if needed
        if self.target_classes or export:
            self.session_dir = os.path.join(output_dir, self.session_id)
            os.makedirs(self.session_dir, exist_ok=True)

        if export and self.target_classes:
            jsonl_path = os.path.join(self.session_dir, 'detections.jsonl')
            self.metadata_file = open(jsonl_path, 'w')
            print(f"Exporting detection metadata to: {jsonl_path}")

        self.window_name = 'YOLOv11 + Norfair Object Tracker'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.colors = self._generate_colors(80)

        print(f"Webcam resolution: {self.width}x{self.height}")
        print(f"Tracking mode: {'All objects' if track_all else 'Click to select'}")
        if self.target_classes:
            print(f"Target classes: {', '.join(self.target_classes)}")
            print(f"Hit threshold: {hit_threshold:.0%} | Near-miss threshold: {near_miss_threshold:.0%}")

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
        cls = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
            if c < self.conf_threshold:
                continue
            center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
            norfair_detections.append(Detection(
                points=center,
                scores=np.array([c]),
                data={
                    'bbox': [x1, y1, x2, y2],
                    'class_id': cl,
                    'class_name': self.model.names[cl],
                    'confidence': float(c)
                }
            ))
        return norfair_detections

    def _target_detections(self, tracked_objects):
        """Return live tracked objects matching target classes above near-miss threshold."""
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
        clip_id = f"clip_{self.clip_count:04d}"
        tmp_path = os.path.join(self.session_dir, f'_{clip_id}_tmp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_path, fourcc, self.record_fps, (self.width, self.height))

        for buffered_frame in self.preroll:
            writer.write(buffered_frame)

        self.active_clip = {
            'clip_id': clip_id,
            'tmp_path': tmp_path,
            'writer': writer,
            'class_name': class_name,
            'peak_confidence': 0.0,
            'frames_since_detection': 0,
            'detections': [],
            'start_time': datetime.now().isoformat(),
            'start_frame': self.frame_index,
        }
        print(f"[CLIP] Started {clip_id} for '{class_name}'")

    def _close_clip(self):
        clip = self.active_clip
        clip['writer'].release()

        tier = 'hits' if clip['peak_confidence'] >= self.hit_threshold else 'near_misses'
        tier_dir = os.path.join(self.session_dir, tier)
        os.makedirs(tier_dir, exist_ok=True)

        final_mp4 = os.path.join(tier_dir, f"{clip['clip_id']}.mp4")
        os.rename(clip['tmp_path'], final_mp4)

        sidecar = {
            'session_id': self.session_id,
            'clip_id': clip['clip_id'],
            'tier': tier,
            'class_name': clip['class_name'],
            'peak_confidence': round(clip['peak_confidence'], 4),
            'start_time': clip['start_time'],
            'end_time': datetime.now().isoformat(),
            'total_frames': self.frame_index - clip['start_frame'],
            'detections': clip['detections'],
        }
        sidecar_path = os.path.join(tier_dir, f"{clip['clip_id']}.json")
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar, f, indent=2)

        print(f"[CLIP] Saved {tier}/{clip['clip_id']}.mp4 (peak conf: {clip['peak_confidence']:.0%})")
        self.active_clip = None

    def _update_clip(self, raw_frame, target_hits):
        """Drive the clip state machine for one frame."""
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
                    self.active_clip['detections'].append({
                        'frame_id': self.frame_index,
                        'timestamp': datetime.now().isoformat(),
                        'track_id': obj.id,
                        'confidence': round(float(conf), 4),
                        'bbox': [round(float(v), 1) for v in obj.last_detection.data['bbox']],
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
            'frame_id': self.frame_index,
            'timestamp': datetime.now().isoformat(),
            'detections': [
                {
                    'track_id': obj.id,
                    'class_name': obj.last_detection.data['class_name'],
                    'class_id': int(obj.last_detection.data['class_id']),
                    'confidence': round(obj.last_detection.data['confidence'], 4),
                    'bbox': [round(float(v), 1) for v in obj.last_detection.data['bbox']],
                }
                for obj in target_hits
            ],
        }
        self.metadata_file.write(json.dumps(record) + '\n')

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_tracked_objects(self, frame, tracked_objects):
        for obj in tracked_objects:
            if not hasattr(obj, 'last_detection') or obj.last_detection.points is None:
                continue
            if not self.track_all and obj.id not in self.selected_track_ids:
                continue
            if 'bbox' not in obj.last_detection.data:
                continue

            x1, y1, x2, y2 = obj.last_detection.data['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            class_name = obj.last_detection.data.get('class_name', 'unknown')
            confidence = obj.last_detection.data.get('confidence', 0.0)
            cls = obj.last_detection.data.get('class_id', 0)

            is_selected = obj.id in self.selected_track_ids
            is_target = class_name in self.target_classes

            if is_selected:
                color, thickness = (0, 255, 0), 3
            elif is_target:
                color, thickness = (0, 165, 255), 3  # orange for target class
            else:
                color, thickness = self.colors[cls % len(self.colors)], 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"#{obj.id} {class_name} {confidence:.2f}"
            if is_selected:
                label = f"[TRACKED] {label}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_info(self, frame, fps, num_detections, num_tracked):
        overlay = frame.copy()
        box_h = 200 if self.target_classes else 180
        cv2.rectangle(overlay, (10, 10), (500, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        mode = "Track All Objects" if self.track_all else "Click to Select"
        if self.active_clip:
            clip_line = f"  REC {self.active_clip['clip_id']} | peak {self.active_clip['peak_confidence']:.0%}"
        elif self.target_classes:
            clip_line = f"  Clips saved: {self.clip_count}"
        else:
            clip_line = None

        info_text = [
            f"Model: YOLOv11 + Norfair | FPS: {fps}",
            f"Mode: {mode}",
            f"Detections: {num_detections} | Tracked: {num_tracked}",
            f"Selected: {len(self.selected_track_ids)}",
        ]
        if clip_line is not None:
            info_text.append(clip_line)
        info_text += ["", "Controls:",
                      "  Click - Select/Deselect object (selective mode)",
                      "  M - Toggle tracking mode", "  R - Reset selected objects", "  Q - Quit"]

        num_status = 5 if clip_line is not None else 4
        y_offset = 30
        for i, text in enumerate(info_text):
            is_status = i < num_status
            color = (0, 255, 255) if is_status else (200, 200, 200)
            if clip_line is not None and i == 4 and self.active_clip:
                color = (0, 0, 255)  # red while recording
            font_scale = 0.6 if is_status else 0.5
            cv2.putText(frame, text, (15, y_offset + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2 if is_status else 1)

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
        start_time = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break

                # Keep raw copy for preroll/clip recording (before annotation)
                raw_frame = frame.copy()

                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                detections = self.yolo_detections_to_norfair_detections(results)
                tracked_objects = self.tracker.update(detections=detections)

                target_hits = self._target_detections(tracked_objects)

                # Clip recording and export (uses unannotated raw_frame)
                self._update_clip(raw_frame, target_hits)
                self._write_export(target_hits)

                # Preroll always gets the current raw frame
                self.preroll.append(raw_frame)
                self.frame_index += 1

                if not self.track_all:
                    self.check_click_on_object(tracked_objects)

                self.draw_tracked_objects(frame, tracked_objects)

                fps_counter += 1
                if time.time() - start_time > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    start_time = time.time()

                if self.track_all:
                    num_tracked = len([obj for obj in tracked_objects if obj.live_points.any()])
                else:
                    num_tracked = len([obj for obj in tracked_objects
                                       if obj.live_points.any() and obj.id in self.selected_track_ids])

                self.draw_info(frame, fps_display, len(detections), num_tracked)
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
                    'session_id': self.session_id,
                    'start_time': self.session_start.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_frames': self.frame_index,
                    'clips_saved': self.clip_count,
                    'target_classes': list(self.target_classes),
                    'hit_threshold': self.hit_threshold,
                    'near_miss_threshold': self.near_miss_threshold,
                    'object_counts': {k: len(v) for k, v in self.object_counts.items()},
                }
                summary_path = os.path.join(self.session_dir, 'session_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"Session summary: {summary_path}")

            if self.metadata_file:
                self.metadata_file.close()

            self.cap.release()
            cv2.destroyAllWindows()
            print("Tracker closed")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='YOLOv11 + Norfair Object Tracker - Real-time multi-object tracking'
    )
    parser.add_argument(
        '--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
        help='YOLO model size: n(nano-fastest), s(small), m(medium), l(large), x(extra-slowest but accurate)'
    )
    parser.add_argument(
        '--conf', type=float, default=0.25,
        help='Confidence threshold for detections (0.0-1.0, default: 0.25)'
    )
    parser.add_argument(
        '--selective', action='store_true',
        help='Start in selective mode (click to track specific objects)'
    )
    parser.add_argument(
        '--target-class', nargs='+', metavar='CLASS', dest='target_classes',
        help='Class name(s) to trigger clip recording (e.g. --target-class person car)'
    )
    parser.add_argument(
        '--hit-threshold', type=float, default=0.80,
        help='Confidence >= this is saved as a "hit" clip (default: 0.80)'
    )
    parser.add_argument(
        '--near-miss-threshold', type=float, default=0.30,
        help='Confidence >= this triggers recording; below hit-threshold = "near_miss" (default: 0.30)'
    )
    parser.add_argument(
        '--export', action='store_true',
        help='Write per-frame detection metadata to detections.jsonl'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./sessions',
        help='Root directory for session output (default: ./sessions)'
    )

    args = parser.parse_args()

    try:
        tracker_app = YOLONorfairTracker(
            model_size=args.model,
            conf_threshold=args.conf,
            track_all=not args.selective,
            target_classes=args.target_classes,
            hit_threshold=args.hit_threshold,
            near_miss_threshold=args.near_miss_threshold,
            export=args.export,
            output_dir=args.output_dir,
        )
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
