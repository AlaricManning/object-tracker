import cv2
import numpy as np
import sys
import time
from ultralytics import YOLO
from norfair import Detection, Tracker, Video
from norfair.drawing import draw_boxes, draw_tracked_boxes


class YOLONorfairTracker:
    def __init__(self, model_size='n', conf_threshold=0.25, track_all=True):
        """
        Initialize YOLOv11 + Norfair multi-object tracker

        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
                       n = nano (fastest, least accurate)
                       s = small
                       m = medium
                       l = large
                       x = extra large (slowest, most accurate)
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            track_all: If True, track all detected objects. If False, only track clicked objects
        """
        self.conf_threshold = conf_threshold
        self.track_all = track_all
        self.selected_track_ids = set()  # IDs of objects user wants to track
        self.clicking_pos = None

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

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Get actual resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create window
        self.window_name = 'YOLOv11 + Norfair Object Tracker'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Color palette for different object classes
        self.colors = self._generate_colors(80)  # COCO has 80 classes

        print(f"Webcam resolution: {self.width}x{self.height}")
        print(f"Tracking mode: {'All objects' if track_all else 'Click to select'}")

    def _generate_colors(self, n):
        """Generate n distinct colors"""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for object selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicking_pos = (x, y)

    def yolo_detections_to_norfair_detections(self, yolo_results):
        """Convert YOLO detections to Norfair Detection objects"""
        norfair_detections = []

        # Get detection results
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]

                # Skip if confidence is too low
                if conf < self.conf_threshold:
                    continue

                # Calculate center point and dimensions
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                # Create Norfair detection
                # Norfair expects points in format [[x, y]]
                detection = Detection(
                    points=np.array([[center_x, center_y]]),
                    scores=np.array([conf]),
                    data={
                        'bbox': [x1, y1, x2, y2],
                        'class': cls,
                        'class_name': class_name,
                        'confidence': conf
                    }
                )
                norfair_detections.append(detection)

        return norfair_detections

    def check_click_on_object(self, tracked_objects):
        """Check if user clicked on any tracked object"""
        if self.clicking_pos is None:
            return

        click_x, click_y = self.clicking_pos
        self.clicking_pos = None  # Reset

        for obj in tracked_objects:
            if not obj.live_points.any():
                continue

            bbox = obj.estimate[0] if hasattr(obj, 'estimate') else None
            if bbox is None and 'bbox' in obj.last_detection.data:
                bbox = obj.last_detection.data['bbox']

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # Check if click is inside bounding box
                if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                    if obj.id in self.selected_track_ids:
                        self.selected_track_ids.remove(obj.id)
                        print(f"Unselected object #{obj.id}")
                    else:
                        self.selected_track_ids.add(obj.id)
                        print(f"Selected object #{obj.id} ({obj.last_detection.data.get('class_name', 'unknown')})")
                    break

    def draw_tracked_objects(self, frame, tracked_objects):
        """Draw bounding boxes and labels for tracked objects"""
        for obj in tracked_objects:
            if not obj.live_points.any():
                continue

            # Skip if in selective mode and not selected
            if not self.track_all and obj.id not in self.selected_track_ids:
                continue

            # Get bounding box from last detection
            if 'bbox' not in obj.last_detection.data:
                continue

            x1, y1, x2, y2 = obj.last_detection.data['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get class info
            class_name = obj.last_detection.data.get('class_name', 'unknown')
            confidence = obj.last_detection.data.get('confidence', 0.0)
            cls = obj.last_detection.data.get('class', 0)

            # Determine if this object is selected
            is_selected = obj.id in self.selected_track_ids

            # Choose color
            if is_selected:
                color = (0, 255, 0)  # Green for selected
                thickness = 3
            else:
                color = self.colors[cls % len(self.colors)]
                thickness = 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Prepare label
            label = f"#{obj.id} {class_name} {confidence:.2f}"
            if is_selected:
                label = f"[TRACKED] {label}"

            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 10),
                         (x1 + text_width + 5, y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_info(self, frame, fps, num_detections, num_tracked):
        """Draw information overlay"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Information text
        mode = "Track All Objects" if self.track_all else "Click to Select"
        info_text = [
            f"Model: YOLOv11 + Norfair | FPS: {fps}",
            f"Mode: {mode}",
            f"Detections: {num_detections} | Tracked: {num_tracked}",
            f"Selected: {len(self.selected_track_ids)}",
            "",
            "Controls:",
            "  Click - Select/Deselect object (selective mode)",
            "  M - Toggle tracking mode",
            "  R - Reset selected objects",
            "  Q - Quit"
        ]

        y_offset = 30
        for i, text in enumerate(info_text):
            color = (0, 255, 255) if i < 4 else (200, 200, 200)
            font_scale = 0.6 if i < 4 else 0.5
            cv2.putText(frame, text, (15, y_offset + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1 if i >= 4 else 2)

    def run(self):
        """Main tracking loop"""
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
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break

                # Run YOLO detection
                results = self.model(frame, conf=self.conf_threshold, verbose=False)

                # Convert to Norfair detections
                detections = self.yolo_detections_to_norfair_detections(results)

                # Update tracker
                tracked_objects = self.tracker.update(detections=detections)

                # Check for clicks on objects
                if not self.track_all:
                    self.check_click_on_object(tracked_objects)

                # Draw tracked objects
                self.draw_tracked_objects(frame, tracked_objects)

                # Calculate FPS
                fps_counter += 1
                if time.time() - start_time > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    start_time = time.time()

                # Count tracked objects
                if self.track_all:
                    num_tracked = len([obj for obj in tracked_objects if obj.live_points.any()])
                else:
                    num_tracked = len([obj for obj in tracked_objects
                                      if obj.live_points.any() and obj.id in self.selected_track_ids])

                # Draw info overlay
                self.draw_info(frame, fps_display, len(detections), num_tracked)

                # Display frame
                cv2.imshow(self.window_name, frame)

                # Handle keyboard input
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
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            print("Tracker closed")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='YOLOv11 + Norfair Object Tracker - Real-time multi-object tracking'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLO model size: n(nano-fastest), s(small), m(medium), l(large), x(extra-slowest but accurate)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (0.0-1.0, default: 0.25)'
    )
    parser.add_argument(
        '--selective',
        action='store_true',
        help='Start in selective mode (click to track specific objects)'
    )

    args = parser.parse_args()

    try:
        tracker_app = YOLONorfairTracker(
            model_size=args.model,
            conf_threshold=args.conf,
            track_all=not args.selective
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
