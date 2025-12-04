import cv2
import numpy as np
import sys
import time


class MultiObjectTracker:
    def __init__(self, tracker_type='KCF'):
        """
        Initialize multi-object webcam tracker

        Args:
            tracker_type: 'CSRT', 'KCF', 'MOSSE', 'MIL'
        """
        self.tracker_type = tracker_type
        self.trackers = []  # List of (tracker, bbox, tracker_id, color) tuples
        self.next_tracker_id = 1

        # Selection state
        self.selecting = False
        self.init_pos = None
        self.current_pos = None

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
        self.window_name = 'Multi-Object Webcam Tracker'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Color palette for different tracked objects
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Lime
        ]

    def get_color(self, tracker_id):
        """Get a unique color for a tracker ID"""
        return self.colors[(tracker_id - 1) % len(self.colors)]

    def create_tracker(self):
        """Create a new tracker instance"""
        tracker_types = {
            'CSRT': cv2.TrackerCSRT_create,
            'KCF': cv2.TrackerKCF_create,
            'MOSSE': cv2.TrackerMOSSE_create,
            'MIL': cv2.TrackerMIL_create,
        }

        if self.tracker_type in tracker_types:
            return tracker_types[self.tracker_type]()
        else:
            print(f"Unknown tracker type: {self.tracker_type}, using KCF")
            return cv2.TrackerKCF_create()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for object selection"""

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start selection
            self.selecting = True
            self.init_pos = (x, y)
            self.current_pos = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            # Update current position while selecting
            if self.selecting:
                self.current_pos = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # End selection and add new tracker
            self.selecting = False
            self.current_pos = (x, y)

            # Calculate bounding box
            x1, y1 = self.init_pos
            x2, y2 = self.current_pos

            # Ensure proper ordering
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # Create bbox (x, y, width, height)
            w = x_max - x_min
            h = y_max - y_min

            # Minimum size check
            if w > 20 and h > 20:
                bbox = (x_min, y_min, w, h)
                # Will be initialized on next frame
                return bbox

    def add_tracker(self, frame, bbox):
        """Add a new tracker for the selected bounding box"""
        tracker = self.create_tracker()
        success = tracker.init(frame, bbox)

        if success:
            tracker_id = self.next_tracker_id
            color = self.get_color(tracker_id)
            self.trackers.append((tracker, bbox, tracker_id, color))
            self.next_tracker_id += 1
            print(f"Tracker #{tracker_id} initialized at bbox: {bbox}")
            return True
        else:
            print("Failed to initialize tracker")
            return False

    def remove_last_tracker(self):
        """Remove the most recently added tracker"""
        if self.trackers:
            removed = self.trackers.pop()
            tracker_id = removed[2]
            print(f"Removed tracker #{tracker_id}")
        else:
            print("No trackers to remove")

    def reset_all_trackers(self):
        """Clear all trackers"""
        count = len(self.trackers)
        self.trackers.clear()
        self.next_tracker_id = 1
        print(f"Reset: removed {count} tracker(s)")

    def draw_selection(self, frame):
        """Draw selection rectangle while selecting"""
        if self.selecting and self.init_pos and self.current_pos:
            x1, y1 = self.init_pos
            x2, y2 = self.current_pos

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Draw text
            cv2.putText(frame, "Selecting...", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def draw_bbox(self, frame, bbox, tracker_id, color, success):
        """Draw tracking bounding box"""
        x, y, w, h = [int(v) for v in bbox]

        # Adjust color brightness based on tracking success
        if not success:
            color = tuple(int(c * 0.5) for c in color)  # Dim color if tracking lost

        # Draw rectangle
        thickness = 3 if success else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        # Draw label with ID
        label = f"#{tracker_id}"
        if not success:
            label += " (LOST)"

        # Background for text
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(frame, (x, y - text_height - 10),
                     (x + text_width + 5, y), color, -1)

        # Text
        cv2.putText(frame, label, (x + 2, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_info(self, frame, fps):
        """Draw information overlay"""
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Instructions
        info_text = [
            f"Tracker: {self.tracker_type} | FPS: {fps}",
            f"Active Trackers: {len(self.trackers)}",
            "",
            "Controls:",
            "  Click & Drag - Select object to track",
            "  D - Delete last tracker",
            "  R - Reset all trackers",
            "  Q - Quit"
        ]

        y_offset = 30
        for i, text in enumerate(info_text):
            color = (0, 255, 255) if i < 2 else (200, 200, 200)
            font_scale = 0.6 if i < 2 else 0.5
            cv2.putText(frame, text, (15, y_offset + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1 if i >= 2 else 2)

    def update_trackers(self, frame):
        """Update all trackers and remove failed ones"""
        updated_trackers = []

        for tracker, bbox, tracker_id, color in self.trackers:
            success, new_bbox = tracker.update(frame)

            if success:
                # Keep successful tracker with updated bbox
                updated_trackers.append((tracker, new_bbox, tracker_id, color))
                self.draw_bbox(frame, new_bbox, tracker_id, color, True)
            else:
                # Draw last known position with "LOST" indicator
                self.draw_bbox(frame, bbox, tracker_id, color, False)
                # Keep tracker for one more frame in case it recovers
                # In production, you might want to remove it after N failed frames
                updated_trackers.append((tracker, bbox, tracker_id, color))

        self.trackers = updated_trackers

    def run(self):
        """Main tracking loop"""
        print(f"Starting Multi-Object Webcam Tracker with {self.tracker_type}")
        print(f"Resolution: {self.width}x{self.height}")
        print("\nControls:")
        print("  Click and drag to select objects to track")
        print("  D - Delete last tracker")
        print("  R - Reset all trackers")
        print("  Q - Quit\n")

        fps_display = 0
        fps_counter = 0
        start_time = time.time()
        pending_bbox = None

        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break

            # Initialize pending tracker if bbox was selected
            if pending_bbox is not None:
                self.add_tracker(frame, pending_bbox)
                pending_bbox = None

            # Check if selection was completed
            if not self.selecting and self.init_pos and self.current_pos:
                x1, y1 = self.init_pos
                x2, y2 = self.current_pos

                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                w = x_max - x_min
                h = y_max - y_min

                if w > 20 and h > 20:
                    pending_bbox = (x_min, y_min, w, h)

                # Reset selection state
                self.init_pos = None
                self.current_pos = None

            # Update all trackers
            if self.trackers:
                self.update_trackers(frame)

            # Draw selection rectangle if selecting
            self.draw_selection(frame)

            # Calculate FPS
            fps_counter += 1
            if time.time() - start_time > 1.0:
                fps_display = fps_counter
                fps_counter = 0
                start_time = time.time()

            # Draw info overlay
            self.draw_info(frame, fps_display)

            # Display frame
            cv2.imshow(self.window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                self.reset_all_trackers()
            elif key == ord('d') or key == ord('D'):
                self.remove_last_tracker()

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nTracker closed")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-Object Webcam Tracker - Track multiple objects in real-time'
    )
    parser.add_argument(
        '--tracker',
        type=str,
        default='KCF',
        choices=['CSRT', 'KCF', 'MOSSE', 'MIL'],
        help='Tracker algorithm to use (default: KCF for balanced performance)'
    )

    args = parser.parse_args()

    try:
        tracker_app = MultiObjectTracker(tracker_type=args.tracker)
        tracker_app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
