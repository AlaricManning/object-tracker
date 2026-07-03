import cv2
import numpy as np


def draw_tracked_objects(frame, tracked_objects, track_all, selected_track_ids, target_classes, colors):
    for obj in tracked_objects:
        if not hasattr(obj, 'last_detection') or obj.last_detection.points is None:
            continue
        if not track_all and obj.id not in selected_track_ids:
            continue
        if 'bbox' not in obj.last_detection.data:
            continue

        x1, y1, x2, y2 = obj.last_detection.data['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        class_name = obj.last_detection.data.get('class_name', 'unknown')
        confidence = obj.last_detection.data.get('confidence', 0.0)
        cls        = obj.last_detection.data.get('class_id', 0)

        is_selected = obj.id in selected_track_ids
        is_target   = class_name in target_classes

        if is_selected:
            color, thickness = (0, 255, 0), 3
        elif is_target:
            color, thickness = (0, 165, 255), 3
        else:
            color, thickness = colors[cls % len(colors)], 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"#{obj.id} {class_name} {confidence:.2f}"
        if is_selected:
            label = f"[TRACKED] {label}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_info(frame, fps, num_detections, num_tracked,
              track_all, selected_track_ids, target_classes, active_clip, clip_count):
    overlay = frame.copy()
    box_h = 200 if target_classes else 180
    cv2.rectangle(overlay, (10, 10), (500, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    mode = "Track All Objects" if track_all else "Click to Select"
    if active_clip:
        clip_line = f"  REC {active_clip['clip_id']} | peak {active_clip['peak_confidence']:.0%}"
    elif target_classes:
        clip_line = f"  Clips saved: {clip_count}"
    else:
        clip_line = None

    info_text = [
        f"Model: YOLOv11 + Norfair | FPS: {fps}",
        f"Mode: {mode}",
        f"Detections: {num_detections} | Tracked: {num_tracked}",
        f"Selected: {len(selected_track_ids)}",
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
        if clip_line is not None and i == 4 and active_clip:
            color = (0, 0, 255)
        font_scale = 0.6 if is_status else 0.5
        cv2.putText(frame, text, (15, y_offset + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2 if is_status else 1)
