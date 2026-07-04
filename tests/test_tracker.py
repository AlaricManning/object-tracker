"""
Tests for tracker.py — load_config and pure helper methods.
YOLONorfairTracker.__init__ is patched out so tests run without a webcam,
YOLO model, or any hardware.
"""
import os
import tempfile
import pytest
import yaml
from unittest.mock import MagicMock, patch

from tracker import load_config, YOLONorfairTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tracker(**attrs):
    """Instantiate YOLONorfairTracker without running __init__."""
    with patch.object(YOLONorfairTracker, '__init__', return_value=None):
        t = YOLONorfairTracker({})
    defaults = dict(
        target_classes={'person'},
        near_miss_threshold=0.30,
        selected_track_ids=set(),
        clicking_pos=None,
        track_all=True,
    )
    for k, v in {**defaults, **attrs}.items():
        setattr(t, k, v)
    return t


def make_tracked_obj(obj_id=1, class_name='person', confidence=0.9,
                     bbox=None, live=True):
    import numpy as np
    bbox = bbox or [10.0, 10.0, 100.0, 100.0]
    obj = MagicMock()
    obj.id = obj_id
    obj.live_points.any.return_value = live
    obj.last_detection = MagicMock()
    obj.last_detection.data = {
        'class_name': class_name,
        'confidence': confidence,
        'bbox':       bbox,
    }
    return obj


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_returns_dict():
    cfg = {'model': {'size': 'n'}, 'tracking': {'mode': 'all'}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name
    try:
        result = load_config(path)
        assert result == cfg
    finally:
        os.unlink(path)


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config('/nonexistent/path/config.yaml')


def test_load_config_nested_values():
    cfg = {
        'capture': {
            'target_classes': ['person', 'car'],
            'hit_threshold': 0.80,
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name
    try:
        result = load_config(path)
        assert result['capture']['target_classes'] == ['person', 'car']
        assert result['capture']['hit_threshold'] == 0.80
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# _target_detections
# ---------------------------------------------------------------------------

def test_target_detections_returns_matching_class():
    t = make_tracker(target_classes={'person'}, near_miss_threshold=0.30)
    obj = make_tracked_obj(class_name='person', confidence=0.85)
    result = t._target_detections([obj])
    assert obj in result


def test_target_detections_filters_wrong_class():
    t = make_tracker(target_classes={'person'}, near_miss_threshold=0.30)
    obj = make_tracked_obj(class_name='car', confidence=0.90)
    assert t._target_detections([obj]) == []


def test_target_detections_filters_low_confidence():
    t = make_tracker(target_classes={'person'}, near_miss_threshold=0.30)
    obj = make_tracked_obj(class_name='person', confidence=0.20)
    assert t._target_detections([obj]) == []


def test_target_detections_keeps_at_threshold():
    t = make_tracker(target_classes={'person'}, near_miss_threshold=0.30)
    obj = make_tracked_obj(class_name='person', confidence=0.30)
    assert obj in t._target_detections([obj])


def test_target_detections_empty_when_no_target_classes():
    t = make_tracker(target_classes=set())
    obj = make_tracked_obj(class_name='person', confidence=0.95)
    assert t._target_detections([obj]) == []


def test_target_detections_skips_dead_tracks():
    t = make_tracker(target_classes={'person'}, near_miss_threshold=0.30)
    obj = make_tracked_obj(class_name='person', confidence=0.90, live=False)
    assert t._target_detections([obj]) == []


def test_target_detections_multiple_classes():
    t = make_tracker(target_classes={'person', 'car'}, near_miss_threshold=0.30)
    person = make_tracked_obj(obj_id=1, class_name='person', confidence=0.80)
    car    = make_tracked_obj(obj_id=2, class_name='car',    confidence=0.70)
    dog    = make_tracked_obj(obj_id=3, class_name='dog',    confidence=0.90)
    result = t._target_detections([person, car, dog])
    assert person in result
    assert car in result
    assert dog not in result


# ---------------------------------------------------------------------------
# check_click_on_object
# ---------------------------------------------------------------------------

def test_click_selects_object():
    t = make_tracker(track_all=False)
    obj = make_tracked_obj(obj_id=3, bbox=[10.0, 10.0, 100.0, 100.0])
    t.clicking_pos = (50, 50)  # inside bbox
    t.check_click_on_object([obj])
    assert 3 in t.selected_track_ids


def test_click_deselects_already_selected():
    t = make_tracker(track_all=False)
    t.selected_track_ids = {3}
    obj = make_tracked_obj(obj_id=3, bbox=[10.0, 10.0, 100.0, 100.0])
    t.clicking_pos = (50, 50)
    t.check_click_on_object([obj])
    assert 3 not in t.selected_track_ids


def test_click_outside_bbox_does_not_select():
    t = make_tracker(track_all=False)
    obj = make_tracked_obj(obj_id=3, bbox=[10.0, 10.0, 100.0, 100.0])
    t.clicking_pos = (500, 500)  # outside bbox
    t.check_click_on_object([obj])
    assert 3 not in t.selected_track_ids


def test_no_clicking_pos_does_nothing():
    t = make_tracker(track_all=False)
    obj = make_tracked_obj(obj_id=3, bbox=[10.0, 10.0, 100.0, 100.0])
    t.clicking_pos = None
    t.check_click_on_object([obj])
    assert t.selected_track_ids == set()


def test_click_clears_clicking_pos_after_use():
    t = make_tracker(track_all=False)
    obj = make_tracked_obj(obj_id=1, bbox=[0.0, 0.0, 200.0, 200.0])
    t.clicking_pos = (100, 100)
    t.check_click_on_object([obj])
    assert t.clicking_pos is None


# ---------------------------------------------------------------------------
# _update_clip — clip-relative frame IDs and capture timestamps
# ---------------------------------------------------------------------------

import io
import klv


def make_recording_tracker(start_frame=100, preroll_frames=40, frame_index=105):
    """Tracker with an active clip whose klv_file is an in-memory buffer."""
    t = make_tracker(
        session_id='sess-test',
        frame_index=frame_index,
        object_counts={},
        postroll_frames=40,
    )
    t.active_clip = {
        'clip_id':                'clip_0001',
        'writer':                 MagicMock(),
        'klv_file':               io.BytesIO(),
        'class_name':             'person',
        'peak_confidence':        0.0,
        'frames_since_detection': 0,
        'detections':             [],
        'start_frame':            start_frame,
        'preroll_frames':         preroll_frames,
    }
    return t


def test_klv_frame_id_is_clip_relative():
    # session frame 105, clip triggered at 100 with 40 pre-roll frames
    # → this frame is video frame 45 of the clip file
    t = make_recording_tracker(start_frame=100, preroll_frames=40, frame_index=105)
    t._update_clip(MagicMock(), [make_tracked_obj()], '2026-07-04T12:00:00')
    packet, _ = klv.decode_packet(t.active_clip['klv_file'].getvalue())
    assert packet['frame_id'] == 45


def test_klv_frame_id_on_trigger_frame_equals_preroll_count():
    t = make_recording_tracker(start_frame=100, preroll_frames=40, frame_index=100)
    t._update_clip(MagicMock(), [make_tracked_obj()], '2026-07-04T12:00:00')
    packet, _ = klv.decode_packet(t.active_clip['klv_file'].getvalue())
    assert packet['frame_id'] == 40


def test_klv_timestamp_is_passed_frame_ts():
    t = make_recording_tracker()
    t._update_clip(MagicMock(), [make_tracked_obj()], '2026-07-04T09:30:00.123456')
    packet, _ = klv.decode_packet(t.active_clip['klv_file'].getvalue())
    assert packet['timestamp'] == '2026-07-04T09:30:00.123456'


def test_sidecar_detection_has_video_and_session_frame():
    t = make_recording_tracker(start_frame=100, preroll_frames=40, frame_index=105)
    t._update_clip(MagicMock(), [make_tracked_obj()], '2026-07-04T12:00:00')
    det = t.active_clip['detections'][0]
    assert det['frame_id'] == 45          # matches KLV / video file
    assert det['session_frame'] == 105    # session-wide, local debugging only
    assert det['timestamp'] == '2026-07-04T12:00:00'


def test_start_clip_records_preroll_count(tmp_path):
    from collections import deque
    t = make_tracker(
        clip_count=0,
        session_dir=str(tmp_path),
        frame_index=10,
        record_fps=20,
        width=640,
        height=480,
        preroll=deque(['f1', 'f2', 'f3']),
    )
    with patch('tracker.cv2.VideoWriter') as writer_cls:
        t._start_clip('person')
    assert t.active_clip['preroll_frames'] == 3
    assert t.active_clip['start_frame'] == 10
    assert writer_cls.return_value.write.call_count == 3
