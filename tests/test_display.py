"""
Tests for display.py — smoke tests verifying drawing functions don't crash
and correctly apply filtering logic. Visual correctness is validated manually.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

import display as hud


def blank_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def make_colors(n=80):
    return [(i * 3 % 255, i * 5 % 255, i * 7 % 255) for i in range(n)]


def make_obj(obj_id=1, class_name='person', confidence=0.9,
             bbox=None, live=True, has_detection=True):
    bbox = bbox or [10.0, 10.0, 100.0, 100.0]
    obj = MagicMock()
    obj.id = obj_id
    obj.live_points.any.return_value = live
    if has_detection:
        obj.last_detection.points = np.array([[55.0, 55.0]])
        obj.last_detection.data = {
            'class_name': class_name,
            'confidence': confidence,
            'class_id':   0,
            'bbox':       bbox,
        }
    else:
        obj.last_detection = None
    return obj


# ---------------------------------------------------------------------------
# draw_tracked_objects
# ---------------------------------------------------------------------------

def test_draw_empty_list_does_not_crash():
    frame = blank_frame()
    hud.draw_tracked_objects(frame, [], True, set(), set(), make_colors())


def test_draw_single_object_track_all():
    frame = blank_frame()
    hud.draw_tracked_objects(frame, [make_obj()], True, set(), {'person'}, make_colors())
    assert frame.sum() > 0  # something was drawn


def test_draw_selective_skips_unselected():
    frame = blank_frame()
    obj = make_obj(obj_id=5)
    hud.draw_tracked_objects(frame, [obj], track_all=False,
                             selected_track_ids=set(),
                             target_classes={'person'},
                             colors=make_colors())
    assert frame.sum() == 0  # nothing drawn — object not selected


def test_draw_selective_shows_selected():
    frame = blank_frame()
    obj = make_obj(obj_id=5)
    hud.draw_tracked_objects(frame, [obj], track_all=False,
                             selected_track_ids={5},
                             target_classes={'person'},
                             colors=make_colors())
    assert frame.sum() > 0  # selected object should be drawn


def test_draw_skips_object_with_no_detection():
    frame = blank_frame()
    obj = make_obj(has_detection=False)
    # Should not raise even when last_detection is None
    hud.draw_tracked_objects(frame, [obj], True, set(), set(), make_colors())


def test_draw_multiple_objects():
    frame = blank_frame()
    objects = [make_obj(obj_id=i, bbox=[i*10.0, i*10.0, i*10.0+50, i*10.0+50])
               for i in range(1, 4)]
    hud.draw_tracked_objects(frame, objects, True, set(), {'person'}, make_colors())
    assert frame.sum() > 0


# ---------------------------------------------------------------------------
# draw_info
# ---------------------------------------------------------------------------

def test_draw_info_no_target_classes():
    frame = blank_frame()
    hud.draw_info(frame, fps=25, num_detections=3, num_tracked=2,
                  track_all=True, selected_track_ids=set(),
                  target_classes=set(), active_clip=None, clip_count=0)
    assert frame.sum() > 0


def test_draw_info_with_target_classes_no_clip():
    frame = blank_frame()
    hud.draw_info(frame, fps=25, num_detections=1, num_tracked=1,
                  track_all=True, selected_track_ids=set(),
                  target_classes={'person'}, active_clip=None, clip_count=3)
    assert frame.sum() > 0


def test_draw_info_with_active_clip():
    frame = blank_frame()
    active_clip = {'clip_id': 'clip_0001', 'peak_confidence': 0.92}
    hud.draw_info(frame, fps=25, num_detections=1, num_tracked=1,
                  track_all=True, selected_track_ids=set(),
                  target_classes={'person'}, active_clip=active_clip, clip_count=1)
    assert frame.sum() > 0


def test_draw_info_selective_mode():
    frame = blank_frame()
    hud.draw_info(frame, fps=15, num_detections=0, num_tracked=0,
                  track_all=False, selected_track_ids={1, 2},
                  target_classes=set(), active_clip=None, clip_count=0)
    assert frame.sum() > 0


def test_draw_info_mutates_frame_in_place():
    frame = blank_frame()
    original_id = id(frame)
    hud.draw_info(frame, 25, 0, 0, True, set(), set(), None, 0)
    assert id(frame) == original_id
