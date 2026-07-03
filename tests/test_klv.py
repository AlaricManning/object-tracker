"""
Tests for klv.py — KLV encode/decode round-trips and error handling.
All tests use pure in-memory bytes; no filesystem or network required.
"""
import struct
import pytest
import klv

SAMPLE = dict(
    frame_id=42,
    timestamp='2026-07-03T10:00:00.123456',
    session_id='2026-07-03T10-00-00',
    clip_id='clip_0001',
    track_id=7,
    class_name='person',
    confidence=0.95,
    bbox=[100.0, 200.0, 300.0, 400.0],
)


def encode(**overrides):
    return klv.encode_packet(**{**SAMPLE, **overrides})


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

def test_encode_returns_bytes():
    assert isinstance(encode(), bytes)


def test_length_prefix_matches_payload():
    data = encode()
    declared_len = struct.unpack_from('>H', data, 0)[0]
    assert declared_len == len(data) - 2


def test_packet_is_non_empty():
    assert len(encode()) > 2


# ---------------------------------------------------------------------------
# Round-trip: strings
# ---------------------------------------------------------------------------

def test_round_trip_session_id():
    fields, _ = klv.decode_packet(encode())
    assert fields['session_id'] == SAMPLE['session_id']


def test_round_trip_clip_id():
    fields, _ = klv.decode_packet(encode())
    assert fields['clip_id'] == SAMPLE['clip_id']


def test_round_trip_timestamp():
    fields, _ = klv.decode_packet(encode())
    assert fields['timestamp'] == SAMPLE['timestamp']


def test_round_trip_class_name():
    fields, _ = klv.decode_packet(encode())
    assert fields['class_name'] == SAMPLE['class_name']


# ---------------------------------------------------------------------------
# Round-trip: integers
# ---------------------------------------------------------------------------

def test_round_trip_frame_id():
    fields, _ = klv.decode_packet(encode())
    assert fields['frame_id'] == SAMPLE['frame_id']


def test_round_trip_track_id():
    fields, _ = klv.decode_packet(encode())
    assert fields['track_id'] == SAMPLE['track_id']


def test_round_trip_frame_id_zero():
    fields, _ = klv.decode_packet(encode(frame_id=0))
    assert fields['frame_id'] == 0


def test_round_trip_large_frame_id():
    fields, _ = klv.decode_packet(encode(frame_id=99999))
    assert fields['frame_id'] == 99999


# ---------------------------------------------------------------------------
# Round-trip: floats (float32 has ~7 significant digits)
# ---------------------------------------------------------------------------

def test_round_trip_confidence():
    fields, _ = klv.decode_packet(encode(confidence=0.85))
    assert abs(fields['confidence'] - 0.85) < 0.0001


def test_round_trip_confidence_zero():
    fields, _ = klv.decode_packet(encode(confidence=0.0))
    assert fields['confidence'] == pytest.approx(0.0, abs=1e-4)


def test_round_trip_confidence_one():
    fields, _ = klv.decode_packet(encode(confidence=1.0))
    assert fields['confidence'] == pytest.approx(1.0, abs=1e-4)


def test_round_trip_bbox():
    bbox = [10.5, 20.25, 300.75, 400.0]
    fields, _ = klv.decode_packet(encode(bbox=bbox))
    assert fields['x1'] == pytest.approx(bbox[0], abs=0.01)
    assert fields['y1'] == pytest.approx(bbox[1], abs=0.01)
    assert fields['x2'] == pytest.approx(bbox[2], abs=0.01)
    assert fields['y2'] == pytest.approx(bbox[3], abs=0.01)


def test_round_trip_bbox_zeros():
    fields, _ = klv.decode_packet(encode(bbox=[0.0, 0.0, 0.0, 0.0]))
    assert fields['x1'] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Multiple packets via iter_packets
# ---------------------------------------------------------------------------

def test_iter_packets_single():
    data = encode()
    packets = list(klv.iter_packets(data))
    assert len(packets) == 1
    assert packets[0]['class_name'] == 'person'


def test_iter_packets_three():
    data = encode(track_id=1) + encode(track_id=2) + encode(track_id=3)
    packets = list(klv.iter_packets(data))
    assert len(packets) == 3
    assert [p['track_id'] for p in packets] == [1, 2, 3]


def test_iter_packets_different_classes():
    data = encode(class_name='person') + encode(class_name='car')
    packets = list(klv.iter_packets(data))
    assert packets[0]['class_name'] == 'person'
    assert packets[1]['class_name'] == 'car'


def test_iter_packets_all_fields_present():
    packets = list(klv.iter_packets(encode()))
    p = packets[0]
    for key in ('session_id', 'clip_id', 'frame_id', 'timestamp',
                'track_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'):
        assert key in p, f"missing field: {key}"


# ---------------------------------------------------------------------------
# decode_packet offset behaviour
# ---------------------------------------------------------------------------

def test_decode_packet_returns_correct_next_offset():
    data = encode()
    _, next_offset = klv.decode_packet(data, 0)
    assert next_offset == len(data)


def test_decode_packet_with_leading_offset():
    prefix = b'\x00' * 8
    data = prefix + encode()
    fields, next_offset = klv.decode_packet(data, 8)
    assert fields['class_name'] == 'person'
    assert next_offset == len(data)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_value_too_long_raises():
    with pytest.raises(ValueError, match="too long"):
        klv.encode_packet(**{**SAMPLE, 'class_name': 'x' * 256})


def test_truncated_header_raises():
    with pytest.raises(ValueError, match="header"):
        klv.decode_packet(b'\x00')


def test_empty_bytes_raises():
    with pytest.raises(ValueError):
        klv.decode_packet(b'')


def test_truncated_payload_raises():
    data = encode()
    with pytest.raises(ValueError, match="payload"):
        klv.decode_packet(data[:4])
