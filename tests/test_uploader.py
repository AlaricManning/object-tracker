"""
Tests for uploader.py — S3 key construction and error wrapping.
boto3 is mocked throughout; no real AWS calls are made.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from botocore.exceptions import BotoCoreError, ClientError

from uploader import S3Uploader


def make_uploader(bucket='test-bucket', prefix='raw', region='us-east-1'):
    with patch('boto3.client'):
        return S3Uploader(bucket=bucket, prefix=prefix, region=region)


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------

def test_key_with_prefix():
    u = make_uploader(prefix='raw')
    assert u._key('session-1', 'hits', 'clip_0001.ts') == 'raw/session-1/hits/clip_0001.ts'


def test_key_without_prefix():
    u = make_uploader(prefix='')
    assert u._key('session-1', 'hits', 'clip_0001.ts') == 'session-1/hits/clip_0001.ts'


def test_key_strips_trailing_slash_from_prefix():
    u = make_uploader(prefix='raw/')
    assert u._key('session-1', 'file.ts') == 'raw/session-1/file.ts'


def test_key_single_part():
    u = make_uploader(prefix='raw')
    assert u._key('file.json') == 'raw/file.json'


# ---------------------------------------------------------------------------
# upload_clip — success path
# ---------------------------------------------------------------------------

def test_upload_clip_calls_s3_twice():
    u = make_uploader()
    u.client = MagicMock()
    u.upload_clip('/tmp/clip.ts', '/tmp/clip.klv', 'sess-1', 'clip_0001', 'hits')
    assert u.client.upload_file.call_count == 2


def test_upload_clip_ts_key():
    u = make_uploader(prefix='raw')
    u.client = MagicMock()
    u.upload_clip('/tmp/clip.ts', '/tmp/clip.klv', 'sess-1', 'clip_0001', 'hits')
    ts_key = u.client.upload_file.call_args_list[0][0][2]
    assert ts_key == 'raw/sess-1/hits/clip_0001.ts'


def test_upload_clip_klv_key():
    u = make_uploader(prefix='raw')
    u.client = MagicMock()
    u.upload_clip('/tmp/clip.ts', '/tmp/clip.klv', 'sess-1', 'clip_0001', 'hits')
    klv_key = u.client.upload_file.call_args_list[1][0][2]
    assert klv_key == 'raw/sess-1/hits/clip_0001.klv'


def test_upload_clip_returns_s3_uris():
    u = make_uploader(bucket='my-bucket', prefix='raw')
    u.client = MagicMock()
    ts_uri, klv_uri = u.upload_clip('/tmp/clip.ts', '/tmp/clip.klv', 'sess-1', 'clip_0001', 'hits')
    assert ts_uri == 's3://my-bucket/raw/sess-1/hits/clip_0001.ts'
    assert klv_uri == 's3://my-bucket/raw/sess-1/hits/clip_0001.klv'


def test_upload_clip_near_misses_tier():
    u = make_uploader(prefix='raw')
    u.client = MagicMock()
    u.upload_clip('/tmp/clip.ts', '/tmp/clip.klv', 'sess-1', 'clip_0002', 'near_misses')
    ts_key = u.client.upload_file.call_args_list[0][0][2]
    assert 'near_misses' in ts_key


# ---------------------------------------------------------------------------
# upload_clip — error wrapping
# ---------------------------------------------------------------------------

def test_upload_clip_wraps_botocore_error():
    u = make_uploader()
    u.client = MagicMock()
    u.client.upload_file.side_effect = BotoCoreError()
    with pytest.raises(RuntimeError, match="S3 upload failed"):
        u.upload_clip('/tmp/clip.ts', '/tmp/clip.klv', 'sess-1', 'clip_0001', 'hits')


def test_upload_clip_wraps_client_error():
    u = make_uploader()
    u.client = MagicMock()
    u.client.upload_file.side_effect = ClientError({'Error': {'Code': '403', 'Message': 'Forbidden'}}, 'PutObject')
    with pytest.raises(RuntimeError, match="S3 upload failed"):
        u.upload_clip('/tmp/clip.ts', '/tmp/clip.klv', 'sess-1', 'clip_0001', 'hits')


def test_upload_clip_error_includes_clip_id():
    u = make_uploader()
    u.client = MagicMock()
    u.client.upload_file.side_effect = BotoCoreError()
    with pytest.raises(RuntimeError, match="clip_0007"):
        u.upload_clip('/tmp/clip.ts', '/tmp/clip.klv', 'sess-1', 'clip_0007', 'hits')


# ---------------------------------------------------------------------------
# upload_summary
# ---------------------------------------------------------------------------

def test_upload_summary_key():
    u = make_uploader(prefix='raw')
    u.client = MagicMock()
    u.upload_summary('/tmp/session_summary.json', 'sess-1')
    key = u.client.upload_file.call_args[0][2]
    assert key == 'raw/sess-1/session_summary.json'


def test_upload_summary_returns_uri():
    u = make_uploader(bucket='my-bucket', prefix='raw')
    u.client = MagicMock()
    uri = u.upload_summary('/tmp/session_summary.json', 'sess-1')
    assert uri == 's3://my-bucket/raw/sess-1/session_summary.json'


def test_upload_summary_wraps_error():
    u = make_uploader()
    u.client = MagicMock()
    u.client.upload_file.side_effect = BotoCoreError()
    with pytest.raises(RuntimeError, match="summary upload failed"):
        u.upload_summary('/tmp/session_summary.json', 'sess-1')
