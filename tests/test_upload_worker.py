"""
Tests for upload_worker.py — verifies the worker drains the queue correctly.
UploadQueue uses a real SQLite DB (tmp_path); S3Uploader is mocked.
"""
import pytest
from unittest.mock import MagicMock
from upload_queue import UploadQueue
from upload_worker import UploadWorker


@pytest.fixture
def queue(tmp_path):
    return UploadQueue(str(tmp_path / 'upload_queue.db'))


@pytest.fixture
def uploader():
    m = MagicMock()
    m.upload_clip.return_value = ('s3://bucket/clip.ts', 's3://bucket/clip.klv')
    m.upload_summary.return_value = 's3://bucket/summary.json'
    return m


def make_worker(queue, uploader):
    return UploadWorker(queue, uploader)


CLIP = dict(session_id='sess-1', clip_id='clip_0001', tier='hits',
            ts_path='/tmp/clip.ts', klv_path='/tmp/clip.klv')
SUMMARY = dict(session_id='sess-1', local_path='/tmp/session_summary.json')


# ---------------------------------------------------------------------------
# Single-cycle processing (_process_pending directly, no thread)
# ---------------------------------------------------------------------------

def test_worker_uploads_clip(queue, uploader):
    queue.enqueue_clip(**CLIP)
    worker = make_worker(queue, uploader)
    worker._process_pending()
    uploader.upload_clip.assert_called_once_with(
        '/tmp/clip.ts', '/tmp/clip.klv', 'sess-1', 'clip_0001', 'hits'
    )


def test_worker_uploads_summary(queue, uploader):
    queue.enqueue_summary(**SUMMARY)
    worker = make_worker(queue, uploader)
    worker._process_pending()
    uploader.upload_summary.assert_called_once_with('/tmp/session_summary.json', 'sess-1')


def test_worker_marks_done_on_success(queue, uploader):
    queue.enqueue_clip(**CLIP)
    worker = make_worker(queue, uploader)
    worker._process_pending()
    assert queue.pending_count() == 0


def test_worker_marks_failed_on_upload_error(queue, uploader):
    uploader.upload_clip.side_effect = RuntimeError('timeout')
    queue.enqueue_clip(**CLIP)
    worker = make_worker(queue, uploader)
    worker._process_pending()
    # Still pending (will retry), but attempt count incremented
    assert queue.pending_count() == 1
    with queue._conn() as conn:
        row = conn.execute('SELECT attempts FROM uploads').fetchone()
    assert row['attempts'] == 1


def test_worker_uploads_multiple_items(queue, uploader):
    queue.enqueue_clip(**CLIP)
    queue.enqueue_clip(**{**CLIP, 'clip_id': 'clip_0002'})
    queue.enqueue_summary(**SUMMARY)
    worker = make_worker(queue, uploader)
    worker._process_pending()
    assert queue.pending_count() == 0
    assert uploader.upload_clip.call_count == 2
    assert uploader.upload_summary.call_count == 1


def test_worker_partial_failure_leaves_failed_pending(queue, uploader):
    uploader.upload_clip.side_effect = [
        RuntimeError('err'),                                      # first clip fails
        ('s3://b/clip.ts', 's3://b/clip.klv'),                   # second clip succeeds
    ]
    queue.enqueue_clip(**CLIP)
    queue.enqueue_clip(**{**CLIP, 'clip_id': 'clip_0002'})
    worker = make_worker(queue, uploader)
    worker._process_pending()
    assert queue.pending_count() == 1  # first clip still pending


# ---------------------------------------------------------------------------
# stop() / shutdown drain
# ---------------------------------------------------------------------------

def test_stop_drains_queue(queue, uploader):
    queue.enqueue_clip(**CLIP)
    worker = make_worker(queue, uploader)
    worker.start()
    worker.stop(timeout=10)
    assert queue.pending_count() == 0


def test_stop_is_idempotent(queue, uploader):
    worker = make_worker(queue, uploader)
    worker.start()
    worker.stop(timeout=5)
    # Should not raise even if called on an already-stopped worker
    worker.stop(timeout=1)


def test_empty_queue_stop_does_not_hang(queue, uploader):
    worker = make_worker(queue, uploader)
    worker.start()
    worker.stop(timeout=5)  # nothing to drain, should return quickly
