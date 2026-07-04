"""
Tests for upload_worker.py — verifies the worker drains the queue correctly.
UploadQueue uses a real SQLite DB (tmp_path); S3Uploader is mocked.
"""
import pytest
from unittest.mock import MagicMock, patch
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


# ---------------------------------------------------------------------------
# Transcode stage (_finish_clip)
# ---------------------------------------------------------------------------

def enqueue_transcode_clip(queue, tmp_path, ts_exists=False, mp4_exists=True):
    """Enqueue a clip whose mp4/ts live in tmp_path; returns (mp4, ts) paths."""
    mp4 = tmp_path / '_clip_0001_tmp.mp4'
    ts  = tmp_path / 'hits' / 'clip_0001.ts'
    ts.parent.mkdir(exist_ok=True)
    if mp4_exists:
        mp4.write_bytes(b'fake mp4')
    if ts_exists:
        ts.write_bytes(b'fake ts')
    queue.enqueue_clip(session_id='sess-1', clip_id='clip_0001', tier='hits',
                       ts_path=str(ts), klv_path='/tmp/clip.klv', mp4_path=str(mp4))
    return mp4, ts


def test_worker_transcodes_before_upload(queue, uploader, tmp_path):
    mp4, ts = enqueue_transcode_clip(queue, tmp_path)
    with patch('upload_worker.transcode_to_ts') as transcode:
        transcode.side_effect = lambda m, t: open(t, 'wb').write(b'ts data')
        make_worker(queue, uploader)._process_pending()
    transcode.assert_called_once_with(str(mp4), str(ts))
    uploader.upload_clip.assert_called_once()
    assert queue.pending_count() == 0


def test_worker_deletes_mp4_after_transcode(queue, uploader, tmp_path):
    mp4, ts = enqueue_transcode_clip(queue, tmp_path)
    with patch('upload_worker.transcode_to_ts') as transcode:
        transcode.side_effect = lambda m, t: open(t, 'wb').write(b'ts data')
        make_worker(queue, uploader)._process_pending()
    assert not mp4.exists()
    assert ts.exists()


def test_worker_skips_transcode_when_ts_exists(queue, uploader, tmp_path):
    # Resume case: crashed after transcode, before upload
    mp4, ts = enqueue_transcode_clip(queue, tmp_path, ts_exists=True)
    with patch('upload_worker.transcode_to_ts') as transcode:
        make_worker(queue, uploader)._process_pending()
    transcode.assert_not_called()
    assert not mp4.exists()  # leftover mp4 still cleaned up
    uploader.upload_clip.assert_called_once()


def test_worker_transcodes_without_uploader(queue, tmp_path):
    # upload disabled — clip must still be transcoded and marked done
    mp4, ts = enqueue_transcode_clip(queue, tmp_path)
    with patch('upload_worker.transcode_to_ts') as transcode:
        transcode.side_effect = lambda m, t: open(t, 'wb').write(b'ts data')
        make_worker(queue, None)._process_pending()
    transcode.assert_called_once()
    assert queue.pending_count() == 0


def test_worker_transcode_failure_retries(queue, uploader, tmp_path):
    mp4, ts = enqueue_transcode_clip(queue, tmp_path)
    with patch('upload_worker.transcode_to_ts') as transcode:
        transcode.side_effect = RuntimeError('FFmpeg transcode failed')
        make_worker(queue, uploader)._process_pending()
    uploader.upload_clip.assert_not_called()
    assert mp4.exists()          # source kept for the retry
    assert queue.pending_count() == 1


def test_worker_summary_skipped_without_uploader(queue):
    queue.enqueue_summary(**SUMMARY)
    make_worker(queue, None)._process_pending()
    assert queue.pending_count() == 0  # marked done, nothing to upload
