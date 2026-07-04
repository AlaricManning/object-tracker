"""
Tests for upload_queue.py — SQLite queue behaviour without any network calls.
"""
import pytest
from datetime import datetime
from upload_queue import UploadQueue, MAX_ATTEMPTS


@pytest.fixture
def queue(tmp_path):
    return UploadQueue(str(tmp_path / 'upload_queue.db'))


CLIP = dict(session_id='sess-1', clip_id='clip_0001', tier='hits',
            ts_path='/tmp/clip.ts', klv_path='/tmp/clip.klv')
SUMMARY = dict(session_id='sess-1', local_path='/tmp/session_summary.json')


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

def test_enqueue_clip_creates_pending(queue):
    queue.enqueue_clip(**CLIP)
    assert queue.pending_count() == 1


def test_enqueue_summary_creates_pending(queue):
    queue.enqueue_summary(**SUMMARY)
    assert queue.pending_count() == 1


def test_enqueue_multiple(queue):
    queue.enqueue_clip(**CLIP)
    queue.enqueue_clip(**{**CLIP, 'clip_id': 'clip_0002'})
    queue.enqueue_summary(**SUMMARY)
    assert queue.pending_count() == 3


# ---------------------------------------------------------------------------
# get_pending
# ---------------------------------------------------------------------------

def test_get_pending_returns_new_items(queue):
    queue.enqueue_clip(**CLIP)
    items = queue.get_pending()
    assert len(items) == 1
    assert items[0]['clip_id'] == 'clip_0001'
    assert items[0]['status'] == 'pending'


def test_get_pending_returns_clip_fields(queue):
    queue.enqueue_clip(**CLIP)
    item = queue.get_pending()[0]
    assert item['type']       == 'clip'
    assert item['session_id'] == 'sess-1'
    assert item['tier']       == 'hits'
    assert item['ts_path']    == '/tmp/clip.ts'
    assert item['klv_path']   == '/tmp/clip.klv'


def test_get_pending_returns_summary_fields(queue):
    queue.enqueue_summary(**SUMMARY)
    item = queue.get_pending()[0]
    assert item['type']       == 'summary'
    assert item['local_path'] == '/tmp/session_summary.json'


def test_get_pending_respects_limit(queue):
    for i in range(5):
        queue.enqueue_clip(**{**CLIP, 'clip_id': f'clip_{i:04d}'})
    assert len(queue.get_pending(limit=3)) == 3


def test_get_pending_excludes_done(queue):
    queue.enqueue_clip(**CLIP)
    item = queue.get_pending()[0]
    queue.mark_done(item['id'])
    assert queue.get_pending() == []


def test_get_pending_excludes_max_attempts(queue):
    queue.enqueue_clip(**CLIP)
    item = queue.get_pending()[0]
    for _ in range(MAX_ATTEMPTS):
        queue.mark_failed(item['id'], 'err')
    assert queue.get_pending() == []
    assert queue.pending_count() == 0


# ---------------------------------------------------------------------------
# mark_done
# ---------------------------------------------------------------------------

def test_mark_done_removes_from_pending(queue):
    queue.enqueue_clip(**CLIP)
    item = queue.get_pending()[0]
    queue.mark_done(item['id'])
    assert queue.pending_count() == 0


def test_mark_done_does_not_affect_other_items(queue):
    queue.enqueue_clip(**CLIP)
    queue.enqueue_clip(**{**CLIP, 'clip_id': 'clip_0002'})
    items = queue.get_pending()
    queue.mark_done(items[0]['id'])
    assert queue.pending_count() == 1


# ---------------------------------------------------------------------------
# mark_failed + backoff
# ---------------------------------------------------------------------------

def test_mark_failed_keeps_item_pending(queue):
    queue.enqueue_clip(**CLIP)
    item = queue.get_pending()[0]
    queue.mark_failed(item['id'], 'network error')
    assert queue.pending_count() == 1


def test_mark_failed_sets_next_retry_in_future(queue):
    queue.enqueue_clip(**CLIP)
    item = queue.get_pending()[0]
    before = datetime.now()
    queue.mark_failed(item['id'], 'err')
    with queue._conn() as conn:
        row = conn.execute(
            'SELECT next_retry_at FROM uploads WHERE id = ?', (item['id'],)
        ).fetchone()
    assert datetime.fromisoformat(row['next_retry_at']) > before


def test_mark_failed_hides_item_until_backoff_expires(queue):
    queue.enqueue_clip(**CLIP)
    item = queue.get_pending()[0]
    queue.mark_failed(item['id'], 'err')
    # Immediately after failure the item should not be returned
    assert queue.get_pending() == []


def test_backoff_increases_with_attempts(queue):
    queue.enqueue_clip(**CLIP)
    item_id = queue.get_pending()[0]['id']

    retry_times = []
    for _ in range(3):
        queue.mark_failed(item_id, 'err')
        with queue._conn() as conn:
            row = conn.execute(
                'SELECT next_retry_at FROM uploads WHERE id = ?', (item_id,)
            ).fetchone()
        retry_times.append(datetime.fromisoformat(row['next_retry_at']))

    assert retry_times[1] > retry_times[0]
    assert retry_times[2] > retry_times[1]


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------

def test_persists_across_instances(tmp_path):
    db = str(tmp_path / 'upload_queue.db')
    q1 = UploadQueue(db)
    q1.enqueue_clip(**CLIP)

    q2 = UploadQueue(db)
    assert q2.pending_count() == 1
    assert q2.get_pending()[0]['clip_id'] == 'clip_0001'


def test_done_persists_across_instances(tmp_path):
    db = str(tmp_path / 'upload_queue.db')
    q1 = UploadQueue(db)
    q1.enqueue_clip(**CLIP)
    item = q1.get_pending()[0]
    q1.mark_done(item['id'])

    q2 = UploadQueue(db)
    assert q2.pending_count() == 0
