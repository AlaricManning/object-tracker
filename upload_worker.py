"""
Background thread that drains the UploadQueue, calling S3Uploader for each item.
Runs as a non-daemon thread so a clean shutdown can wait for in-flight uploads.
"""
import threading

from upload_queue import UploadQueue, MAX_ATTEMPTS

POLL_INTERVAL = 5  # seconds between queue sweeps


class UploadWorker(threading.Thread):
    def __init__(self, queue: UploadQueue, uploader):
        super().__init__(name='UploadWorker', daemon=False)
        self._queue    = queue
        self._uploader = uploader
        self._stop_event     = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            self._process_pending()
            self._stop_event.wait(POLL_INTERVAL)
        for item in self._queue.get_pending():
            self._upload(item)

    def stop(self, timeout: int = 30):
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        self.join(timeout=timeout)
        remaining = self._queue.pending_count()
        if remaining:
            print(f"[S3]  {remaining} upload(s) still pending — will retry on next run")

    def _process_pending(self):
        for item in self._queue.get_pending():
            if self._stop_event.is_set():
                break
            self._upload(item)

    def _upload(self, item: dict):
        try:
            if item['type'] == 'clip':
                ts_uri, klv_uri = self._uploader.upload_clip(
                    item['ts_path'], item['klv_path'],
                    item['session_id'], item['clip_id'], item['tier'],
                )
                print(f"[S3]  Uploaded → {ts_uri}")
                print(f"[S3]          → {klv_uri}")
            elif item['type'] == 'summary':
                uri = self._uploader.upload_summary(item['local_path'], item['session_id'])
                print(f"[S3]  Summary  → {uri}")
            self._queue.mark_done(item['id'])
        except RuntimeError as e:
            attempts = item['attempts'] + 1
            self._queue.mark_failed(item['id'], str(e))
            if attempts >= MAX_ATTEMPTS:
                print(f"[S3]  Giving up on {item.get('clip_id') or 'summary'} after {MAX_ATTEMPTS} attempts: {e}")
            else:
                print(f"[S3]  Upload failed (attempt {attempts}/{MAX_ATTEMPTS}), will retry: {e}")
