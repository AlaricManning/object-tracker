"""
Background thread that finishes clips off the frame loop: transcodes the
temp MP4 to H.264/MPEG-TS, then uploads to S3 via S3Uploader.

The uploader may be None (upload disabled) — clips are still transcoded.
Runs as a non-daemon thread so a clean shutdown can wait for in-flight work.
"""
import os
import subprocess
import threading

from upload_queue import UploadQueue, MAX_ATTEMPTS

POLL_INTERVAL = 5  # seconds between queue sweeps


def transcode_to_ts(mp4_path: str, ts_path: str):
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', mp4_path,
         '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast',
         '-f', 'mpegts', ts_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg transcode failed:\n{result.stderr.decode()}")


class UploadWorker(threading.Thread):
    def __init__(self, queue: UploadQueue, uploader=None, delete_local=False):
        super().__init__(name='UploadWorker', daemon=False)
        self._queue        = queue
        self._uploader     = uploader
        # only ever delete local files when S3 has a copy — no uploader, no delete
        self._delete_local = delete_local and uploader is not None
        self._stop_event   = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            self._process_pending()
            self._stop_event.wait(POLL_INTERVAL)
        for item in self._queue.get_pending():
            self._process_item(item)

    def stop(self, timeout: int = 30):
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        self.join(timeout=timeout)
        remaining = self._queue.pending_count()
        if remaining:
            print(f"[S3]  {remaining} item(s) still pending — will retry on next run")

    def _process_pending(self):
        for item in self._queue.get_pending():
            if self._stop_event.is_set():
                break
            self._process_item(item)

    def _process_item(self, item: dict):
        try:
            if item['type'] == 'clip':
                self._finish_clip(item)
            elif item['type'] == 'summary' and self._uploader:
                uri = self._uploader.upload_summary(item['local_path'], item['session_id'])
                print(f"[S3]  Summary  → {uri}")
            self._queue.mark_done(item['id'])
        except (RuntimeError, OSError) as e:
            attempts = item['attempts'] + 1
            self._queue.mark_failed(item['id'], str(e))
            if attempts >= MAX_ATTEMPTS:
                print(f"[S3]  Giving up on {item.get('clip_id') or 'summary'} after {MAX_ATTEMPTS} attempts: {e}")
            else:
                print(f"[S3]  Processing failed (attempt {attempts}/{MAX_ATTEMPTS}), will retry: {e}")

    def _finish_clip(self, item: dict):
        # Stage 1: transcode, unless the .ts already exists (resume after crash)
        if item.get('mp4_path') and not os.path.exists(item['ts_path']):
            transcode_to_ts(item['mp4_path'], item['ts_path'])
            print(f"[CLIP] Transcoded {item['clip_id']} → {item['ts_path']}")
        if item.get('mp4_path') and os.path.exists(item['mp4_path']):
            os.remove(item['mp4_path'])

        # Stage 2: upload (skipped when upload is disabled)
        if self._uploader:
            ts_uri, klv_uri = self._uploader.upload_clip(
                item['ts_path'], item['klv_path'],
                item['session_id'], item['clip_id'], item['tier'],
            )
            print(f"[S3]  Uploaded → {ts_uri}")
            print(f"[S3]          → {klv_uri}")
            if self._delete_local:
                for path in (item['ts_path'], item['klv_path']):
                    if os.path.exists(path):
                        os.remove(path)
