"""
S3 uploader for completed clips.
Credentials via ~/.aws/credentials or environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""
import os
import boto3
import boto3.exceptions
from botocore.exceptions import BotoCoreError, ClientError

_BOTO_ERRORS = (BotoCoreError, ClientError, boto3.exceptions.S3UploadFailedError)


class S3Uploader:
    def __init__(self, bucket: str, prefix: str = '', region: str = None):
        self.bucket = bucket
        self.prefix = prefix.rstrip('/')
        self.client = boto3.client('s3', region_name=region)

    def _key(self, *parts: str) -> str:
        segments = [self.prefix] + list(parts) if self.prefix else list(parts)
        return '/'.join(segments)

    def upload_file(self, local_path: str, s3_key: str) -> str:
        """Upload a single file and return its s3:// URI."""
        self.client.upload_file(local_path, self.bucket, s3_key)
        return f"s3://{self.bucket}/{s3_key}"

    def upload_clip(
        self,
        ts_path: str,
        klv_path: str,
        session_id: str,
        clip_id: str,
        tier: str,
    ) -> tuple[str, str]:
        """
        Upload a .ts video and its .klv sidecar together.
        Returns (ts_uri, klv_uri).
        Raises on any upload failure — caller decides whether to retry or skip.
        """
        ts_key  = self._key(session_id, tier, f'{clip_id}.ts')
        klv_key = self._key(session_id, tier, f'{clip_id}.klv')

        try:
            ts_uri  = self.upload_file(ts_path,  ts_key)
            klv_uri = self.upload_file(klv_path, klv_key)
        except _BOTO_ERRORS as e:
            raise RuntimeError(f"S3 upload failed for {clip_id}: {e}") from e

        return ts_uri, klv_uri

    def upload_summary(self, local_path: str, session_id: str) -> str:
        """Upload session_summary.json and return its s3:// URI."""
        key = self._key(session_id, 'session_summary.json')
        try:
            return self.upload_file(local_path, key)
        except _BOTO_ERRORS as e:
            raise RuntimeError(f"S3 summary upload failed: {e}") from e
