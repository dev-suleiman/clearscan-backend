"""S3-compatible storage service for Neon object storage."""

import logging
from io import BytesIO
from datetime import datetime

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

logger = logging.getLogger(__name__)


class ObjectStorageError(RuntimeError):
    """Raised when the configured object storage rejects an operation."""


class S3StorageService:
    """Handles image storage and retrieval with Neon S3-compatible storage."""

    def __init__(self):
        endpoint = self._clean_setting(settings.AWS_ENDPOINT_URL_S3)
        access_key = self._clean_setting(settings.AWS_ACCESS_KEY_ID)
        secret_key = self._clean_setting(settings.AWS_SECRET_ACCESS_KEY)
        region = self._clean_setting(settings.AWS_REGION)
        bucket = self._clean_setting(settings.AWS_BUCKET_NAME)
        self.enabled = settings.S3_ENABLED and all([endpoint, access_key, secret_key, bucket])
        if self.enabled:
            self.client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                ),
            )
            self.bucket = bucket
        else:
            self.client = None
            self.bucket = None

    @staticmethod
    def _clean_setting(value: str) -> str:
        return value.strip().strip('"').strip("'").strip()

    def upload_image(self, image_bytes: bytes, user_id: int, session_id: int, image_type: str = "original") -> str | None:
        """
        Upload image to S3.

        Args:
            image_bytes: Image file content
            user_id: User ID for path organization
            session_id: Scan session ID
            image_type: "original" or "enhanced"

        Returns:
            S3 key (path) of uploaded image, or None if storage disabled.
        """
        if not self.enabled:
            return None

        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            key = f"uploads/user_{user_id}/session_{session_id}_{image_type}_{timestamp}.png"

            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=image_bytes,
                ContentType="image/png",
            )
            logger.info(f"Uploaded image to S3: {key}")
            return key
        except (BotoCoreError, ClientError) as e:
            error_code = e.response.get("Error", {}).get("Code", "unknown") if isinstance(e, ClientError) else type(e).__name__
            logger.error("Failed to upload image to S3: code=%s bucket=%s", error_code, self.bucket)
            raise ObjectStorageError(f"Object storage upload failed: {error_code}") from e

    def get_signed_url(self, key: str, expiration: int = 3600) -> str | None:
        """
        Generate signed URL for image retrieval.

        Args:
            key: S3 object key
            expiration: URL expiration in seconds (default 1 hour)

        Returns:
            Signed URL, or None if storage disabled.
        """
        if not self.enabled:
            return None

        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expiration,
            )
            return url
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to generate signed URL for {key}: {e}")
            return None

    def delete_image(self, key: str) -> bool:
        """Delete image from S3."""
        if not self.enabled:
            return False

        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Deleted image from S3: {key}")
            return True
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to delete image from S3: {e}")
            return False


# Global instance
storage = S3StorageService()
