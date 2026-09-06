"""S3-compatible storage service for Neon object storage."""

import logging
from io import BytesIO
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)


class S3StorageService:
    """Handles image storage and retrieval with Neon S3-compatible storage."""

    def __init__(self):
        self.enabled = settings.S3_ENABLED and all(
            [
                settings.AWS_ENDPOINT_URL_S3,
                settings.AWS_ACCESS_KEY_ID,
                settings.AWS_SECRET_ACCESS_KEY,
            ]
        )
        if self.enabled:
            self.client = boto3.client(
                "s3",
                endpoint_url=settings.AWS_ENDPOINT_URL_S3,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
            )
            self.bucket = settings.AWS_BUCKET_NAME
        else:
            self.client = None
            self.bucket = None

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
        except ClientError as e:
            logger.error(f"Failed to upload image to S3: {e}")
            return None

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
        except ClientError as e:
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
        except ClientError as e:
            logger.error(f"Failed to delete image from S3: {e}")
            return False


# Global instance
storage = S3StorageService()
