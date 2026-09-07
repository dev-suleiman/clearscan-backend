from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    MODEL_DIR: str = "model_files"
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list[str] = [".png", ".jpg", ".jpeg"]
    CORS_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str = "sqlite:///./clearscan.db"
    JWT_SECRET: str = "change-this-secret-in-production"
    JWT_EXPIRE_DAYS: int = 7

    # S3-compatible storage (Neon)
    AWS_ENDPOINT_URL_S3: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-2"
    AWS_BUCKET_NAME: str = "assets"
    S3_ENABLED: bool = False

    @field_validator(
        "AWS_ENDPOINT_URL_S3",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_BUCKET_NAME",
        mode="before",
    )
    @classmethod
    def clean_s3_setting(cls, value: str) -> str:
        if not isinstance(value, str):
            return value
        return value.strip().strip('"').strip("'").strip()

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]


settings = Settings()
