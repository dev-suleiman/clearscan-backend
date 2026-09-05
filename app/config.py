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

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]


settings = Settings()
