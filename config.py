from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Crawler Pipeline"
    API_V1_STR: str = "/api/v1"

    # Database
    DB_DSN: str = "postgresql://postgres:postgres@localhost:5432/crawler_db"

    # Crawler defaults
    TARGET_DEPOSIT_AMOUNT: int = 500
    MAX_EXECUTION_TIME_SEC: int = 180
    MAX_CONCURRENT_TASKS: int = 5
    PROXY_SERVER_URL: str | None = None
    CATCHALL_EMAIL_DOMAIN: str = "verify.com"

    # Metrics
    METRICS_PORT: int = 9090

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()