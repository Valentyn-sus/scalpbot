from enum import Enum
from typing import Any, Literal
import uuid
from pydantic import BaseModel, Field, HttpUrl, field_validator


class AuthType(str, Enum):
    EMAIL = "email"
    GMAIL = "gmail"


class ConfigOverrides(BaseModel):
    target_deposit_amount: int | None = Field(default=None, gt=0)
    max_execution_time_sec: int | None = Field(default=None, gt=0)


class JobRequest(BaseModel):
    domain: str = Field(..., min_length=1)
    source: str | None = None
    priority: Literal["low", "normal", "high"] = "normal"
    auth: AuthType
    config_overrides: ConfigOverrides | None = None
    webhook_url: HttpUrl | None = None

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        clean_domain = v.strip().lower()
        if not clean_domain or " " in clean_domain:
            raise ValueError("Некоректний формат domain")
        return clean_domain


class JobResponse(BaseModel):
    task_id: uuid.UUID
    domain: str
    status: str = "ACCEPTED"
    message: str = (
        "Задача прийнята в роботу. Виконання розпочато у фоновому режимі."
    )


class WebhookPayload(BaseModel):
    task_id: uuid.UUID
    domain: str
    status: str
    result: dict[str, Any] | None = None
    timestamp: str