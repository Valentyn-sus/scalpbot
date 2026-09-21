import json
import uuid
from typing import Any
from asyncpg import Record
from app.db.connection import get_pool


class TaskRepository:

    @staticmethod
    async def create_task(
        task_id: uuid.UUID,
        domain: str,
        source: str | None = None,
        status: str = "PENDING",
    ) -> None:
        pool = get_pool()
        query = """
            INSERT INTO crawler_tasks (task_id, domain, source, status)
            VALUES ($1, $2, $3, $4)
        """
        await pool.execute(query, task_id, domain, source, status)

    @staticmethod
    async def add_history_entry(
        task_id: uuid.UUID,
        step_name: str,
        step_status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        pool = get_pool()
        query = """
            INSERT INTO crawler_task_history (task_id, step_name, step_status, details)
            VALUES ($1, $2, $3, $4::jsonb)
        """
        details_json = json.dumps(details) if details else None
        await pool.execute(query, task_id, step_name, step_status, details_json)

    @staticmethod
    async def update_task_status(
        task_id: uuid.UUID, status: str, final_result: dict[str, Any] | None = None
    ) -> None:
        pool = get_pool()
        query = """
            UPDATE crawler_tasks
            SET status = $1,
                final_result = $2::jsonb,
                updated_at = NOW()
            WHERE task_id = $3
        """
        result_json = json.dumps(final_result) if final_result else None
        await pool.execute(query, status, result_json, task_id)

    @staticmethod
    async def get_credentials(
        domain_cluster: str, auth_method: str
    ) -> Record | None:
        pool = get_pool()
        query = """
            SELECT domain_cluster, auth_method, email, password, session_cookies
            FROM bot_credentials
            WHERE domain_cluster = $1 AND auth_method = $2
        """
        return await pool.fetchrow(query, domain_cluster, auth_method)

    @staticmethod
    async def upsert_credentials(
        domain_cluster: str,
        auth_method: str,
        email: str,
        password: str,
        session_cookies: list[dict[str, Any]] | None = None,
    ) -> None:
        pool = get_pool()
        query = """
            INSERT INTO bot_credentials (domain_cluster, auth_method, email, password, session_cookies)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            ON CONFLICT (domain_cluster) DO UPDATE
            SET auth_method = EXCLUDED.auth_method,
                email = EXCLUDED.email,
                password = EXCLUDED.password,
                session_cookies = EXCLUDED.session_cookies,
                created_at = NOW()
        """
        cookies_json = json.dumps(session_cookies) if session_cookies else None
        await pool.execute(
            query, domain_cluster, auth_method, email, password, cookies_json
        )