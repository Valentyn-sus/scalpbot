import asyncpg
from app.config import settings

pool: asyncpg.Pool | None = None


async def init_db() -> asyncpg.Pool:
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(dsn=settings.DB_DSN, min_size=2, max_size=10)
    return pool


async def close_db() -> None:
    global pool
    if pool:
        await pool.close()
        pool = None


def get_pool() -> asyncpg.Pool:
    if pool is None:
        raise RuntimeError("Database connection pool is not initialized")
    return pool