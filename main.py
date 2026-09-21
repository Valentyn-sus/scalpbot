from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1.endpoints import router as api_v1_router
from app.config import settings
from app.db.connection import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

app.include_router(api_v1_router, prefix=settings.API_V1_STR)