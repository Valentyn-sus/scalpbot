import uuid
from fastapi import APIRouter, BackgroundTasks, status
from app.api.v1.schemas import JobRequest, JobResponse
from app.db.repository import TaskRepository
from app.services.orchestrator import run_pipeline

router = APIRouter()

@router.post(
    "/crawler/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_crawler_job(
    payload: JobRequest, background_tasks: BackgroundTasks
):
    task_id = uuid.uuid4()

    await TaskRepository.create_task(
        task_id=task_id,
        domain=payload.domain,
        source=payload.source,
        status="ACCEPTED",
    )

    await TaskRepository.add_history_entry(
        task_id=task_id,
        step_name="JOB_ACCEPTED",
        step_status="SUCCESS",
        details={"auth_method": payload.auth.value},
    )

    # Запуск фонового виконання пайплайну
    background_tasks.add_task(run_pipeline, task_id, payload)

    return JobResponse(task_id=task_id, domain=payload.domain, status="ACCEPTED")


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthcheck():
    return {"status": "ok"}