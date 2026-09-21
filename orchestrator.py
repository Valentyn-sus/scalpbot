import logging
import uuid
from app.api.v1.schemas import JobRequest
from app.crawler.runner import PreAuthRunner
from app.db.repository import TaskRepository

logger = logging.getLogger("orchestrator")


async def run_pipeline(task_id: uuid.UUID, job_request: JobRequest) -> None:
    try:
        target_amount = 1000.0
        if job_request.config_overrides and "target_deposit_amount" in job_request.config_overrides:
            target_amount = float(job_request.config_overrides["target_deposit_amount"])

        runner = PreAuthRunner(
            task_id=task_id,
            domain=job_request.domain,
            auth_method=job_request.auth.value,
            target_amount=target_amount,
        )

        final_result = await runner.run()

        await TaskRepository.update_task_status(
            task_id=task_id,
            status="SUCCESS",
            final_result=final_result,
        )
        logger.info(f"Task {task_id} successfully completed.")

    except Exception as e:
        logger.exception(f"Task {task_id} failed: {e}")
        await TaskRepository.add_history_entry(
            task_id, "PIPELINE_ERROR", "FAILED", {"error": str(e)}
        )
        await TaskRepository.update_task_status(
            task_id=task_id,
            status="FAIL_PARSING_FAILED",
            final_result={"error": str(e)},
        )