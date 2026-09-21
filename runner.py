from typing import Any
from app.crawler.test_runner import run_crawler_flow


class PreAuthRunner:
    def __init__(
        self,
        task_id,
        domain: str,
        auth_method: str,
        target_amount: float = 1000.0,
        proxy_url: str | None = None,
    ):
        self.task_id = task_id
        self.domain = domain
        self.auth_method = auth_method
        self.target_amount = target_amount
        self.proxy_url = proxy_url

    async def run(self) -> dict[str, Any]:
        return await run_crawler_flow(
            task_id=self.task_id,
            domain=self.domain,
            auth_method=self.auth_method,
            target_amount=self.target_amount,
            proxy_url=self.proxy_url,
        )