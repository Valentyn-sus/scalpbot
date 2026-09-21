import os
import json
import re
import random
import asyncio
from typing import Any
from playwright.async_api import async_playwright
from seleniumbase import sb_cdp

from app.config import settings
from app.db.repository import TaskRepository
from app.crawler.utils import (
    generate_synthetic_identity,
    capture_screenshot,
)
from app.crawler.ocr import RequisitiveParser


def launch_undetected_chrome():
    """Синхронный запуск SeleniumBase CDP с патчами против антиботов.
    Запускается в отдельном потоке (to_thread).
    """
    sb = sb_cdp.Chrome()
    return sb.get_endpoint_url()


async def human_type(page, locator, text: str, min_delay: int = 50, max_delay: int = 200) -> None:
    await locator.click()
    for char in text:
        await page.keyboard.type(char)
        await page.wait_for_timeout(random.randint(min_delay, max_delay))


async def fill_first_available(page, locators: list[str], value: str, timeout: int = 3000) -> None:
    for sel in locators:
        loc = page.locator(sel)
        try:
            await loc.wait_for(state="visible", timeout=timeout)
            await loc.fill(value)
            return
        except Exception:
            continue
    raise Exception("Ни один из локаторов не найден видимым")


async def submit_deposit(page, amount: str = "1000") -> bool:
    for frame in page.frames:
        try:
            input_field = frame.locator(
                "input[type='number'], input[type='text'], input[placeholder*='Сума'], input[placeholder*='Сумма']"
            ).first
            if await input_field.is_visible(timeout=1000):
                await input_field.click()
                await input_field.fill("")
                await page.keyboard.type(str(amount), delay=random.randint(50, 150))

                next_btn = frame.locator(
                    "button:has-text('Далі'), button:has-text('Далее'), [data-testid*='submit']"
                ).first
                await next_btn.click()
                return True
        except Exception:
            continue
    return False


async def get_card_number(page) -> str | None:
    for frame in page.frames:
        try:
            label = frame.get_by_text("Номер картки", exact=False).first
            if await label.is_visible(timeout=1000):
                parent_text = await label.locator("xpath=..").inner_text()
                card_number = re.sub(r"\D", "", parent_text)
                if len(card_number) >= 16:
                    return card_number
        except Exception:
            continue

    for frame in page.frames:
        try:
            card_el = frame.locator("text=/\\d{4}\\s?\\d{4}\\s?\\d{4}\\s?\\d{4}/").first
            if await card_el.is_visible(timeout=1000):
                text = await card_el.inner_text()
                return re.sub(r"\D", "", text)
        except Exception:
            continue

    return None


async def perform_registration(page, context, domain: str, auth_method: str, task_id) -> dict:
    locator_reg = page.get_by_role(
        "button",
        name=re.compile(r"реєстрація|registration|register|sign up|регистрация", re.IGNORECASE),
    ).first

    await locator_reg.click(force=True)
    await page.wait_for_timeout(2000)

    identity = generate_synthetic_identity(settings.CATCHALL_EMAIL_DOMAIN)

    await fill_first_available(
        page,
        [
            "#email",
            "input[name='login']",
            "[data-testid='email-field-input']",
        ],
        identity["email"],
    )

    await fill_first_available(
        page,
        [
            "#password",
            "input.TBYQA3yH5WwlOyibKUklwzzl[type='password']",
            "[data-testid='password-field-input']",
        ],
        identity["password"],
    )

    await page.wait_for_timeout(2000)
    await capture_screenshot(page, task_id, "step3_form_filled")

    submit_btn = page.get_by_role(
        "button",
        name=re.compile(r"зареєструватися|register|sign up|зарегистрироваться", re.IGNORECASE),
    ).first
    await submit_btn.click(force=True)
    await page.wait_for_timeout(10000)

    cookies = await context.cookies()
    await TaskRepository.upsert_credentials(
        domain_cluster=domain,
        auth_method=auth_method,
        email=identity["email"],
        password=identity["password"],
        session_cookies=cookies,
    )
    return identity


async def run_crawler_flow(
    task_id,
    domain: str,
    auth_method: str,
    target_amount: float = 1000.0,
    proxy_url: str | None = None,
) -> dict[str, Any]:
    target_url = domain if domain.startswith("http") else f"https://{domain}"

    saved_creds = await TaskRepository.get_credentials(domain, auth_method)
    await TaskRepository.add_history_entry(task_id, "BROWSER_LAUNCH", "STARTED")

    # 1. Запускаем непробиваемый Chrome через SeleniumBase CDP в неблокирующем потоке
    endpoint_url = await asyncio.to_thread(launch_undetected_chrome)

    async with async_playwright() as p:
        # 2. Подключаемся асинхронным Playwright через CDP
        browser = await p.chromium.connect_over_cdp(endpoint_url)
        context = await browser.new_context(ignore_https_errors=True)
        page = await context.new_page()

        await TaskRepository.add_history_entry(task_id, "BROWSER_LAUNCH", "SUCCESS")

        # --- 1. PERIMETER SCAN ---
        await TaskRepository.add_history_entry(task_id, "PERIMETER_SCAN", "STARTED")
        try:
            await page.goto(target_url, timeout=30000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            shot_path = await capture_screenshot(page, task_id, "PERIMETER_SCAN_ERROR")
            await TaskRepository.add_history_entry(
                task_id, "PERIMETER_SCAN", "FAILED", {"error": str(e), "screenshot": shot_path}
            )
            await browser.close()
            raise RuntimeError(f"Не удалось загрузить сайт: {e}")

        shot_perimeter = await capture_screenshot(page, task_id, "step1")
        await TaskRepository.add_history_entry(
            task_id, "PERIMETER_SCAN", "SUCCESS", {"screenshot": shot_perimeter}
        )

        # --- 2. AUTH OR REGISTER ---
        await TaskRepository.add_history_entry(task_id, "AUTH_OR_REGISTER", "STARTED")
        auth_action = "registration"

        cookies_valid = False
        if saved_creds and saved_creds.get("session_cookies"):
            cookies = saved_creds["session_cookies"]
            if isinstance(cookies, str):
                try:
                    cookies = json.loads(cookies)
                except Exception:
                    cookies = []

            if isinstance(cookies, list) and len(cookies) > 0:
                await context.add_cookies(cookies)
                await page.reload()
                await page.wait_for_timeout(2000)

                deposit_check = page.locator(
                    "text=/поповнити рахунок|поповнити|каса|deposit|пополнить/i"
                ).first
                if await deposit_check.is_visible(timeout=3000):
                    cookies_valid = True
                    auth_action = "login_via_cookies"

        if not cookies_valid:
            await perform_registration(page, context, domain, auth_method, task_id)
            auth_action = "registration"

        shot_auth = await capture_screenshot(page, task_id, "step4")
        await TaskRepository.add_history_entry(
            task_id, "AUTH_OR_REGISTER", "SUCCESS", {"action": auth_action, "screenshot": shot_auth}
        )

        # --- 3. NAVIGATE TO CASHIER ---
        await TaskRepository.add_history_entry(task_id, "NAVIGATE_TO_CASHIER", "STARTED")

        deposit_btn = page.locator(
            "text=/поповнити рахунок|поповнити|каса|deposit|пополнить|пополнить счёт|Mono|Privat24/i"
        ).first
        await deposit_btn.click(force=True, timeout=5000)
        await page.wait_for_timeout(3000)

        card_clicked = False
        for frame in page.frames:
            try:
                target = frame.get_by_text("Перевод на карту", exact=False).first
                if await target.is_visible(timeout=1000):
                    await target.click(force=True)
                    card_clicked = True
                    await page.wait_for_timeout(3000)
                    break
            except Exception:
                continue

        if not card_clicked:
            for frame in page.frames:
                try:
                    cell = frame.locator("[data-testid='modulor-list-cell']").filter(
                        has_text="Переказ на картку"
                    ).first
                    if await cell.is_visible(timeout=1000):
                        await cell.click(force=True)
                        card_clicked = True
                        break
                except Exception:
                    continue

        shot_way = await capture_screenshot(page, task_id, "deposit_way_chosen")
        await TaskRepository.add_history_entry(
            task_id, "NAVIGATE_TO_CASHIER", "SUCCESS", {"screenshot": shot_way}
        )

        # --- 4. GENERATE INVOICE ---
        await TaskRepository.add_history_entry(task_id, "GENERATE_INVOICE", "STARTED")
        await submit_deposit(page, str(int(target_amount)))
        await page.wait_for_timeout(10000)

        invoice_shot = await capture_screenshot(page, task_id, "payment_gateway")
        await TaskRepository.add_history_entry(
            task_id, "GENERATE_INVOICE", "SUCCESS", {"screenshot": invoice_shot}
        )

        # --- 5. PARSING REQUISITES ---
        await TaskRepository.add_history_entry(task_id, "PARSING_REQUISITES", "STARTED")
        card_num = await get_card_number(page)

        final_result = {}
        if card_num:
            final_result["card_number"] = card_num

        if not final_result.get("card_number"):
            shot_bytes = await page.screenshot(full_page=True)
            ocr_res = await RequisitiveParser.extract_from_screenshot(shot_bytes)
            if ocr_res.get("card_number"):
                final_result["card_number"] = ocr_res["card_number"]

        final_result["screenshot"] = invoice_shot

        if not final_result.get("card_number"):
            await TaskRepository.add_history_entry(
                task_id, "PARSING_REQUISITES", "FAILED", {"reason": "Card not found", "screenshot": invoice_shot}
            )
            await browser.close()
            raise RuntimeError("Не удалось извлечь номер карты для оплаты")

        await TaskRepository.add_history_entry(
            task_id, "PARSING_REQUISITES", "SUCCESS", final_result
        )

        await browser.close()
        return final_result