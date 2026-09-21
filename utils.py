import os
import random
import re
from datetime import datetime
from faker import Faker

fake = Faker("uk_UA")


def generate_synthetic_identity(domain: str) -> dict[str, str]:
    first_name = fake.first_name()
    last_name = fake.last_name()
    username = f"{fake.user_name()}_{fake.random_int(100, 9999)}"
    email = f"{username}@{domain}"
    password = f"P@ss_{fake.password(length=10, special_chars=True)}"

    return {
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "email": email,
        "password": password,
    }


async def human_type(page, locator, text: str, min_delay: int = 40, max_delay: int = 160):
    """Имитация печати человеком с разной скоростью нажатия клавиш"""
    await locator.click()
    await page.wait_for_timeout(random.randint(200, 500))
    for char in text:
        await page.keyboard.type(char)
        await page.wait_for_timeout(random.randint(min_delay, max_delay))

async def human_click(page, locator):
    """Наведение мыши с паузой перед кликом"""
    box = await locator.bounding_box()
    if box:
        # Небольшой случайный сдвиг относительно центра элемента
        x = box["x"] + box["width"] / 2 + random.randint(-5, 5)
        y = box["y"] + box["height"] / 2 + random.randint(-5, 5)
        await page.mouse.move(x, y, steps=random.randint(5, 15))
        await page.wait_for_timeout(random.randint(150, 400))
        await page.mouse.down()
        await page.wait_for_timeout(random.randint(50, 120))
        await page.mouse.up()
    else:
        await locator.click(force=True)

async def fill_first_available(page, locators: list[str], value: str, timeout: int = 3000) -> None:
    for sel in locators:
        loc = page.locator(sel)
        try:
            await loc.wait_for(state="visible", timeout=timeout)
            await human_type(page, loc, value)
            return
        except Exception:
            continue
    raise Exception("Ни один из локаторов не найден видимым")


async def capture_screenshot(page, task_id, step_name: str) -> str:
    """Фиксация скриншотов на каждом шаге"""
    os.makedirs("screenshots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"screenshots/{task_id}_{step_name}_{timestamp}.png"
    try:
        await page.screenshot(path=file_path, full_page=True)
        return file_path
    except Exception:
        return ""