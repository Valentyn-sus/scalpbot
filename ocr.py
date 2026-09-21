import io
import re
from typing import Any
import pytesseract
from PIL import Image


class RequisitiveParser:
    CARD_PATTERN = r"\b(?:\d[ -]*?){13,16}\b"
    IBAN_PATTERN = r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}"

    @classmethod
    def get_card_number_from_frames(cls, page) -> str | None:
        """Из вашего test.py: глубокий поиск номера карты по iframe"""
        for frame in page.frames:
            try:
                # 1. Поиск через заголовок "Номер картки"
                label = frame.get_by_text("Номер картки", exact=False).first
                if label.is_visible(timeout=1000):
                    parent_text = label.locator("xpath=..").inner_text()
                    card_number = re.sub(r"\D", "", parent_text)
                    if len(card_number) >= 16:
                        return card_number
            except Exception:
                continue

        # 2. Резервный поиск по регулярному выражению (16 цифр)
        for frame in page.frames:
            try:
                card_el = frame.locator("text=/\\d{4}\\s?\\d{4}\\s?\\d{4}\\s?\\d{4}/").first
                if card_el.is_visible(timeout=1000):
                    return re.sub(r"\D", "", card_el.inner_text())
            except Exception:
                continue

        return None

    @classmethod
    async def extract_from_screenshot(cls, screenshot_bytes: bytes) -> dict[str, Any]:
        """OCR Fallback с использованием Tesseract"""
        try:
            image = Image.open(io.BytesIO(screenshot_bytes))
            text = pytesseract.image_to_string(image, lang="eng+ukr")
            
            result = {}
            card_match = re.search(cls.CARD_PATTERN, text)
            if card_match:
                result["card_number"] = re.sub(r"\D", "", card_match.group(0))
            
            iban_match = re.search(cls.IBAN_PATTERN, text)
            if iban_match:
                result["iban"] = iban_match.group(0).replace(" ", "")

            return result
        except Exception:
            return {}