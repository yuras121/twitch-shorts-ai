import os
import json
import logging
import re
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class HighlightResult(BaseModel):
    id: int = Field(default=1, description="Порядковий номер моменту")
    start_time: float = Field(..., description="Початок кліпу в секундах")
    end_time: float = Field(..., description="Кінець кліпу в секундах")
    hook_title: str = Field(..., description="Вірусний заголовок / хук для Shorts/TikTok")
    category: str = Field(default="funny", description="Категорія (funny, hype, rage, fail, talk, emotional)")
    reason: str = Field(..., description="Чому саме цей момент має вірусний потенціал")
    viral_score: int = Field(default=90, description="Оцінка вірусності від 1 до 100")
    recommended_tags: List[str] = Field(default_factory=list, description="Хештеги для публікації (#shorts, #twitch тощо)")


def init_gemini(api_key: Optional[str] = None) -> None:
    """Ініціалізація Gemini API клієнта з ключем."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "Gemini API Key не знайдено! Введіть ключ у веб-інтерфейсі або додайте у файл .env як GEMINI_API_KEY."
        )
    genai.configure(api_key=key.strip())


def clean_json_text(text: str) -> str:
    """Очищення тексту від можливих markdown обгорток ```json ... ```"""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def analyze_transcript_for_multiple_highlights(
    transcript_text: str,
    target_duration_min: int = 30,
    target_duration_max: int = 60,
    top_k: int = 3,
    highlight_mood: str = "Будь-який (найкращий)",
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash",
) -> List[HighlightResult]:
    """
    Глибокий аналіз транскрипту стріму за допомогою Gemini API:
    - Пошук ТОП-K моментів без перетинів таймкодів
    - Перевірка логічних меж речень
    - Рейтинг вірусності (1-100), категорія та вірусні хештеги
    - Розумний парсинг і валідація тривалості
    """
    init_gemini(api_key)

    if not transcript_text or not transcript_text.strip():
        raise ValueError("Транскрипт порожній! Немає тексту для аналізу.")

    mood_instructions = ""
    if highlight_mood and highlight_mood != "Будь-який (найкращий)":
        mood_instructions = f"Пріоритетно обирай моменти в тональності / стилі: '{highlight_mood}'. "

    system_instruction = (
        "Ти — експерт №1 у сфері вірусного відеовиробництва для YouTube Shorts, TikTok та Reels. "
        "Твоє завдання: детально проаналізувати транскрипт аудіо стріму з таймкодами та відібрати "
        f"ТОП-{top_k} найбільш захоплюючих моментів хронометражем від {target_duration_min} до {target_duration_max} секунд.\n\n"
        f"{mood_instructions}"
        "Суворі правила відбору:\n"
        "1. Чіткий Hook у перші 3 секунди: глядач має зацікавитися миттєво.\n"
        "2. ЖОДНИХ обривів фраз: початок кліпу повинен збігатися з початком речення, а кінець — із завершеною думкою або кульмінацією.\n"
        "3. Жодних перетинів: відрізки часу [start_time, end_time] не повинні перекривати один одного.\n"
        f"4. Суворе дотримання хронометражу: різниця (end_time - start_time) суворо між {target_duration_min} та {target_duration_max} сек.\n"
        "5. Формат відповіді: ТІЛЬКИ чистий JSON масив об'єктів з ключами: "
        "id, start_time, end_time, hook_title, category, reason, viral_score, recommended_tags."
    )

    prompt = f"""
Проаналізуй поданий нижче транскрипт стріму:
\"\"\"{transcript_text}\"\"\"

Поверни валідний JSON-масив ТОП-{top_k} моментів за схемою:
[
  {{
    "id": 1,
    "start_time": 120.5,
    "end_time": 165.0,
    "hook_title": "Шокуючий момент у грі!",
    "category": "funny / hype / rage / fail / talk / emotional",
    "reason": "Чому саме цей фрагмент викличе максимальний інтерес та коментарі",
    "viral_score": 95,
    "recommended_tags": ["#shorts", "#twitch", "#gaming", "#viral"]
  }}
]
"""

    generation_config = genai.types.GenerationConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )

    # Пробуємо основну модель, з фолбеком на gemini-1.5-pro якщо потрібно
    models_to_try = [model_name, "gemini-1.5-flash", "gemini-1.5-pro"]
    last_err = None

    for m_name in dict.fromkeys(models_to_try):
        try:
            model = genai.GenerativeModel(
                model_name=m_name,
                system_instruction=system_instruction,
            )
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            raw_content = clean_json_text(response.text)
            data = json.loads(raw_content)

            if isinstance(data, dict):
                # Якщо модель повернула об'єкт {"highlights": [...]} або один об'єкт
                if "highlights" in data and isinstance(data["highlights"], list):
                    data = data["highlights"]
                elif "moments" in data and isinstance(data["moments"], list):
                    data = data["moments"]
                else:
                    data = [data]

            results: List[HighlightResult] = []
            for item in data:
                try:
                    res = HighlightResult(**item)
                    dur = res.end_time - res.start_time
                    if dur <= 5:  # надто короткий
                        continue
                    # Безпечне коректування тривалості
                    if dur < target_duration_min:
                        res.end_time = res.start_time + target_duration_min
                    results.append(res)
                except Exception as parse_item_err:
                    logger.warning(f"Пропуск невалідного моменту: {parse_item_err}")

            if not results:
                continue

            # Сортування за оцінкою вірусності
            results.sort(key=lambda x: x.viral_score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.warning(f"Помилка генерації через модель {m_name}: {e}")
            last_err = e

    raise RuntimeError(f"Не вдалося отримати аналіз від Gemini API: {last_err}")
