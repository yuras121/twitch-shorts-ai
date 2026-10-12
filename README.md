# 🎬 Stream to Shorts/TikTok Studio Pro (AI)

Модульний та автономний пайплайн для автоматичного перетворення довгих трансляцій (Twitch/YouTube VODs) у вертикальні (9:16) ролики для YouTube Shorts / TikTok / Instagram Reels за допомогою **Google Gemini API**, **Faster-Whisper** та **FFmpeg (NVENC)**.

---

## ⚡ Нові можливості версії Studio Pro:
1. **Мульти-пошук (ТОП-1...ТОП-5 моментів)**: Gemini знаходить кілька найкращих фрагментів із рейтингом вірусності (1-100), категорією (`funny`, `hype`, `rage`, `podcast`) та детальним поясненням.
2. **Фільтр настрою (Mood Filter)**: Можливість шукати спеціально кумедні меми, епічні клатчі, емоційні рейджі або цікаві розмови.
3. **Двоетапний пайплайн (Interactive Workflow)**:
   - *Крок 1*: Швидкий AI-аналіз стріму та отримання списку моментів (без витрати часу на рендер усього відео).
   - *Крок 2*: Вибір конкретного моменту зі списку ➔ вибір стилю ➔ миттєва точкова нарізка і рендер.
4. **Режими кадрування 9:16**:
   - `Blur Background`: естетичний розмитий фон стріму позаду + чітке центроване відео.
   - `Smart Center Crop`: повноекранне вертикальне заповнення 1080x1920.
5. **Дизайнерські стилі караоке-субтитрів**:
   - TikTok Неон (жовтий активний)
   - Cyberpunk (зелений активний)
   - Instagram (блакитний активний)
   - Класичний білий
   - Налаштування кількості слів на екрані (1–5 слів).

---

## 📁 Структура файлів

* [`requirements.txt`](file:///C:/Users/admin/.gemini/antigravity/scratch/stream-shorts-pipeline/requirements.txt): Стабільні бібліотеки.
* [`gemini_analyzer.py`](file:///C:/Users/admin/.gemini/antigravity/scratch/stream-shorts-pipeline/gemini_analyzer.py): AI модуль аналізу, рейтингу вірусності та фільтрації настрою.
* [`video_pipeline.py`](file:///C:/Users/admin/.gemini/antigravity/scratch/stream-shorts-pipeline/video_pipeline.py): Завантаження через yt-dlp, word-level Whisper, генератор караоке ASS та FFmpeg NVENC рендерер.
* [`app.py`](file:///C:/Users/admin/.gemini/antigravity/scratch/stream-shorts-pipeline/app.py): Двоетапний веб-інтерфейс Gradio Studio Pro.
* [`colab_setup.ipynb`](file:///C:/Users/admin/.gemini/antigravity/scratch/stream-shorts-pipeline/colab_setup.ipynb): Готовий блокнот для запуску на Google Colab T4 GPU.

---

## 🚀 Запуск у Google Colab (T4 GPU)

1. Відкрийте [Google Colab](https://colab.research.google.com/) та переконайтеся, що вибрано **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
2. Запустіть комірку:
```bash
# 1. Монтування Google Drive
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/Twitch_Shorts

# 2. Встановлення FFmpeg та шрифтів
!apt-get update -qq && apt-get install -y -qq ffmpeg fonts-impact

# 3. Клонування репозиторію та встановлення залежностей
!git clone https://github.com/yuras121/twitch-shorts-ai.git
%cd twitch-shorts-ai
!pip install -q -r requirements.txt

# 4. Запуск веб-інтерфейсу з публічним посиланням
!python app.py
```
3. Відкрийте посилання `https://xxxx.gradio.live`, вставте посилання на стрім, натисніть **«1. Знайти хайлайти AI»**, оберіть найкращий момент і натисніть **«2. Згенерувати вертикальний кліп 9:16»**.
