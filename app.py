import os
import sys
import shutil
import logging
import traceback
import gradio as gr
from dotenv import load_dotenv

from gemini_analyzer import analyze_transcript_for_multiple_highlights, HighlightResult
from video_pipeline import (
    get_stream_audio_transcript_quick,
    download_precise_video_clip,
    transcribe_clip_words,
    generate_karaoke_ass,
    render_vertical_video,
    SUBTITLE_PRESETS,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_COLAB_DRIVE_DIR = "/content/drive/MyDrive/Twitch_Shorts"
LOCAL_OUTPUT_DIR = "./output_shorts"
TEMP_DIR = "./temp_pipeline"

# Кеш поточного стану для швидкої інтерактивної роботи
CACHED_STATE = {
    "url": "",
    "highlights": [],
    "transcript": "",
}


def scan_stream_for_highlights(
    stream_url: str,
    gemini_key: str,
    duration_min: int,
    duration_max: int,
    scan_limit_minutes: int,
    top_k: int,
    highlight_mood: str,
    whisper_model: str,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Крок 1: Швидкий AI-аналіз стріму та пошук ТОП-моментів без рендеру.
    """
    if not stream_url or not stream_url.strip():
        raise gr.Error("Введіть коректне посилання на стрім (Twitch або YouTube)!")

    api_key = gemini_key.strip() if gemini_key else os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise gr.Error("Gemini API Key обов'язковий! Введіть його у відповідне поле.")

    os.makedirs(TEMP_DIR, exist_ok=True)

    progress(0.15, desc=f"1/2: Whisper аналізує звук (перші {scan_limit_minutes} хв)...")
    transcript_text, _ = get_stream_audio_transcript_quick(
        url=stream_url.strip(),
        max_duration_seconds=int(scan_limit_minutes * 60),
        model_size=whisper_model,
        temp_dir=TEMP_DIR,
    )

    if not transcript_text.strip():
        raise gr.Error("Не вдалося розпізнати мову (можливо, звук відсутній або захищений).")

    progress(0.65, desc=f"2/2: Gemini AI відбирає ТОП-{top_k} вірусних моментів...")
    highlights = analyze_transcript_for_multiple_highlights(
        transcript_text=transcript_text,
        target_duration_min=duration_min,
        target_duration_max=duration_max,
        top_k=int(top_k),
        highlight_mood=highlight_mood,
        api_key=api_key,
    )

    CACHED_STATE["url"] = stream_url.strip()
    CACHED_STATE["highlights"] = highlights
    CACHED_STATE["transcript"] = transcript_text

    report_lines = [f"🔥 ЗНАЙДЕНО {len(highlights)} ВІРУСНИХ МОМЕНТІВ:\n"]
    dropdown_choices = []

    for i, h in enumerate(highlights, 1):
        dur = h.end_time - h.start_time
        tags_str = " ".join(h.recommended_tags) if h.recommended_tags else "#shorts #viral"
        line = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Момент #{i} | Категорія: [{h.category.upper()}] | Рейтинг: {h.viral_score}/100\n"
            f"🎯 Хук: \"{h.hook_title}\"\n"
            f"⏱ Таймкод: {h.start_time:.1f}s - {h.end_time:.1f}s ({dur:.1f} сек)\n"
            f"💡 Чому завіруситься: {h.reason}\n"
            f"🏷 Рекомендовані теги: {tags_str}\n"
        )
        report_lines.append(line)
        dropdown_choices.append(f"Момент #{i}: {h.hook_title} ({dur:.0f}s, {h.category})")

    progress(1.0, desc="Аналіз готовий! Оберіть момент нижче для створення відео.")
    return (
        "\n".join(report_lines),
        gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None, interactive=True),
    )


def generate_single_short(
    selected_choice: str,
    crop_mode: str,
    subtitle_style: str,
    words_per_subtitle: int,
    show_hook_banner: bool,
    enable_loudnorm: bool,
    whisper_model: str,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Крок 2: Точкова нарізка обраного фрагмента, вшивання субтитрів та рендер 9:16.
    """
    if not selected_choice or not CACHED_STATE["highlights"]:
        raise gr.Error("Спочатку натисніть '🔍 1. Знайти хайлайти AI' та оберіть момент!")

    idx = 0
    try:
        idx = int(selected_choice.split(":")[0].replace("Момент #", "").strip()) - 1
    except Exception:
        idx = 0

    h = CACHED_STATE["highlights"][idx]
    stream_url = CACHED_STATE["url"]

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    progress(0.2, desc="1/3: Завантаження точного шматка відео через yt-dlp...")
    clip_raw_path = os.path.join(TEMP_DIR, f"clip_{h.id}.mp4")
    clip_downloaded = download_precise_video_clip(
        url=stream_url,
        start_time=h.start_time,
        end_time=h.end_time,
        output_path=clip_raw_path,
    )

    progress(0.5, desc="2/3: Транскрипція слів та караоке-субтитри...")
    words = transcribe_clip_words(clip_downloaded, model_size=whisper_model)
    ass_path = os.path.join(TEMP_DIR, f"subs_{h.id}.ass")

    banner_text = h.hook_title if show_hook_banner else None
    generate_karaoke_ass(
        words=words,
        output_ass_path=ass_path,
        hook_banner=banner_text,
        words_per_group=words_per_subtitle,
        style_preset=subtitle_style,
    )

    progress(0.8, desc=f"3/3: FFmpeg 9:16 рендер ({crop_mode})...")
    out_filename = f"short_{h.category}_{int(h.start_time)}_{int(h.end_time)}.mp4"
    final_video = os.path.join(LOCAL_OUTPUT_DIR, out_filename)

    render_vertical_video(
        input_video=clip_downloaded,
        output_video=final_video,
        subtitles_ass_path=ass_path,
        crop_mode=crop_mode,
        use_nvenc=True,
        audio_normalize=enable_loudnorm,
    )

    drive_status = ""
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(DEFAULT_COLAB_DRIVE_DIR, exist_ok=True)
        drive_dest = os.path.join(DEFAULT_COLAB_DRIVE_DIR, out_filename)
        shutil.copy2(final_video, drive_dest)
        drive_status = f"\n💾 Автоматично збережено на Google Drive:\n{drive_dest}"

    tags_str = " ".join(h.recommended_tags) if h.recommended_tags else "#shorts #viral #gaming"
    post_caption = (
        f"📱 ГОТОВО ДО ПУБЛІКАЦІЇ:\n"
        f"Заголовок: {h.hook_title}\n"
        f"Опис / Теги: {tags_str}\n"
        f"Категорія: {h.category} | Рейтинг вірусності: {h.viral_score}/100\n"
        f"{drive_status}"
    )

    progress(1.0, desc="Готово! Відео відрендерено.")
    return final_video, post_caption


def build_app():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple"), title="Stream to Viral Shorts Studio Pro") as demo:
        gr.Markdown(
            """
            # 🎬 Stream to Viral Shorts / TikTok Studio Pro
            ### Повний автономний пайплайн: Faster-Whisper + Gemini 1.5 + yt-dlp + FFmpeg NVENC (9:16)
            """
        )

        with gr.Row():
            # Ліва колонка: Джерело та параметри аналізу
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ Джерело та AI Аналітика")
                stream_url_input = gr.Textbox(
                    label="🔗 Посилання на VOD (Twitch або YouTube)",
                    placeholder="https://www.youtube.com/watch?v=... або https://www.twitch.tv/videos/...",
                    lines=1,
                )
                gemini_key_input = gr.Textbox(
                    label="🔑 Gemini API Key (якщо не вказано в .env)",
                    placeholder="AIzaSy...",
                    type="password",
                )

                with gr.Row():
                    mood_input = gr.Dropdown(
                        choices=[
                            "Будь-який (найкращий)",
                            "Смішний / Кумедний (Humor/Meme)",
                            "Хайповий / Епічний (Hype/Clutch)",
                            "Емоційний / Рейдж (Rage/Fail)",
                            "Цікава розмова / Інсайд (Podcast/Story)",
                        ],
                        value="Будь-який (найкращий)",
                        label="🎭 Настрій моменту (Mood)",
                    )
                    top_k_input = gr.Slider(
                        minimum=1, maximum=5, value=3, step=1, label="Скільки моментів шукати (ТОП-K)"
                    )

                with gr.Accordion("⚙️ Тонкі налаштування сканування", open=False):
                    with gr.Row():
                        min_slider = gr.Slider(
                            minimum=15, maximum=45, value=30, step=5, label="Мін. тривалість кліпу (с)"
                        )
                        max_slider = gr.Slider(
                            minimum=45, maximum=90, value=60, step=5, label="Макс. тривалість кліпу (с)"
                        )
                    scan_limit = gr.Slider(
                        minimum=15, maximum=240, value=60, step=15, label="Ліміт сканування стріму (хвилин)"
                    )
                    whisper_choice = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium"],
                        value="base",
                        label="Модель Faster-Whisper",
                    )

                find_btn = gr.Button("🔍 1. Знайти хайлайти AI", variant="secondary", size="lg")

            # Права колонка: Вибір хайлайту, оформлення та запуск рендеру
            with gr.Column(scale=1):
                gr.Markdown("### 2️⃣ Вибір моменту та стилізація")
                highlights_report = gr.Textbox(
                    label="📋 Знайдені хайлайти (AI Аналітика)",
                    lines=8,
                    placeholder="Тут з'явиться детальний звіт із рейтингом вірусності, тегами та хуками...",
                )
                moment_selector = gr.Dropdown(
                    label="🎯 Оберіть момент для створення відео",
                    choices=[],
                    interactive=True,
                )

                with gr.Row():
                    crop_choice = gr.Radio(
                        choices=["Blur Background (Розмитий фон)", "Smart Center Crop (Кроп на весь екран)"],
                        value="Blur Background (Розмитий фон)",
                        label="📐 Формат кадру (9:16)",
                    )
                    sub_style = gr.Dropdown(
                        choices=list(SUBTITLE_PRESETS.keys()),
                        value="TikTok Жовтий (Неон)",
                        label="🎨 Стиль караоке-субтитрів",
                    )

                with gr.Row():
                    words_group = gr.Slider(
                        minimum=1, maximum=5, value=3, step=1, label="Слів на екрані одночасно"
                    )
                    show_banner = gr.Checkbox(
                        label="🔥 Додати хук-банер угорі (перші 5 сек)",
                        value=True,
                    )
                    audio_norm = gr.Checkbox(
                        label="🔊 Нормалізація звуку голосу (Loudnorm)",
                        value=True,
                    )

                render_btn = gr.Button("🚀 2. Згенерувати вертикальний Shorts 9:16", variant="primary", size="lg")

        gr.Markdown("---")
        with gr.Row():
            video_output = gr.Video(label="📱 Готовий вертикальний Shorts (9:16)")
            final_status = gr.Textbox(label="📊 Готовий опис для соцмереж та шлях збереження", lines=6)

        # Події
        find_btn.click(
            fn=scan_stream_for_highlights,
            inputs=[
                stream_url_input,
                gemini_key_input,
                min_slider,
                max_slider,
                scan_limit,
                top_k_input,
                mood_input,
                whisper_choice,
            ],
            outputs=[highlights_report, moment_selector],
        )

        render_btn.click(
            fn=generate_single_short,
            inputs=[
                moment_selector,
                crop_choice,
                sub_style,
                words_group,
                show_banner,
                audio_norm,
                whisper_choice,
            ],
            outputs=[video_output, final_status],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
