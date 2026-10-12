import os
import sys
import shutil
import logging
import subprocess
import re
from typing import List, Dict, Any, Optional, Tuple
from faster_whisper import WhisperModel
import yt_dlp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def format_ass_timestamp(seconds: float) -> str:
    """Конвертація секунд у точний формат таймкоду ASS: H:MM:SS.cs"""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        secs += 1
        centis = 0
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


def get_stream_audio_transcript_quick(
    url: str,
    max_duration_seconds: int = 3600,
    model_size: str = "base",
    device: str = "cuda",
    compute_type: str = "float16",
    temp_dir: str = "./temp_work",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Завантажує оптимізоване аудіо стріму або його фрагмента та транскрибує його
    для первинного аналізу Gemini.
    """
    os.makedirs(temp_dir, exist_ok=True)
    audio_path = os.path.join(temp_dir, "overview_audio.m4a")
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            pass

    # yt-dlp опції для безпечного та швидкого зчитування аудіо
    ydl_opts = {
        "format": "ba[ext=m4a]/ba/b",
        "outtmpl": audio_path,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "download_ranges": yt_dlp.utils.download_range_func(None, [(0, max_duration_seconds)]),
        "force_keyframes_at_cuts": True,
    }

    logger.info(f"Отримання аудіодоріжки стріму (ліміт: {max_duration_seconds} сек)...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error(f"Помилка завантаження аудіо через yt-dlp: {e}")
        raise RuntimeError(f"Не вдалося завантажити аудіодоріжку зі стріму: {e}")

    # Перевірка наявності отриманого файлу (якщо розширення відрізняється)
    if not os.path.exists(audio_path):
        base_name, _ = os.path.splitext(audio_path)
        for ext in [".m4a", ".mp3", ".webm", ".opus", ".aac"]:
            cand = f"{base_name}{ext}"
            if os.path.exists(cand):
                audio_path = cand
                break

    if not os.path.exists(audio_path):
        raise FileNotFoundError("Аудіофайл не знайдено після завантаження через yt-dlp.")

    logger.info(f"Первинна транскрипція Faster-Whisper ({model_size} на {device})...")
    actual_device = device if device == "cuda" else "cpu"
    actual_compute = compute_type if device == "cuda" else "int8"

    try:
        model = WhisperModel(model_size, device=actual_device, compute_type=actual_compute)
    except Exception as e:
        logger.warning(f"CUDA недоступна для Whisper, перемикаємося на CPU (int8): {e}")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, _ = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    transcript_lines = []
    raw_segments = []
    for s in segments:
        text = s.text.strip()
        if text:
            line = f"[{s.start:.1f} - {s.end:.1f}] {text}"
            transcript_lines.append(line)
            raw_segments.append({"start": s.start, "end": s.end, "text": text})

    full_text = "\n".join(transcript_lines)
    return full_text, raw_segments


def download_precise_video_clip(
    url: str,
    start_time: float,
    end_time: float,
    output_path: str,
) -> str:
    """
    Завантажує ТОЧНИЙ фрагмент стріму через yt-dlp без завантаження всього відео.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass

    ydl_opts = {
        "format": "bv*[ext=mp4]+ba*[ext=m4a]/b[ext=mp4]/best",
        "outtmpl": output_path,
        "nocheckcertificate": True,
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        "force_keyframes_at_cuts": True,
        "quiet": False,
        "no_warnings": True,
    }

    logger.info(f"Завантаження фрагмента {start_time:.1f}s -> {end_time:.1f}s через yt-dlp...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error(f"Помилка завантаження фрагмента: {e}")
        raise RuntimeError(f"Не вдалося завантажити фрагмент стріму: {e}")

    if not os.path.exists(output_path):
        base_name, _ = os.path.splitext(output_path)
        for ext in [".mp4", ".mkv", ".webm"]:
            candidate = f"{base_name}{ext}"
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Файл {output_path} не знайдено після завантаження.")

    return output_path


def transcribe_clip_words(
    media_path: str,
    model_size: str = "base",
    device: str = "cuda",
    compute_type: str = "float16",
) -> List[Dict[str, Any]]:
    """
    Детальна транскрипція вже нарізаного кліпу на рівні окремих слів (word-level timestamps).
    """
    actual_device = device if device == "cuda" else "cpu"
    actual_compute = compute_type if device == "cuda" else "int8"

    try:
        model = WhisperModel(model_size, device=actual_device, compute_type=actual_compute)
    except Exception:
        logger.warning("CUDA недоступна, запуск Whisper на CPU")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, _ = model.transcribe(
        media_path,
        word_timestamps=True,
        vad_filter=True,
    )

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                clean_w = w.word.strip()
                if clean_w:
                    words.append({
                        "start": w.start,
                        "end": w.end,
                        "word": clean_w,
                        "probability": w.probability,
                    })
        else:
            clean_s = segment.text.strip()
            if clean_s:
                words.append({
                    "start": segment.start,
                    "end": segment.end,
                    "word": clean_s,
                    "probability": 1.0,
                })
    return words


# Кольорові палітри для субтитрів (ASS PrimaryColour у форматі &HAABBGGRR)
SUBTITLE_PRESETS = {
    "TikTok Жовтий (Неон)": {
        "active_color": "&H0000FFFF",   # BGR: Yellow
        "normal_color": "&H00FFFFFF",   # White
        "font_name": "Impact",
        "font_size": 66,
        "outline": 6,
        "shadow": 3,
    },
    "Cyberpunk Зелений": {
        "active_color": "&H0000FF00",   # Green
        "normal_color": "&H00FFFFFF",
        "font_name": "Impact",
        "font_size": 66,
        "outline": 6,
        "shadow": 3,
    },
    "Instagram Блакитний": {
        "active_color": "&H00FFFF00",   # Cyan
        "normal_color": "&H00FFFFFF",
        "font_name": "Impact",
        "font_size": 66,
        "outline": 6,
        "shadow": 3,
    },
    "Вогняний Червоний": {
        "active_color": "&H000022FF",   # Bright Orange/Red
        "normal_color": "&H00FFFFFF",
        "font_name": "Impact",
        "font_size": 66,
        "outline": 6,
        "shadow": 3,
    },
    "Класичний Білий": {
        "active_color": "&H00FFFFFF",
        "normal_color": "&H00CCCCCC",
        "font_name": "Impact",
        "font_size": 66,
        "outline": 6,
        "shadow": 3,
    },
}


def generate_karaoke_ass(
    words: List[Dict[str, Any]],
    output_ass_path: str,
    hook_banner: Optional[str] = None,
    words_per_group: int = 3,
    style_preset: str = "TikTok Жовтий (Неон)",
    video_width: int = 1080,
    video_height: int = 1920,
) -> str:
    """
    Генерує файл стилізованих ASS субтитрів:
    - Вибір палітри кольорів
    - Вирівнювання по центру екрана (Alignment 5)
    - Караоке-підсвічування активного слова з легким збільшенням (zoom 115%)
    - Додатковий верхній банер (Hook Banner) для утримання уваги (Alignment 8)
    """
    cfg = SUBTITLE_PRESETS.get(style_preset, SUBTITLE_PRESETS["TikTok Жовтий (Неон)"])
    act_col = cfg["active_color"]
    norm_col = cfg["normal_color"]
    fname = cfg["font_name"]
    fsize = cfg["font_size"]
    outline = cfg["outline"]
    shadow = cfg["shadow"]

    ass_header = f"""[Script Info]
Title: Viral Shorts Dynamic Karaoke Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {video_width}
PlayResY: {video_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KaraokeActive,{fname},{fsize},{act_col},{norm_col},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},5,40,40,240,1
Style: KaraokeNormal,{fname},{fsize},{norm_col},{act_col},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},5,40,40,240,1
Style: HookBanner,{fname},54,&H0000FFFF,&H00FFFFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,1,5,3,8,40,40,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    # Додаємо вірусний заголовок-хук у верхній частині екрана на перші 5 секунд
    if hook_banner and hook_banner.strip():
        banner_clean = hook_banner.strip().replace("{", "").replace("}", "")
        events.append(f"Dialogue: 1,0:00:00.00,0:00:05.50,HookBanner,,0,0,0,,{{\\fade(200,400)}}🔥 {banner_clean}")

    # Караоке-субтитри
    for i in range(0, len(words), words_per_group):
        group = words[i : i + words_per_group]
        if not group:
            continue

        for current_idx, current_word in enumerate(group):
            w_start = format_ass_timestamp(current_word["start"])
            w_end = format_ass_timestamp(current_word["end"])

            text_parts = []
            for j, w in enumerate(group):
                word_clean = w["word"].replace("{", "").replace("}", "")
                if j == current_idx:
                    # Підсвічене активне слово
                    text_parts.append(f"{{\\c{act_col}\\fscx115\\fscy115}}{word_clean}{{\\r}}")
                else:
                    # Звичайне слово
                    text_parts.append(f"{{\\c{norm_col}}}{word_clean}{{\\r}}")

            full_line = " ".join(text_parts)
            events.append(f"Dialogue: 0,{w_start},{w_end},KaraokeNormal,,0,0,0,,{full_line}")

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "\n".join(events) + "\n")

    return output_ass_path


def render_vertical_video(
    input_video: str,
    output_video: str,
    subtitles_ass_path: Optional[str] = None,
    crop_mode: str = "Blur Background (Розмитий фон)",
    use_nvenc: bool = True,
    audio_normalize: bool = True,
) -> str:
    """
    Високоякісний рендер FFmpeg:
    1. Режими кадрування:
       - "Blur Background": красивий розмитий фон 9:16 + по центру відео
       - "Smart Center Crop": заповнення всього екрана 1080x1920
    2. Аудіо-нормалізація (loudnorm): гучний та чіткий звук голосу
    3. Апаратне кодування h264_nvenc для GPU або софтверний libx264
    4. Вшивання стилізованих ASS субтитрів із безпечним екрануванням шляхів
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    if os.path.exists(output_video):
        try:
            os.remove(output_video)
        except OSError:
            pass

    escaped_ass = ""
    if subtitles_ass_path and os.path.exists(subtitles_ass_path):
        clean_path = os.path.abspath(subtitles_ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        escaped_ass = f",ass='{clean_path}'"

    if "Smart Center Crop" in crop_mode:
        filter_complex = (
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920{escaped_ass}[v]"
        )
    else:
        # Blur background (оптимально для ігрових стрімів)
        filter_complex = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg_blur];"
            f"[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fg_scaled];"
            f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2{escaped_ass}[v]"
        )

    # Аудіо фільтр для нормалізації гучності
    audio_filters = []
    if audio_normalize:
        audio_filters.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])

    encoders_to_try = []
    if use_nvenc:
        encoders_to_try.append(("h264_nvenc", ["-preset", "p4", "-cq", "21"]))
    encoders_to_try.append(("libx264", ["-preset", "fast", "-crf", "22"]))

    for encoder, extra_args in encoders_to_try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "0:a?",
            *audio_filters,
            "-c:v", encoder,
            *extra_args,
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            output_video,
        ]

        logger.info(f"Запуск FFmpeg ({encoder}, режим: {crop_mode})...")
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            logger.info(f"Відео успішно відрендерено: {output_video}")
            return output_video
        except subprocess.CalledProcessError as e:
            logger.warning(f"Помилка рендеру через {encoder}: {e.stderr[:300]}")
            if encoder == "h264_nvenc":
                logger.info("Спроба рендеру на CPU (libx264)...")
                continue
            raise RuntimeError(f"FFmpeg рендер завершився помилкою: {e.stderr}")

    raise RuntimeError("Не вдалося відрендерити відео жодним доступним кодеком.")
